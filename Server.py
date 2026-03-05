from flask import Flask, jsonify, send_file, request, send_from_directory

import os
import json
import threading
import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical

# from flask_cors import CORS  # 1. Add this import

import os

app = Flask(__name__)
# CORS(app)

# =============================================================
# CONFIGURATION  — every tunable value lives here.
# All of these are also writable at runtime via POST /api/config
# or POST /api/restart (which rebuilds data + task queue).
# =============================================================

CONFIG = {
    # --- Identity ---
    "MODEL_ID":  "0009",   # 4-digit zero-padded model identifier
    "DATA_ID":   "2009",   # 4-digit zero-padded dataset identifier

    # --- Fleet control ---
    "MAX_CLIENTS":          3,      # devices recruited per aggregation round
    "MAX_ROUNDS":           5,      # server stops cleanly after this many rounds
    "REPETITIVE_TRAINING":  True,   # allow same device_id across rounds

    # --- Data ---
    "N_BINS":        10,    # number of data segments / bins
    "ITEMS_PER_BIN": 6000,  # samples per bin

    # --- Training hyperparameters (forwarded to every device in the task JSON) ---
    "NUM_EPOCHS":  2,
    "BATCH_SIZE":  100,
    "INPUT_SHAPE": [28, 28],
    "NUM_CLASSES": 10,
    "INPUT_TENSOR_NAME":  {"x": "FloatBuffer"},
    "OUTPUT_TENSOR_NAME": {"loss": "FloatBuffer", "output": "FloatBuffer"},

    # --- UI / display extras (not used server-side, echoed to dashboard) ---
    "DATASET":      "Fashion-MNIST",
    "ARCHITECTURE": "MobileNet",
}

# =============================================================


# --- DIRECTORY SETUP ---
BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR     = os.path.join(BASE_DIR, "android_training_bins")
UPLOAD_DIR       = os.path.join(BASE_DIR, "uploads")
GLOBAL_CKPT_PATH = os.path.join(BASE_DIR, "global_model", "global.ckpt")

os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(os.path.dirname(GLOBAL_CKPT_PATH), exist_ok=True)

# Purge stale checkpoints from a previous crashed round
_stale = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".ckpt")]
for _f in _stale:
    os.remove(os.path.join(UPLOAD_DIR, _f))
if _stale:
    print(f"[!] Purged {len(_stale)} stale checkpoint(s) from uploads/ on startup.")


# =============================================================
# HIERARCHICAL ID REGISTRY
# =============================================================

class IDRegistry:
    """
    Deterministic, atomic, zero-padded ID generator.

    Hierarchy:  Model(4) -> Data(4) -> Segment(7) -> Task(7)
    task_Id = model_id(4) + data_id(4) + segment_seq(7) + task_seq(7)  [22 digits]
    """
    def __init__(self):
        self._lock             = threading.Lock()
        self._segment_counters = {}
        self._task_counters    = {}

    def reset(self):
        with self._lock:
            self._segment_counters.clear()
            self._task_counters.clear()

    def generateDataSegmentId(self, data_id: str) -> tuple:
        with self._lock:
            seq             = self._segment_counters.get(data_id, 0) + 1
            self._segment_counters[data_id] = seq
            seg_seq         = f"{seq:07d}"
            return f"{data_id}{seg_seq}", seg_seq

    def generateTaskId(self, model_id: str, data_id: str, segment_sequence: str) -> tuple:
        with self._lock:
            key      = (model_id, data_id, segment_sequence)
            seq      = self._task_counters.get(key, 0) + 1
            self._task_counters[key] = seq
            task_seq = f"{seq:07d}"
            return f"{model_id}{data_id}{segment_sequence}{task_seq}", task_seq

    def readable(self, task_Id: str) -> str:
        return f"{task_Id[0:4]}_{task_Id[4:8]}_{task_Id[8:15]}_{task_Id[15:22]}"


id_registry = IDRegistry()


# =============================================================
# SERVER STATE
# =============================================================

task_queue       = deque()
task_queue_lock  = threading.Lock()

assigned_devices = set()   # device_ids that received a task this round

server_state = {
    "round":               1,
    "aggregation_done":    False,
    "finished":            False,
    "restarting":          False,
    "segments_total":      0,
    "segments_dispatched": 0,
    "clients_uploaded":    0,
}

# In-memory structured log consumed by the dashboard
_log      = []
_log_lock = threading.Lock()


# =============================================================
# STRUCTURED LOGGER
# =============================================================

def add_log(message: str, level: str = "info"):
    prefix = {"info": "[*]", "warn": "[!]", "error": "[X]", "success": "[+]"}.get(level, "[*]")
    print(f"{prefix} {message}")
    with _log_lock:
        _log.append({
            "time":  datetime.now().strftime("%H:%M:%S"),
            "level": level,
            "msg":   message,
        })
        if len(_log) > 300:   # cap log size
            _log.pop(0)


# =============================================================
# FEDERATED AVERAGING
# =============================================================

def perform_federated_averaging():
    add_log("Federated Aggregation: Starting...", "info")
    server_state["aggregation_done"] = False

    client_files = [
        os.path.join(UPLOAD_DIR, f)
        for f in os.listdir(UPLOAD_DIR) if f.endswith(".ckpt")
    ]
    if not client_files:
        add_log("No checkpoints found to aggregate.", "warn")
        return

    reader      = tf.train.load_checkpoint(client_files[0])
    actual_keys = list(reader.get_variable_to_shape_map().keys())
    add_log(f"Detected {len(actual_keys)} weight tensors.", "info")

    weight_accumulator = {key: None for key in actual_keys}
    valid_count = 0

    for ckpt in client_files:
        try:
            reader = tf.train.load_checkpoint(ckpt)
            for key in actual_keys:
                tensor = reader.get_tensor(key)
                weight_accumulator[key] = (
                    tensor.copy() if weight_accumulator[key] is None
                    else weight_accumulator[key] + tensor
                )
            valid_count += 1
            add_log(f"Aggregated: {os.path.basename(ckpt)}", "success")
        except Exception as e:
            add_log(f"Error reading {os.path.basename(ckpt)}: {e}", "error")

    if valid_count == 0:
        add_log("ABORT: No valid checkpoints processed.", "error")
        return

    add_log(f"Averaging {valid_count} clients...", "info")
    tensor_names, tensor_values = [], []
    for key in actual_keys:
        tensor_names.append(key)
        tensor_values.append(tf.convert_to_tensor(weight_accumulator[key] / valid_count))

    tf.raw_ops.Save(
        filename=tf.constant(GLOBAL_CKPT_PATH),
        tensor_names=tf.constant(tensor_names),
        data=tensor_values,
        name="federated_save",
    )

    server_state["aggregation_done"] = True
    server_state["round"]           += 1
    add_log(f"Global model saved. Now on round {server_state['round']}.", "success")

    if server_state["round"] > CONFIG["MAX_ROUNDS"]:
        server_state["finished"] = True
        add_log(f"All {CONFIG['MAX_ROUNDS']} round(s) complete. Server finished.", "success")
        threading.Thread(target=_shutdown_server, daemon=True).start()


def _shutdown_server():
    import time, signal
    time.sleep(1)
    os.kill(os.getpid(), signal.SIGINT)


# =============================================================
# DATA BIN GENERATION + TASK QUEUE BUILD
# =============================================================

def create_android_ready_bins(n_bins, output_dir, items_per_bin, shuffle=True):
    save_path = Path(output_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    add_log("Loading Fashion-MNIST dataset...", "info")
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    images = np.concatenate((x_train, x_test), axis=0)
    labels = np.concatenate((y_train, y_test), axis=0)

    if n_bins * items_per_bin > len(images):
        raise ValueError(f"Not enough data — requested {n_bins * items_per_bin}, have {len(images)}.")

    if shuffle:
        idx    = np.arange(len(images))
        np.random.shuffle(idx)
        images = images[idx]
        labels = labels[idx]

    images = (images / 255.0).astype(np.float32)
    labels = to_categorical(labels, num_classes=CONFIG["NUM_CLASSES"]).astype(np.float32)

    bin_files = []
    for i in range(n_bins):
        s, e      = i * items_per_bin, (i + 1) * items_per_bin
        img_name  = f"images_{i:03d}.bin"
        lbl_name  = f"labels_{i:03d}.bin"
        (save_path / img_name).write_bytes(images[s:e].tobytes())
        (save_path / lbl_name).write_bytes(labels[s:e].tobytes())
        bin_files.append((img_name, lbl_name))
        if i == 0:
            add_log(f"Bin 0 -> images {(save_path/img_name).stat().st_size}B  "
                    f"labels {(save_path/lbl_name).stat().st_size}B", "info")

    add_log(f"{n_bins} bin pair(s) written to '{save_path}'", "success")
    return bin_files


def build_task_queue(bin_files: list):
    task_queue.clear()
    model_id = CONFIG["MODEL_ID"]
    data_id  = CONFIG["DATA_ID"]

    add_log(f"Building task queue — MODEL_ID={model_id}  DATA_ID={data_id}", "info")

    for img_name, lbl_name in bin_files:
        data_segment_id, seg_seq = id_registry.generateDataSegmentId(data_id)
        task_Id, task_seq        = id_registry.generateTaskId(model_id, data_id, seg_seq)

        task_queue.append({
            # IDs
            "task_Id":          task_Id,
            "task_Id_readable": id_registry.readable(task_Id),
            "model_id":         model_id,
            "data_id":          data_id,
            "data_segment_id":  data_segment_id,
            "segment_sequence": seg_seq,
            "task_sequence":    task_seq,
            # Metadata
            "taskType":               "ActiveTask",
            "task_expire_date":       (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "task_completion_status": False,
            "training_type": [
                "Image_Task", "Image_DataInitializer",
                "Image_Trainer", "Image_InferenceValidator",
            ],
            # Hyperparameters — pulled from live CONFIG at queue build time
            "CKPT_FILENAME":       "checkpoint.ckpt",
            "NUM_EPOCHS":          CONFIG["NUM_EPOCHS"],
            "BATCH_SIZE":          CONFIG["BATCH_SIZE"],
            "NUM_TRAININGS":       CONFIG["ITEMS_PER_BIN"],
            "INPUT_SHAPE":         CONFIG["INPUT_SHAPE"],
            "NUM_CLASSES":         CONFIG["NUM_CLASSES"],
            "input_tensor_name":   CONFIG["INPUT_TENSOR_NAME"],
            "output_tensor_name":  CONFIG["OUTPUT_TENSOR_NAME"],
            # File references
            "MODEL_FILENAME":        f"{model_id}_model.tflite",
            "TRAIN_IMAGES_FILENAME": img_name,
            "TRAIN_LABELS_FILENAME": lbl_name,
        })

    server_state["segments_total"]      = len(task_queue)
    server_state["segments_dispatched"] = 0
    server_state["clients_uploaded"]    = 0
    add_log(f"{len(task_queue)} task(s) queued and ready to dispatch.", "success")


def _do_restart(new_config=None):
    """Runs in a background thread: optionally updates CONFIG, rebuilds data + queue."""
    global assigned_devices
    server_state["restarting"] = True
    add_log("Server restart initiated.", "warn")

    if new_config:
        for k, v in new_config.items():
            if k in CONFIG:
                CONFIG[k] = v
                add_log(f"Config updated: {k} = {v}", "info")

    # Reset counters and state
    server_state.update({
        "round": 1, "aggregation_done": False,
        "finished": False, "segments_dispatched": 0,
        "clients_uploaded": 0,
    })
    assigned_devices = set()
    id_registry.reset()

    # Purge old uploads
    for fn in os.listdir(UPLOAD_DIR):
        if fn.endswith(".ckpt") or fn.endswith(".json"):
            os.remove(os.path.join(UPLOAD_DIR, fn))

    try:
        bin_files = create_android_ready_bins(
            n_bins=CONFIG["N_BINS"],
            output_dir=DOWNLOAD_DIR,
            items_per_bin=CONFIG["ITEMS_PER_BIN"],
        )
        build_task_queue(bin_files)
    except Exception as e:
        add_log(f"Restart failed during data build: {e}", "error")

    server_state["restarting"] = False
    add_log("Server restart complete.", "success")


# =============================================================
# DASHBOARD — serve the HTML UI
# =============================================================

@app.route("/")
def dashboard():
    """Serve the dashboard HTML from the same directory as server.py."""
    return send_from_directory(BASE_DIR, "dashboard.html")


# =============================================================
# 1. TASK DISPATCH
# =============================================================

@app.route("/api/task/current", methods=["GET"])
def get_current_task():
    device_id = request.args.get("device_id", "unknown")

    if server_state["finished"]:
        return jsonify({"error": "All training rounds complete. Server is done."}), 410
    if server_state["restarting"]:
        return jsonify({"error": "Server is restarting. Try again shortly."}), 503
    if len(assigned_devices) >= CONFIG["MAX_CLIENTS"]:
        add_log(f"Device {device_id} rejected — cap of {CONFIG['MAX_CLIENTS']} reached.", "warn")
        return jsonify({"error": f"Max client limit ({CONFIG['MAX_CLIENTS']}) reached."}), 403
    if not CONFIG["REPETITIVE_TRAINING"] and device_id in assigned_devices:
        add_log(f"Device {device_id} already assigned. Rejecting.", "warn")
        return jsonify({"error": "Task already completed by this device."}), 403

    with task_queue_lock:
        if not task_queue:
            add_log(f"Device {device_id} requested a task — queue empty.", "warn")
            return jsonify({"error": "No tasks available."}), 404
        task = task_queue.popleft()

    assigned_devices.add(device_id)
    server_state["segments_dispatched"] += 1

    add_log(
        f"Task {id_registry.readable(task['task_Id'])} → {device_id} "
        f"({len(assigned_devices)}/{CONFIG['MAX_CLIENTS']})", "success"
    )
    return jsonify(task)


# =============================================================
# 2. FILE DOWNLOADS
# =============================================================

@app.route("/download/images", methods=["GET"])
def download_images():
    filename  = request.args.get("filename", "images_000.bin")
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    return send_file(file_path, as_attachment=True) if os.path.exists(file_path) \
        else (f"File '{filename}' not found.", 404)


@app.route("/download/labels", methods=["GET"])
def download_labels():
    filename  = request.args.get("filename", "labels_000.bin")
    file_path = os.path.join(DOWNLOAD_DIR, filename)
    return send_file(file_path, as_attachment=True) if os.path.exists(file_path) \
        else (f"File '{filename}' not found.", 404)


@app.route("/download/model", methods=["GET"])
def download_model():
    filename  = request.args.get("filename", f"{CONFIG['MODEL_ID']}_model.tflite")
    file_path = os.path.join(BASE_DIR, "downloads", filename)
    return send_file(file_path, as_attachment=True) if os.path.exists(file_path) \
        else (f"Model file '{filename}' not found.", 404)


# =============================================================
# 3. CHECKPOINT UPLOAD
# =============================================================

@app.route("/api/model/upload", methods=["POST"])
def upload_checkpoint():
    task_Id       = request.form.get("task_Id", "unknown_task")
    device_id     = request.form.get("device_id", "unknown_device")
    task_json_str = request.form.get("task_json", "{}")

    if "model_file" not in request.files:
        return "No file uploaded", 400
    file = request.files["model_file"]
    if file.filename == "":
        return "No selected file", 400

    save_filename = f"{task_Id}_dev_{device_id}.ckpt"
    file.save(os.path.join(UPLOAD_DIR, save_filename))

    with open(os.path.join(UPLOAD_DIR, f"{task_Id}_metadata.json"), "w", encoding="utf-8") as f:
        f.write(task_json_str)

    add_log(f"Received checkpoint: {save_filename}", "success")

    received_ckpts = [fn for fn in os.listdir(UPLOAD_DIR) if fn.endswith(".ckpt")]
    server_state["clients_uploaded"] = len(received_ckpts)
    add_log(f"Checkpoints: {len(received_ckpts)}/{CONFIG['MAX_CLIENTS']}", "info")

    if len(received_ckpts) >= CONFIG["MAX_CLIENTS"]:
        add_log("All clients reported — triggering Federated Averaging.", "info")
        perform_federated_averaging()

        assigned_devices.clear()
        server_state["clients_uploaded"] = 0
        for fn in received_ckpts:
            os.remove(os.path.join(UPLOAD_DIR, fn))

        if not server_state["finished"]:
            add_log("Round reset. Ready for next round.", "info")

    return "Upload Successful", 200


# =============================================================
# 4. STATUS  (primary data source for the dashboard)
# =============================================================

@app.route("/api/status", methods=["GET"])
def get_status():
    # TFLite model size
    tflite_path = os.path.join(BASE_DIR, "downloads", f"{CONFIG['MODEL_ID']}_model.tflite")
    tflite_kb   = round(os.path.getsize(tflite_path) / 1024, 1) if os.path.exists(tflite_path) else 0

    with _log_lock:
        log_snapshot = list(_log)

    return jsonify({
        # Round state
        "round":               server_state["round"],
        "max_rounds":          CONFIG["MAX_ROUNDS"],
        "aggregation_done":    server_state["aggregation_done"],
        "finished":            server_state["finished"],
        "restarting":          server_state["restarting"],

        # Segment / task counters
        "segments_total":      server_state["segments_total"],
        "segments_dispatched": server_state["segments_dispatched"],
        "segments_remaining":  len(task_queue),

        # Device / upload counters
        "num_clients":         CONFIG["MAX_CLIENTS"],
        "assigned_devices":    list(assigned_devices),  # array for the roster panel
        "clients_uploaded":    server_state["clients_uploaded"],

        # Model files
        "tflite_model_size_kb": tflite_kb,
        "global_model_exists":  os.path.exists(GLOBAL_CKPT_PATH + ".index"),

        # Full config — echoed so the dashboard can populate the drawer
        "config": {
            # Identity
            "MODEL_ID":  CONFIG["MODEL_ID"],
            "DATA_ID":   CONFIG["DATA_ID"],
            # Fleet
            "MAX_CLIENTS":         CONFIG["MAX_CLIENTS"],
            "MAX_ROUNDS":          CONFIG["MAX_ROUNDS"],
            "REPETITIVE_TRAINING": CONFIG["REPETITIVE_TRAINING"],
            # Data
            "N_BINS":        CONFIG["N_BINS"],
            "ITEMS_PER_BIN": CONFIG["ITEMS_PER_BIN"],
            # Hyperparams
            "NUM_EPOCHS":  CONFIG["NUM_EPOCHS"],
            "BATCH_SIZE":  CONFIG["BATCH_SIZE"],
            "INPUT_SHAPE": CONFIG["INPUT_SHAPE"],
            "NUM_CLASSES": CONFIG["NUM_CLASSES"],
            # UI extras
            "DATASET":      CONFIG.get("DATASET", "Fashion-MNIST"),
            "ARCHITECTURE": CONFIG.get("ARCHITECTURE", "—"),
            # Dashboard-compat aliases
            "NUM_CLIENTS":       CONFIG["MAX_CLIENTS"],
            "TOTAL_IMAGES":      CONFIG["N_BINS"] * CONFIG["ITEMS_PER_BIN"],
            "IMAGES_PER_CLIENT": CONFIG["ITEMS_PER_BIN"],
            "IMG_SIZE":          CONFIG["INPUT_SHAPE"][0] if CONFIG["INPUT_SHAPE"] else 28,
        },

        # Structured log
        "log": log_snapshot,
    })


# =============================================================
# 5. CONFIG UPDATE  (hot-patch in-memory, no data rebuild)
# =============================================================

ALLOWED_HOT_PATCH = {
    "MAX_CLIENTS", "MAX_ROUNDS", "REPETITIVE_TRAINING",
    "NUM_EPOCHS", "BATCH_SIZE", "NUM_CLASSES", "INPUT_SHAPE",
    "INPUT_TENSOR_NAME", "OUTPUT_TENSOR_NAME",
    "DATASET", "ARCHITECTURE",
    # dashboard-compat aliases (mapped below)
    "NUM_CLIENTS", "IMAGES_PER_CLIENT", "TOTAL_IMAGES", "IMG_SIZE",
    "task_Id",
}

@app.route("/api/config", methods=["POST"])
def update_config():
    payload = request.get_json(force=True, silent=True) or {}
    errors  = []

    # Map dashboard field names → CONFIG keys
    aliases = {
        "NUM_CLIENTS":       "MAX_CLIENTS",
        "IMAGES_PER_CLIENT": "ITEMS_PER_BIN",
        "IMG_SIZE":          "_IMG_SIZE",   # handled specially below
    }

    for k, v in payload.items():
        mapped = aliases.get(k, k)

        if mapped == "_IMG_SIZE":
            try:
                sz = int(v)
                CONFIG["INPUT_SHAPE"] = [sz, sz]
                add_log(f"Config: INPUT_SHAPE = [{sz}, {sz}]", "info")
            except Exception:
                errors.append(f"Invalid IMG_SIZE: {v}")
            continue

        if mapped not in CONFIG and mapped not in ALLOWED_HOT_PATCH:
            errors.append(f"Unknown key: {k}")
            continue

        try:
            CONFIG[mapped] = v
            add_log(f"Config: {mapped} = {v}", "info")
        except Exception as e:
            errors.append(str(e))

    return jsonify({"config": CONFIG, "errors": errors})


# =============================================================
# 6. RESTART  (rebuilds data bins + task queue in background)
# =============================================================

@app.route("/api/restart", methods=["POST"])
def restart_server():
    if server_state["restarting"]:
        return jsonify({"error": "Already restarting."}), 409

    payload    = request.get_json(force=True, silent=True) or {}
    new_config = payload.get("config", None)

    threading.Thread(target=_do_restart, args=(new_config,), daemon=True).start()
    return jsonify({"status": "Restart initiated."})


# =============================================================
# STARTUP
# =============================================================

if __name__ == "__main__":
    bin_files = create_android_ready_bins(
        n_bins=CONFIG["N_BINS"],
        output_dir=DOWNLOAD_DIR,
        items_per_bin=CONFIG["ITEMS_PER_BIN"],
    )
    build_task_queue(bin_files)

    print("===================================================")
    print(" FRACTAL FEDERATED LEARNING SERVER")
    print(f" Dashboard  →  http://127.0.0.1:5001/")
    print(f" MODEL_ID:                   {CONFIG['MODEL_ID']}")
    print(f" DATA_ID:                    {CONFIG['DATA_ID']}")
    print(f" MAX CLIENTS PER ROUND:      {CONFIG['MAX_CLIENTS']}")
    print(f" MAX ROUNDS:                 {CONFIG['MAX_ROUNDS']}")
    print(f" REPETITIVE TRAINING:        {CONFIG['REPETITIVE_TRAINING']}")
    print(f" NUM EPOCHS:                 {CONFIG['NUM_EPOCHS']}")
    print(f" BATCH SIZE:                 {CONFIG['BATCH_SIZE']}")
    print(f" ITEMS PER BIN:              {CONFIG['ITEMS_PER_BIN']}")
    print(f" TASKS IN QUEUE:             {len(task_queue)}")
    print("===================================================")

    app.run(host="0.0.0.0", port=5001, debug=True)