from flask import Flask, jsonify, send_file, request, render_template, redirect, url_for, session, send_from_directory
import os
from dotenv import load_dotenv

load_dotenv()

import json

import threading
import tensorflow as tf
import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
to_categorical = keras.utils.to_categorical
from typing import Optional # <-- Added for Python 3.8 Compatibility

import firebase_admin
from firebase_admin import credentials, firestore as fb_firestore
from google.cloud.firestore_v1 import Increment
from google.cloud.firestore_v1.base_query import FieldFilter # <-- THE WARNING FIX

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SECRETS_DIR  = os.path.join(PROJECT_ROOT, "secrets")
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")

_SERVICE_ACCOUNT_PATH = os.getenv(
    "FIREBASE_SERVICE_ACCOUNT_PATH",
    os.path.join(SECRETS_DIR, "firebase", "legacy-service-account.json")
)
_fb_app = firebase_admin.initialize_app(credentials.Certificate(_SERVICE_ACCOUNT_PATH))
_db     = fb_firestore.client()


app = Flask(__name__)
app.secret_key  = os.getenv('SECRET_KEY', 'fractal_system_secure_key_2026_dev')
ADMIN_PASSWORD  = os.getenv('ADMIN_PASSWORD', 'orchestrate')


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response

def login_required(f):
    def wrapper(*args, **kwargs):
        if not session.get('logged_in'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    wrapper.__name__ = f.__name__
    return wrapper

CONFIG = {
    "MODEL_ID":  "0009",
    "DATA_ID":   "2009",
    "MAX_CLIENTS":          3,
    "MAX_ROUNDS":           5,
    "REPETITIVE_TRAINING":  True,

    "CHECKPOINT_REWARD_RATE": 15.0,
    "N_BINS":        10,
    "ITEMS_PER_BIN": 6000,

    "NUM_EPOCHS":  2,
    "BATCH_SIZE":  100,
    "INPUT_SHAPE": [28, 28],
    "NUM_CLASSES": 10,
    "INPUT_TENSOR_NAME":  {"x": "FloatBuffer"},
    "OUTPUT_TENSOR_NAME": {"loss": "FloatBuffer", "output": "FloatBuffer"},

    "DATASET":      "Fashion-MNIST",
    "ARCHITECTURE": "MobileNet",
}

BASE_DIR         = os.path.dirname(os.path.abspath(__file__))
DOWNLOAD_DIR     = os.path.join(DATA_DIR, "android_training_bins")
UPLOAD_DIR       = os.path.join(DATA_DIR, "legacy_uploads")
MODEL_DIR        = os.path.join(DATA_DIR, "legacy_downloads")
GLOBAL_CKPT_PATH = os.path.join(DATA_DIR, "legacy_global_model", "global.ckpt")

for path in [DOWNLOAD_DIR, UPLOAD_DIR, MODEL_DIR, os.path.dirname(GLOBAL_CKPT_PATH)]:
    os.makedirs(path, exist_ok=True)

_stale = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".ckpt")]
for _f in _stale:
    os.remove(os.path.join(UPLOAD_DIR, _f))
if _stale:
    print(f"[!] Purged {len(_stale)} stale checkpoint(s) from uploads/ on startup.")

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        if request.form.get('password') == ADMIN_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        return render_template('login.html', error="Invalid Authentication Key")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ─────────────────────────────────────────────────────────────────────────────
class IDRegistry:
    def __init__(self):
        self._lock             = threading.Lock()
        self._segment_counters = {}
        self._task_counter     = 0

    def reset(self):
        with self._lock:
            self._segment_counters.clear()
            self._task_counter = 0

    def generateDataSegmentId(self, data_id: str) -> tuple:
        with self._lock:
            seq = self._segment_counters.get(data_id, 0) + 1
            self._segment_counters[data_id] = seq
            seg_seq = f"{seq:07d}"
            return f"{data_id}{seg_seq}", seg_seq

    def generateTaskId(self, model_id: str, data_id: str, segment_sequence: str) -> tuple:
        with self._lock:
            self._task_counter += 1
            task_seq = f"{self._task_counter:07d}"
            return f"{model_id}{data_id}{segment_sequence}{task_seq}", task_seq

    def readable(self, task_Id: str) -> str:
        return f"{task_Id[0:4]}_{task_Id[4:8]}_{task_Id[8:15]}_{task_Id[15:22]}"

id_registry = IDRegistry()

task_queue           = deque()
task_queue_lock      = threading.Lock()
_bin_segment_cache   = []
assigned_devices     = set()

# ── task_Id → (device_id, reward_mbs) ─────────────────────────────────────
# Written at dispatch, read + popped at upload.
_task_device_map      = {}
_task_device_map_lock = threading.Lock()
# ─────────────────────────────────────────────────────────────────────────────

server_state = {
    "round":               1,
    "aggregation_done":    False,
    "finished":            False,
    "restarting":          False,
    "paused":              True,
    "segments_total":      0,
    "segments_dispatched": 0,
    "clients_uploaded":    0,
}

_log      = []
_log_lock = threading.Lock()

def add_log(message: str, level: str = "info"):
    prefix = {"info": "[*]", "warn": "[!]", "error": "[X]", "success": "[+]"}.get(level, "[*]")
    print(f"{prefix} {message}")
    with _log_lock:
        _log.append({"time": datetime.now().strftime("%H:%M:%S"), "level": level, "msg": message})
        if len(_log) > 300:
            _log.pop(0)

# ── Firebase reward helper ────────────────────────────────────────────────────

def _get_email_by_device_id(device_id: str) -> Optional[str]:
    try:
        docs = (
            _db.collection("registered_devices")
               .where(filter=FieldFilter("hardwareId", "==", device_id))
               .limit(1)
               .stream()
        )
        for doc in docs:
            return doc.to_dict().get("email")
    except Exception as e:
        add_log(f"Firestore Device lookup failed: {e}", "error")
    return None

def credit_mbs_for_device(device_id: str, mbs_to_add: float) -> bool:
    """Increment liquid_mbs but strictly CAP at 2048MB (2GB)."""
    if not device_id or device_id.lower() in ("unknown", ""):
        return False

    email = _get_email_by_device_id(device_id)
    if not email:
        add_log(f"credit_mbs: no user for {device_id}", "warn")
        return False

    MAX_CAP = 2048.0 # 2GB Limit

    try:
        user_ref = _db.collection("users").document(email)
        
        # 1. Transactional Get to check current balance
        doc = user_ref.get()
        current_balance = doc.to_dict().get("liquid_mbs", 0.0) if doc.exists else 0.0

        if current_balance >= MAX_CAP:
            add_log(f"Reward Skipped: {email} is already at MAX capacity (2GB).", "info")
            return True

        # 2. Calculate capped increment
        # If current(2040) + reward(15) > 2048 -> only add 8
        final_increment = min(mbs_to_add, MAX_CAP - current_balance)

        user_ref.set(
            {"liquid_mbs": Increment(final_increment)},
            merge=True
        )
        add_log(f"Credited +{final_increment} MB -> {email}. (Total: ~{current_balance + final_increment}MB)", "success")
        return True
    except Exception as e:
        add_log(f"Firestore MB credit failed: {e}", "error")
        return False
    

def perform_federated_averaging():
    add_log("Federated Aggregation: Starting...", "info")
    server_state["aggregation_done"] = False

    client_files = [os.path.join(UPLOAD_DIR, f)
                    for f in os.listdir(UPLOAD_DIR) if f.endswith(".ckpt")]
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

def init_bin_segment_map(bin_files: list):
    global _bin_segment_cache
    _bin_segment_cache = []
    data_id = CONFIG["DATA_ID"]
    for img_name, lbl_name in bin_files:
        data_segment_id, seg_seq = id_registry.generateDataSegmentId(data_id)
        _bin_segment_cache.append((img_name, lbl_name, data_segment_id, seg_seq))
    add_log(f"Segment map initialised: {len(_bin_segment_cache)} bin(s).", "info")

def build_task_queue():
    task_queue.clear()
    model_id = CONFIG["MODEL_ID"]
    data_id  = CONFIG["DATA_ID"]
    add_log(f"Building task queue — MODEL_ID={model_id}  DATA_ID={data_id}", "info")

    for img_name, lbl_name, data_segment_id, seg_seq in _bin_segment_cache:
        task_Id, task_seq = id_registry.generateTaskId(model_id, data_id, seg_seq)
        task_queue.append({
            "task_Id":          task_Id,
            "task_Id_readable": id_registry.readable(task_Id),
            "model_id":         model_id,
            "data_id":          data_id,
            "data_segment_id":  data_segment_id,
            "segment_sequence": seg_seq,
            "task_sequence":    task_seq,

            "architecture":     CONFIG["ARCHITECTURE"],
            "reward_rate":      CONFIG["CHECKPOINT_REWARD_RATE"],

            "taskType":               "ActiveTask",
            "task_expire_date":       (datetime.now() + timedelta(days=7)).strftime("%Y-%m-%d"),
            "task_completion_status": False,
            "training_type": [
                "Image_Task", "Image_DataInitializer",
                "Image_Trainer", "Image_InferenceValidator",
            ],

            "CKPT_FILENAME":       "checkpoint.ckpt",
            "NUM_EPOCHS":          CONFIG["NUM_EPOCHS"],
            "BATCH_SIZE":          CONFIG["BATCH_SIZE"],
            "NUM_TRAININGS":       CONFIG["ITEMS_PER_BIN"],
            "INPUT_SHAPE":         CONFIG["INPUT_SHAPE"],
            "NUM_CLASSES":         CONFIG["NUM_CLASSES"],
            "input_tensor_name":   CONFIG["INPUT_TENSOR_NAME"],
            "output_tensor_name":  CONFIG["OUTPUT_TENSOR_NAME"],

            "MODEL_FILENAME":        f"{model_id}_model.tflite",
            "TRAIN_IMAGES_FILENAME": img_name,
            "TRAIN_LABELS_FILENAME": lbl_name,
        })

    server_state["segments_total"]      = len(task_queue)
    server_state["segments_dispatched"] = 0
    server_state["clients_uploaded"]    = 0
    add_log(f"{len(task_queue)} task(s) queued and ready to dispatch.", "success")

def _do_restart(new_config=None):
    global assigned_devices
    server_state["restarting"] = True
    add_log("Server restart initiated.", "warn")

    if new_config:
        for k, v in new_config.items():
            if k in CONFIG:
                CONFIG[k] = v
                add_log(f"Config updated: {k} = {v}", "info")

    server_state.update({
        "round": 1, "aggregation_done": False,
        "finished": False, "segments_dispatched": 0,
        "clients_uploaded": 0,
    })
    assigned_devices = set()
    id_registry.reset()

    with _task_device_map_lock:
        _task_device_map.clear()

    for fn in os.listdir(UPLOAD_DIR):
        if fn.endswith(".ckpt") or fn.endswith(".json"):
            os.remove(os.path.join(UPLOAD_DIR, fn))

    try:
        bin_files = create_android_ready_bins(
            n_bins=CONFIG["N_BINS"],
            output_dir=DOWNLOAD_DIR,
            items_per_bin=CONFIG["ITEMS_PER_BIN"],
        )
        init_bin_segment_map(bin_files)
        build_task_queue()
    except Exception as e:
        add_log(f"Restart failed during data build: {e}", "error")

    server_state["restarting"] = False
    add_log("Server restart complete.", "success")


@app.route("/api/task/current", methods=["GET"])
def get_current_task():
    # device_id IS the hardwareId sent by the Android client
    device_id = request.args.get("device_id", "unknown")

    if server_state["finished"]:
        return jsonify({"error": "All training rounds complete. Server is done."}), 410
    if server_state["restarting"]:
        return jsonify({"error": "Server is restarting. Try again shortly."}), 503
    if server_state["paused"]:
        return jsonify({"error": "Server is paused."}), 503
    if len(assigned_devices) >= CONFIG["MAX_CLIENTS"]:
        add_log(f"Device {device_id} rejected — cap of {CONFIG['MAX_CLIENTS']} reached.", "warn")
        return jsonify({"error": f"Max client limit ({CONFIG['MAX_CLIENTS']}) reached."}), 403
    if not CONFIG["REPETITIVE_TRAINING"] and device_id in assigned_devices:
        add_log(f"Device {device_id} already assigned. Rejecting.", "warn")
        return jsonify({"error": "Task already completed by this device."}), 403

    with task_queue_lock:
        if not task_queue:
            if _bin_segment_cache:
                add_log("All bins dispatched — refilling queue circularly.", "info")
                build_task_queue()
            else:
                add_log(f"Device {device_id} requested a task — queue empty, no bin cache.", "warn")
                return jsonify({"error": "No tasks available."}), 404
        task = task_queue.popleft()

    assigned_devices.add(device_id)
    server_state["segments_dispatched"] += 1

    # ── Remember which Device ID owns this task ─────────────────────────────
    with _task_device_map_lock:
        _task_device_map[task["task_Id"]] = {
            "device_id": device_id,
            "reward_mbs":  float(task["reward_rate"]),  # snapshot at dispatch time
        }
    # ─────────────────────────────────────────────────────────────────────────

    add_log(
        f"Task {id_registry.readable(task['task_Id'])} -> {device_id} "
        f"({len(assigned_devices)}/{CONFIG['MAX_CLIENTS']})", "success"
    )
    return jsonify(task)


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


@app.route("/api/model/upload", methods=["POST"])
def upload_checkpoint():
    task_Id   = (request.form.get("task_Id")
                 or request.form.get("taskId")
                 or "unknown_task")
    device_id = request.form.get("device_id", "unknown_device")

    if "model_file" not in request.files:
        return "No file uploaded", 400
    file = request.files["model_file"]
    if file.filename == "":
        return "No selected file", 400

    ts = int(datetime.now().timestamp() * 1000)
    save_filename = f"{task_Id}_dev_{device_id}_{ts}.ckpt"
    file.save(os.path.join(UPLOAD_DIR, save_filename))
    add_log(f"Received checkpoint: {save_filename}", "success")

    # ── Look up who owned this task and credit their MBs ─────────────────────
    with _task_device_map_lock:
        dispatch_info = _task_device_map.pop(task_Id, None)  # pop = one-time credit

    if dispatch_info:
        # Fallback to mac_address if an old task from before the update is still running
        target_device = dispatch_info.get("device_id", dispatch_info.get("mac_address")) 
        reward_mbs    = dispatch_info["reward_mbs"]
        
        add_log(
            f"Rewarding Device {target_device} with +{reward_mbs} MB(s) "
            f"for task {id_registry.readable(task_Id) if len(task_Id) == 22 else task_Id}",
            "info"
        )
        threading.Thread(
            target=credit_mbs_for_device,
            args=(target_device, reward_mbs),
            daemon=True
        ).start()
    else:
        add_log(f"upload_checkpoint: no dispatch record for task_Id {task_Id} — reward skipped.", "warn")
    # ─────────────────────────────────────────────────────────────────────────

    received_ckpts = [fn for fn in os.listdir(UPLOAD_DIR) if fn.endswith(".ckpt")]
    server_state["clients_uploaded"] = len(received_ckpts)
    add_log(f"Checkpoints: {len(received_ckpts)}/{CONFIG['MAX_CLIENTS']}", "info")

    if len(received_ckpts) >= CONFIG["MAX_CLIENTS"]:
        add_log("All clients reported — triggering Federated Averaging.", "info")
        perform_federated_averaging()

        assigned_devices.clear()
        server_state["clients_uploaded"]    = 0
        server_state["segments_dispatched"] = 0
        server_state["aggregation_done"]    = False

        for fn in received_ckpts:
            os.remove(os.path.join(UPLOAD_DIR, fn))

        if not server_state["finished"]:
            add_log("Round reset. Ready for next round.", "info")

    return "Upload Successful", 200


@app.route("/api/status", methods=["GET"])
def get_status():
    tflite_path = os.path.join(BASE_DIR, "downloads", f"{CONFIG['MODEL_ID']}_model.tflite")
    tflite_kb   = round(os.path.getsize(tflite_path) / 1024, 1) if os.path.exists(tflite_path) else 0

    with _log_lock:
        log_snapshot = list(_log)

    return jsonify({
        "round":               server_state["round"],
        "max_rounds":          CONFIG["MAX_ROUNDS"],
        "aggregation_done":    server_state["aggregation_done"],
        "paused":              server_state["paused"],
        "finished":            server_state["finished"],
        "restarting":          server_state["restarting"],

        "segments_total":      server_state["segments_total"],
        "segments_dispatched": server_state["segments_dispatched"],
        "segments_remaining":  len(task_queue),

        "num_clients":         CONFIG["MAX_CLIENTS"],
        "assigned_devices":    list(assigned_devices),
        "clients_uploaded":    server_state["clients_uploaded"],

        "tflite_model_size_kb": tflite_kb,
        "global_model_exists":  os.path.exists(GLOBAL_CKPT_PATH + ".index"),

        "config": {
            "MODEL_ID":  CONFIG["MODEL_ID"],
            "DATA_ID":   CONFIG["DATA_ID"],
            "MAX_CLIENTS":            CONFIG["MAX_CLIENTS"],
            "MAX_ROUNDS":             CONFIG["MAX_ROUNDS"],
            "REPETITIVE_TRAINING":    CONFIG["REPETITIVE_TRAINING"],
            "CHECKPOINT_REWARD_RATE": CONFIG.get("CHECKPOINT_REWARD_RATE", 15.0),
            "N_BINS":        CONFIG["N_BINS"],
            "ITEMS_PER_BIN": CONFIG["ITEMS_PER_BIN"],
            "NUM_EPOCHS":    CONFIG["NUM_EPOCHS"],
            "BATCH_SIZE":    CONFIG["BATCH_SIZE"],
            "INPUT_SHAPE":   CONFIG["INPUT_SHAPE"],
            "NUM_CLASSES":   CONFIG["NUM_CLASSES"],
            "DATASET":       CONFIG.get("DATASET", "Fashion-MNIST"),
            "ARCHITECTURE":  CONFIG.get("ARCHITECTURE", "—"),
            "NUM_CLIENTS":       CONFIG["MAX_CLIENTS"],
            "TOTAL_IMAGES":      CONFIG["N_BINS"] * CONFIG["ITEMS_PER_BIN"],
            "IMAGES_PER_CLIENT": CONFIG["ITEMS_PER_BIN"],
            "IMG_SIZE":          CONFIG["INPUT_SHAPE"][0] if CONFIG["INPUT_SHAPE"] else 28,
        },
        "log": log_snapshot,
    })

ALLOWED_HOT_PATCH = {
    "MAX_CLIENTS", "MAX_ROUNDS", "REPETITIVE_TRAINING",
    "NUM_EPOCHS", "BATCH_SIZE", "NUM_CLASSES", "INPUT_SHAPE",
    "INPUT_TENSOR_NAME", "OUTPUT_TENSOR_NAME",
    "DATASET", "ARCHITECTURE",
    "NUM_CLIENTS", "IMAGES_PER_CLIENT", "TOTAL_IMAGES", "IMG_SIZE",
    "TASK_ID", "CHECKPOINT_REWARD_RATE",
}

@app.route("/")
@login_required
def dashboard():
    return render_template("dashboard.html")

@app.route("/api/config", methods=["POST"])
@login_required
def update_config():
    payload = request.get_json(force=True, silent=True) or {}
    errors  = []
    aliases = {
        "NUM_CLIENTS":       "MAX_CLIENTS",
        "IMAGES_PER_CLIENT": "ITEMS_PER_BIN",
        "IMG_SIZE":          "_IMG_SIZE",
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

@app.route("/api/restart", methods=["POST"])
@login_required
def restart_server():
    if server_state["restarting"]:
        return jsonify({"error": "Already restarting."}), 409
    payload    = request.get_json(force=True, silent=True) or {}
    new_config = payload.get("config", None)
    threading.Thread(target=_do_restart, args=(new_config,), daemon=True).start()
    return jsonify({"status": "Restart initiated."})

@app.route("/api/pause", methods=["POST"])
@login_required
def toggle_pause():
    server_state["paused"] = not server_state["paused"]
    state = "paused" if server_state["paused"] else "resumed"
    add_log(f"Server {state} by admin.", "warn")
    return jsonify({"paused": server_state["paused"]})


if __name__ == "__main__":
    bin_files = create_android_ready_bins(
        n_bins=CONFIG["N_BINS"],
        output_dir=DOWNLOAD_DIR,
        items_per_bin=CONFIG["ITEMS_PER_BIN"],
    )
    init_bin_segment_map(bin_files)
    build_task_queue()

    print("===================================================")
    print(" FRACTAL FEDERATED LEARNING SERVER")
    print(f" Dashboard  ->  http://127.0.0.1:5000/")
    print(f" MODEL_ID:                   {CONFIG['MODEL_ID']}")
    print(f" DATA_ID:                    {CONFIG['DATA_ID']}")
    print(f" MAX CLIENTS PER ROUND:      {CONFIG['MAX_CLIENTS']}")
    print(f" MAX ROUNDS:                 {CONFIG['MAX_ROUNDS']}")
    print(f" REPETITIVE TRAINING:        {CONFIG['REPETITIVE_TRAINING']}")
    print(f" NUM EPOCHS:                 {CONFIG['NUM_EPOCHS']}")
    print(f" BATCH SIZE:                 {CONFIG['BATCH_SIZE']}")
    print(f" ITEMS PER BIN:              {CONFIG['ITEMS_PER_BIN']}")
    print(f" CHECKPOINT REWARD RATE:     {CONFIG['CHECKPOINT_REWARD_RATE']} MB(s)")
    print(f" TASKS IN QUEUE:             {len(task_queue)}")
    print("===================================================")

    app.run(host="0.0.0.0", port=5000, debug=True)