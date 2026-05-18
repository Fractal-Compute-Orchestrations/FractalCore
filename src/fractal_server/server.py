from flask import (Flask, jsonify, send_file, request,
                   render_template, redirect, url_for,
                   session as flask_session)
import os, json, secrets, threading, shutil
from dotenv import load_dotenv

load_dotenv()

import tensorflow as tf

import numpy as np
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta
from tensorflow import keras
fashion_mnist = keras.datasets.fashion_mnist
to_categorical = keras.utils.to_categorical
from typing import Optional
from firebase_reward import credit_mbs_for_device


import firebase_admin
from firebase_admin import credentials, firestore as fb_firestore
from google.cloud.firestore_v1 import Increment
from google.cloud.firestore_v1.base_query import FieldFilter

BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT          = os.path.dirname(os.path.dirname(BASE_DIR))
DATA_DIR              = os.path.join(PROJECT_ROOT, "data")
SECRETS_DIR           = os.path.join(PROJECT_ROOT, "secrets")
TENANTS_JSON          = os.path.join(SECRETS_DIR, "tenants.json")
USE_FIRESTORE         = True

# ── Firebase ──────────────────────────────────────────────────────────────────
from firebase_reward import _db

# ── Flask ─────────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "fractal_system_secure_key_2026_dev_only")

ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "orchestrate")


SHARED_MODEL_DIR      = os.path.join(DATA_DIR, "downloads")
SHARED_MODEL_FILENAME = "0009_model.tflite"
os.makedirs(SHARED_MODEL_DIR, exist_ok=True)

# ── TFLOPs ────────────────────────────────────────────────────────────────────
FLOPS_PER_IMAGE: float = 12.6e6   # MobileNet 28×28, fwd+bwd per image per epoch

def _tflops_per_task(cfg: dict) -> float:
    return cfg["ITEMS_PER_BIN"] * cfg["NUM_EPOCHS"] * FLOPS_PER_IMAGE / 1e12

def _tflops_full_session(cfg: dict) -> float:
    return (cfg["N_BINS"] * cfg["ITEMS_PER_BIN"]
            * cfg["NUM_EPOCHS"] * cfg["MAX_ROUNDS"]
            * FLOPS_PER_IMAGE / 1e12)

def _calc_reward(total_device_mbs: float, max_clients: int) -> float:
    """Reward per task = total device MB budget ÷ number of client slots."""
    return round(total_device_mbs / max_clients, 4) if max_clients > 0 else 0.0

# ── Fashion-MNIST shared cache ────────────────────────────────────────────────
_fashion_mnist_lock  = threading.Lock()
_fashion_mnist_cache: Optional[tuple] = None

def _load_fashion_mnist():
    global _fashion_mnist_cache
    with _fashion_mnist_lock:
        if _fashion_mnist_cache is None:
            (x_tr, y_tr), (x_te, y_te) = fashion_mnist.load_data()
            _fashion_mnist_cache = (
                np.concatenate((x_tr, x_te)),
                np.concatenate((y_tr, y_te)),
            )
        return _fashion_mnist_cache


@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"]  = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Auth-Token"
    return response


# ─────────────────────────────────────────────────────────────────────────────
#  Token store  — per-login token, sent as X-Auth-Token header from the client.
#  sessionStorage in the browser ensures each tab has its own token.
# ─────────────────────────────────────────────────────────────────────────────
_token_store: dict = {}
_token_lock = threading.Lock()
tokens_json = os.path.join(SECRETS_DIR, "tokens.json")

def _load_tokens():
    global _token_store
    if os.path.exists(tokens_json):
        try:
            with open(tokens_json) as f:
                _token_store = json.load(f)
        except Exception:
            _token_store = {}
    else:
        _token_store = {}

def _save_tokens():
    try:
        with open(tokens_json, "w") as f:
            json.dump(_token_store, f, indent=2)
    except Exception:
        pass

def _issue_token(user: str, role: str) -> str:
    token = secrets.token_hex(32)
    with _token_lock:
        _load_tokens()
        stale = [k for k, v in _token_store.items() if v["user"] == user]
        for k in stale:
            del _token_store[k]
        _token_store[token] = {"user": user, "role": role}
        _save_tokens()
    return token

def _revoke_token(token: str):
    with _token_lock:
        _load_tokens()
        _token_store.pop(token, None)
        _save_tokens()

def _get_auth() -> dict:
    """Authenticate ONLY from X-Auth-Token header. No cookie fallback.
    This is intentional: the cookie fallback caused all tenants to share
    the same flask session and always resolve to the first logged-in user."""
    token = request.headers.get("X-Auth-Token", "").strip()
    if not token:
        return {}
    with _token_lock:
        _load_tokens()
        return dict(_token_store.get(token, {}))


# ─────────────────────────────────────────────────────────────────────────────
#  IDRegistry
# ─────────────────────────────────────────────────────────────────────────────
class IDRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._seg_counters: dict = {}
        self._task_counter: int  = 0

    def reset(self):
        with self._lock:
            self._seg_counters.clear()
            self._task_counter = 0

    def generateDataSegmentId(self, data_id: str):
        with self._lock:
            seq = self._seg_counters.get(data_id, 0) + 1
            self._seg_counters[data_id] = seq
            ss = f"{seq:07d}"
            return f"{data_id}{ss}", ss

    def generateTaskId(self, model_id: str, data_id: str, seg_seq: str):
        with self._lock:
            self._task_counter += 1
            ts = f"{self._task_counter:07d}"
            return f"{model_id}{data_id}{seg_seq}{ts}", ts

    def readable(self, tid: str) -> str:
        return f"{tid[0:4]}_{tid[4:8]}_{tid[8:15]}_{tid[15:22]}" if len(tid) >= 22 else tid


# ─────────────────────────────────────────────────────────────────────────────
#  TenantSession
# ─────────────────────────────────────────────────────────────────────────────
class TenantSession:
    def __init__(self, username: str, n: int):
        self.username = username
        self.n        = n
        self.model_id = f"{n:04d}"
        self.data_id  = f"{n + 1000:04d}"

        self.config: dict = {
            "MODEL_ID":   self.model_id,
            "DATA_ID":    self.data_id,
            "MAX_CLIENTS":          3,
            "MAX_ROUNDS":           5,
            "REPETITIVE_TRAINING":  True,
            "CHECKPOINT_REWARD_RATE": 0.0,
            "N_BINS":        10,
            "ITEMS_PER_BIN": 6000,
            "NUM_EPOCHS":    2,
            "BATCH_SIZE":    100,
            "INPUT_SHAPE":   [28, 28],
            "NUM_CLASSES":   10,
            "INPUT_TENSOR_NAME":  {"x": "FloatBuffer"},
            "OUTPUT_TENSOR_NAME": {"loss": "FloatBuffer", "output": "FloatBuffer"},
            "DATASET":      "Fashion-MNIST",
            "ARCHITECTURE": "MobileNet",
            "AUTO_DELETE_CHECKPOINTS": False,
        }

        self.state: dict = {
            "round": 1, "aggregation_done": False, "finished": False,
            "restarting": False, "paused": True, "running": False,
            "segments_total": 0, "segments_dispatched": 0, "clients_uploaded": 0,
        }

        self.max_tflops:       float = 0.0
        self.remaining_tflops: float = 0.0
        self._tflops_lock = threading.Lock()

        self.task_queue       = deque()
        self.task_queue_lock  = threading.Lock()
        self.assigned_devices: set  = set()
        self._task_device_map: dict = {}
        self._task_device_map_lock  = threading.Lock()
        self._bin_segment_cache: list = []
        self.id_registry = IDRegistry()
        self._log:     list = []
        self._log_lock = threading.Lock()

        self.download_dir     = os.path.join(DATA_DIR, "tenants", username, "bins")
        self.upload_dir       = os.path.join(DATA_DIR, "tenants", username, "uploads")
        self.global_ckpt_path = os.path.join(DATA_DIR, "tenants", username,
                                              "global_model", "global.ckpt")
        for p in [self.download_dir, self.upload_dir,
                  os.path.dirname(self.global_ckpt_path)]:
            os.makedirs(p, exist_ok=True)

    def add_log(self, msg: str, level: str = "info"):
        pfx = {"info":"[*]","warn":"[!]","error":"[X]","success":"[+]"}.get(level,"[*]")
        print(f"[{self.username}] {pfx} {msg}")
        with self._log_lock:
            self._log.append({"time": datetime.now().strftime("%H:%M:%S"),
                               "level": level, "msg": msg})
            if len(self._log) > 300:
                self._log.pop(0)

    def get_status(self) -> dict:
        cfg = self.config
        tflite_path = os.path.join(SHARED_MODEL_DIR, SHARED_MODEL_FILENAME)
        tflite_kb   = round(os.path.getsize(tflite_path) / 1024, 1) \
                      if os.path.exists(tflite_path) else 0
        with self._log_lock:
            log_snap = list(self._log)
        return {
            "round": self.state["round"], "max_rounds": cfg["MAX_ROUNDS"],
            "aggregation_done": self.state["aggregation_done"],
            "paused": self.state["paused"], "finished": self.state["finished"],
            "restarting": self.state["restarting"], "running": self.state["running"],
            "segments_total": self.state["segments_total"],
            "segments_dispatched": self.state["segments_dispatched"],
            "segments_remaining": len(self.task_queue),
            "num_clients": cfg["MAX_CLIENTS"],
            "assigned_devices": list(self.assigned_devices),
            "clients_uploaded": self.state["clients_uploaded"],
            "tflite_model_size_kb": tflite_kb,
            "global_model_exists":  os.path.exists(self.global_ckpt_path),
            "remaining_tflops":    round(self.remaining_tflops,     6),
            "max_tflops":          round(self.max_tflops,           6),
            "tflops_per_task":     round(_tflops_per_task(cfg),     6),
            "tflops_full_session": round(_tflops_full_session(cfg), 6),
            "flops_per_image":     FLOPS_PER_IMAGE,
            "config": {
                "MODEL_ID": cfg["MODEL_ID"], "DATA_ID": cfg["DATA_ID"],
                "MAX_CLIENTS": cfg["MAX_CLIENTS"], "MAX_ROUNDS": cfg["MAX_ROUNDS"],
                "REPETITIVE_TRAINING": cfg["REPETITIVE_TRAINING"],
                "AUTO_DELETE_CHECKPOINTS": cfg.get("AUTO_DELETE_CHECKPOINTS", False),
                "CHECKPOINT_REWARD_RATE": cfg.get("CHECKPOINT_REWARD_RATE", 0.0),
                "N_BINS": cfg["N_BINS"], "ITEMS_PER_BIN": cfg["ITEMS_PER_BIN"],
                "NUM_EPOCHS": cfg["NUM_EPOCHS"], "BATCH_SIZE": cfg["BATCH_SIZE"],
                "INPUT_SHAPE": cfg["INPUT_SHAPE"], "NUM_CLASSES": cfg["NUM_CLASSES"],
                "DATASET": cfg.get("DATASET", "Fashion-MNIST"),
                "ARCHITECTURE": cfg.get("ARCHITECTURE", "—"),
                "IMG_SIZE": cfg["INPUT_SHAPE"][0] if cfg["INPUT_SHAPE"] else 28,
            },
            "log": log_snap,
        }

    def _init_bins(self):
        self.add_log("Loading Fashion-MNIST dataset...", "info")
        images_raw, labels_raw = _load_fashion_mnist()
        n_bins = self.config["N_BINS"]
        items  = self.config["ITEMS_PER_BIN"]
        if n_bins * items > len(images_raw):
            raise ValueError(
                f"Not enough data — need {n_bins * items}, have {len(images_raw)}.")
        idx    = np.random.permutation(len(images_raw))
        images = (images_raw[idx] / 255.0).astype(np.float32)
        labels = to_categorical(labels_raw[idx],
                                self.config["NUM_CLASSES"]).astype(np.float32)
        os.makedirs(self.download_dir, exist_ok=True)
        bin_files = []
        for i in range(n_bins):
            s, e = i * items, (i + 1) * items
            # Prefix with username → globally unique filenames across all tenants.
            # The task JSON carries these exact names so the Android client requests
            # them as-is; the download handler then resolves tenant from the prefix.
            img_n = f"{self.username}_images_{i:03d}.bin"
            lbl_n = f"{self.username}_labels_{i:03d}.bin"
            img_path = os.path.join(self.download_dir, img_n)
            lbl_path = os.path.join(self.download_dir, lbl_n)
            Path(img_path).write_bytes(images[s:e].tobytes())
            Path(lbl_path).write_bytes(labels[s:e].tobytes())
            if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                raise RuntimeError(f"Bin {i} failed to write to disk.")
            bin_files.append((img_n, lbl_n))
        self.add_log(f"{n_bins} bin pair(s) written to '{self.download_dir}'.", "success")
        self._bin_segment_cache = []
        data_id = self.config["DATA_ID"]
        for img_n, lbl_n in bin_files:
            dsid, ss = self.id_registry.generateDataSegmentId(data_id)
            self._bin_segment_cache.append((img_n, lbl_n, dsid, ss))
        self.add_log(f"Segment map: {len(self._bin_segment_cache)} bin(s).", "info")

    def _build_task_queue(self):
        self.task_queue.clear()
        mid, did = self.config["MODEL_ID"], self.config["DATA_ID"]
        self.add_log(f"Building queue MODEL={mid} DATA={did}", "info")
        for img_n, lbl_n, dsid, ss in self._bin_segment_cache:
            tid, tseq = self.id_registry.generateTaskId(mid, did, ss)
            self.task_queue.append({
                "task_Id": tid, "task_Id_readable": self.id_registry.readable(tid),
                "tenant_id": self.username,
                "model_id": mid, "data_id": did,
                "data_segment_id": dsid, "segment_sequence": ss, "task_sequence": tseq,
                "architecture": self.config["ARCHITECTURE"],
                "reward_rate":  self.config["CHECKPOINT_REWARD_RATE"],
                "taskType": "ActiveTask",
                "task_expire_date": (datetime.now()+timedelta(days=7)).strftime("%Y-%m-%d"),
                "task_completion_status": False,
                "training_type": ["Image_Task","Image_DataInitializer",
                                  "Image_Trainer","Image_InferenceValidator"],
                "CKPT_FILENAME":         "checkpoint.ckpt",
                "NUM_EPOCHS":            self.config["NUM_EPOCHS"],
                "BATCH_SIZE":            self.config["BATCH_SIZE"],
                "NUM_TRAININGS":         self.config["ITEMS_PER_BIN"],
                "INPUT_SHAPE":           self.config["INPUT_SHAPE"],
                "NUM_CLASSES":           self.config["NUM_CLASSES"],
                "input_tensor_name":     self.config["INPUT_TENSOR_NAME"],
                "output_tensor_name":    self.config["OUTPUT_TENSOR_NAME"],
                "MODEL_FILENAME":        SHARED_MODEL_FILENAME,
                "TRAIN_IMAGES_FILENAME": img_n,
                "TRAIN_LABELS_FILENAME": lbl_n,
            })
        self.state["segments_total"]      = len(self.task_queue)
        self.state["segments_dispatched"] = 0
        self.state["clients_uploaded"]    = 0
        self.add_log(f"{len(self.task_queue)} task(s) queued.", "success")

    def start(self, max_tflops: float):
        def _run():
            self.state["restarting"] = True
            self.state["running"]    = True
            try:
                for fn in os.listdir(self.upload_dir):
                    if fn.endswith((".ckpt", ".json")):
                        os.remove(os.path.join(self.upload_dir, fn))
                self.assigned_devices.clear()
                with self._task_device_map_lock:
                    self._task_device_map.clear()
                self.state.update({
                    "round": 1, "aggregation_done": False, "finished": False,
                    "segments_dispatched": 0, "clients_uploaded": 0,
                })
                with self._tflops_lock:
                    self.max_tflops       = max_tflops
                    self.remaining_tflops = max_tflops
                self.id_registry.reset()
                self._init_bins()
                self._build_task_queue()
                self.state["paused"] = False
                self.add_log(
                    f"Session started. TFLOPs={max_tflops:.4f} | "
                    f"Reward/task={self.config['CHECKPOINT_REWARD_RATE']} MB", "success")
            except Exception as e:
                self.add_log(f"Session start failed: {e}", "error")
                self.state["running"] = False
            finally:
                self.state["restarting"] = False
        threading.Thread(target=_run, daemon=True).start()

    def do_federated_averaging(self):
        self.add_log("Federated Averaging: Starting...", "info")
        files = [os.path.join(self.upload_dir, f)
                 for f in os.listdir(self.upload_dir) if f.endswith(".ckpt")]
        if not files:
            self.add_log("No checkpoints found.", "warn"); return
        reader = tf.train.load_checkpoint(files[0])
        keys   = list(reader.get_variable_to_shape_map().keys())
        acc    = {k: None for k in keys}
        n_ok   = 0
        for ckpt in files:
            try:
                r = tf.train.load_checkpoint(ckpt)
                for k in keys:
                    t = r.get_tensor(k)
                    acc[k] = t.copy() if acc[k] is None else acc[k] + t
                n_ok += 1
                self.add_log(f"Aggregated: {os.path.basename(ckpt)}", "success")
            except Exception as e:
                self.add_log(f"Error: {os.path.basename(ckpt)}: {e}", "error")
        if n_ok == 0:
            self.add_log("ABORT: No valid checkpoints.", "error"); return
        vals = [tf.convert_to_tensor(acc[k] / n_ok) for k in keys]
        tf.raw_ops.Save(filename=tf.constant(self.global_ckpt_path),
                        tensor_names=tf.constant(list(keys)), data=vals,
                        name="fed_save")
        self.state["aggregation_done"] = True
        self.state["round"] += 1
        self.add_log(f"Global model saved. Round {self.state['round']}.", "success")
        if self.state["round"] > self.config["MAX_ROUNDS"]:
            self.state["finished"] = True
            self.add_log(f"All {self.config['MAX_ROUNDS']} rounds complete.", "success")


# ─────────────────────────────────────────────────────────────────────────────
#  Tenant Registry + JSON persistence
# ─────────────────────────────────────────────────────────────────────────────
_tenants: dict       = {}
_tenants_lock        = threading.Lock()
_tenant_counter: int = 0
_tenant_counter_lock = threading.Lock()

_rr_index: int = -1
_rr_lock        = threading.Lock()

_global_task_tenant_map:  dict = {}
_global_task_tenant_lock  = threading.Lock()

_task_download_map:  dict = {}
_task_download_lock  = threading.Lock()


def _next_n() -> int:
    global _tenant_counter
    with _tenant_counter_lock:
        _tenant_counter += 1
        return _tenant_counter


# def _save_tenants():
#     """Persist tenant credentials and config to tenants.json."""
#     data = []
#     with _tenants_lock:
#         for username, t in _tenants.items():
#             s: TenantSession = t["session"]
#             data.append({
#                 "username":        username,
#                 "password":        t["password"],
#                 "n":               t["n"],
#                 "max_tflops":      t["max_tflops"],
#                 "total_device_mbs": t["total_device_mbs"],
#                 "active":          t["active"],
#                 "config":          s.config,
#             })
#     try:
#         with open(TENANTS_JSON, "w") as f:
#             json.dump({"tenant_counter": _tenant_counter, "tenants": data}, f, indent=2)
#     except Exception as e:
#         print(f"[!] Could not save tenants.json: {e}")

def _save_tenants():
    """Persist tenant credentials and config to local tenants.json."""
    data = []
    with _tenants_lock:
        for username, t in _tenants.items():
            s: TenantSession = t["session"]
            data.append({
                "username":        username,
                "password":        t["password"],
                "n":               t["n"],
                "max_tflops":      t["max_tflops"],
                "total_device_mbs": t["total_device_mbs"],
                "active":          t["active"],
                "config":          s.config,
            })
    try:
        with open(TENANTS_JSON, "w") as f:
            json.dump({"tenant_counter": _tenant_counter, "tenants": data}, f, indent=2)
        print(f"[+] Saved tenants to {TENANTS_JSON}")
    except Exception as e:
        print(f"[!] Could not save tenants.json: {e}")


def _load_tenants():
    """Load tenants from local tenants.json on server startup."""
    global _tenant_counter
    if not os.path.exists(TENANTS_JSON):
        return
    try:
        with open(TENANTS_JSON) as f:
            saved = json.load(f)
        with _tenant_counter_lock:
            _tenant_counter = saved.get("tenant_counter", 0)
        count = 0
        for entry in saved.get("tenants", []):
            username = entry["username"]
            n        = entry["n"]
            sess     = TenantSession(username, n)
            sess.config.update(entry.get("config", {}))
            max_tflops = entry.get("max_tflops", 1.0)
            active     = entry.get("active", True)
            with _tenants_lock:
                _tenants[username] = {
                    "password":        entry["password"],
                    "n":               n,
                    "max_tflops":      max_tflops,
                    "total_device_mbs": entry.get("total_device_mbs", 150.0),
                    "active":          active,
                    "session":         sess,
                }
            if active:
                sess.start(max_tflops)
            count += 1
        print(f"[+] Loaded {count} tenant(s) from tenants.json")
    except Exception as e:
        print(f"[!] Could not load tenants.json: {e}")

# def _load_tenants():
#     """Load tenants from tenants.json on server startup."""
#     global _tenant_counter
#     if not os.path.exists(TENANTS_JSON):
#         return
#     try:
#         with open(TENANTS_JSON) as f:
#             saved = json.load(f)
#         with _tenant_counter_lock:
#             _tenant_counter = saved.get("tenant_counter", 0)
#         for entry in saved.get("tenants", []):
#             username = entry["username"]
#             n        = entry["n"]
#             sess     = TenantSession(username, n)
#             sess.config.update(entry.get("config", {}))
#             with _tenants_lock:
#                 _tenants[username] = {
#                     "password":        entry["password"],
#                     "n":               n,
#                     "max_tflops":      entry.get("max_tflops", 1.0),
#                     "total_device_mbs": entry.get("total_device_mbs", 150.0),
#                     "active":          entry.get("active", True),
#                     "session":         sess,
#                 }
#         print(f"[+] Loaded {len(saved.get('tenants',[]))} tenant(s) from tenants.json")
#     except Exception as e:
#         print(f"[!] Could not load tenants.json: {e}")


def create_tenant(username: str, password: str, max_tflops: float,
                  total_device_mbs: float, active: bool = True):
    with _tenants_lock:
        if username in _tenants:
            return False, "Username already exists."
        n    = _next_n()
        sess = TenantSession(username, n)
        sess.config["CHECKPOINT_REWARD_RATE"] = _calc_reward(
            total_device_mbs, sess.config["MAX_CLIENTS"])
        _tenants[username] = {
            "password":        password,
            "max_tflops":      float(max_tflops),
            "total_device_mbs": float(total_device_mbs),
            "active":          active,
            "n":               n,
            "session":         sess,
        }
        if active:
            sess.start(float(max_tflops))
    _save_tenants()
    return True, n


def _get_next_tenant_rr():
    global _rr_index
    with _tenants_lock:
        keys = list(_tenants.keys())
        snap = {k: _tenants[k] for k in keys}
    if not keys:
        return None, None
    with _rr_lock:
        for i in range(len(keys)):
            idx = (_rr_index + 1 + i) % len(keys)
            u   = keys[idx]
            t   = snap.get(u)
            if not t: continue
            s: TenantSession = t["session"]
            if (t["active"] and s.state.get("running")
                    and not s.state.get("paused")
                    and not s.state.get("finished")
                    and s.remaining_tflops >= _tflops_per_task(s.config)):
                _rr_index = idx
                return u, t
    return None, None



# ─────────────────────────────────────────────────────────────────────────────
#  Auth decorators  — token-only, no cookie fallback
# ─────────────────────────────────────────────────────────────────────────────
def admin_required(f):
    def w(*a, **kw):
        if _get_auth().get("role") != "admin":
            return redirect(url_for("login"))
        return f(*a, **kw)
    w.__name__ = f.__name__; return w

def tenant_required(f):
    def w(*a, **kw):
        if _get_auth().get("role") != "tenant":
            return redirect(url_for("login"))
        return f(*a, **kw)
    w.__name__ = f.__name__; return w

def _cur_tenant() -> str:
    return _get_auth().get("user", "")


# ─────────────────────────────────────────────────────────────────────────────
#  Auth routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    # Without a token header we can't know who they are — send to login
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        u = request.form.get("username", "").strip()
        p = request.form.get("password", "")

        if u == ADMIN_USERNAME and p == ADMIN_PASSWORD:
            token = _issue_token(u, "admin")
            # Return a page that stores the token in sessionStorage then redirects
            return _token_redirect(token, "/admin")

        t = _tenants.get(u)
        if t and t["password"] == p:
            if not t["active"]:
                return render_template("login.html", error="Account inactive.")
            token = _issue_token(u, "tenant")
            return _token_redirect(token, "/tenant")

        return render_template("login.html", error="Invalid credentials.")
    return render_template("login.html")


def _token_redirect(token: str, dest: str) -> str:
    """Return an HTML page that saves the token to sessionStorage then redirects.
    sessionStorage is per-tab so different tenants in different tabs stay isolated."""
    return f"""<!DOCTYPE html><html><head>
<script>
  sessionStorage.setItem('fractal_auth_token', '{token}');
  window.location.href = '{dest}';
</script></head><body></body></html>"""


@app.route("/logout")
def logout():
    token = request.headers.get("X-Auth-Token", "")
    if token:
        _revoke_token(token)
    return """<!DOCTYPE html><html><head>
<script>
  sessionStorage.removeItem('fractal_auth_token');
  window.location.href = '/login';
</script></head><body></body></html>"""


# ─────────────────────────────────────────────────────────────────────────────
#  Admin routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/admin")
def admin_dashboard():
    # Page route — just serve the HTML.
    # The admin.html JS sends X-Auth-Token on every API call; those are protected.
    return render_template("admin.html")


@app.route("/api/admin/tenants")
@admin_required
def api_list_tenants():
    with _tenants_lock:
        res = []
        for u, t in _tenants.items():
            s: TenantSession = t["session"]
            res.append({
                "username":        u,
                "n":               t["n"],
                "model_id":        s.model_id,
                "data_id":         s.data_id,
                "max_tflops":      t["max_tflops"],
                "total_device_mbs": t["total_device_mbs"],
                "reward_per_task": s.config.get("CHECKPOINT_REWARD_RATE", 0.0),
                "active":          t["active"],
                "running":         s.state.get("running",    False),
                "restarting":      s.state.get("restarting", False),
                "finished":        s.state.get("finished",   False),
                "paused":          s.state.get("paused",     True),
                "round":           s.state.get("round",      1),
                "max_rounds":      s.config.get("MAX_ROUNDS", 5),
                "clients_uploaded": s.state.get("clients_uploaded", 0),
                "num_clients":     s.config.get("MAX_CLIENTS", 0),
                "remaining_tflops": round(s.remaining_tflops, 4),
                "tflops_per_task":  round(_tflops_per_task(s.config), 6),
            })
    return jsonify(res)


@app.route("/api/admin/tenant", methods=["POST"])
@admin_required
def api_create_tenant():
    d = request.get_json(force=True, silent=True) or {}
    u, p = d.get("username", "").strip(), d.get("password", "").strip()
    if not u or not p:
        return jsonify({"error": "Username and password required."}), 400
    ok, r = create_tenant(u, p,
                          float(d.get("max_tflops",       1.0)),
                          float(d.get("total_device_mbs", 150.0)),
                          bool(d.get("active", True)))
    if not ok:
        return jsonify({"error": r}), 409
    return jsonify({"success": True, "n": r,
                    "model_id": f"{r:04d}", "data_id": f"{r+1000:04d}"})


@app.route("/api/admin/tenant/<username>/toggle", methods=["POST"])
@admin_required
def api_toggle_tenant(username):
    with _tenants_lock:
        t = _tenants.get(username)
        if not t: return jsonify({"error": "Not found."}), 404
        t["active"] = not t["active"]
        active = t["active"]
    _save_tenants()
    return jsonify({"active": active})


@app.route("/api/admin/tenant/<username>/update", methods=["POST"])
@admin_required
def api_update_tenant(username):
    d = request.get_json(force=True, silent=True) or {}
    with _tenants_lock:
        t = _tenants.get(username)
        if not t: return jsonify({"error": "Not found."}), 404
        if "max_tflops" in d:
            t["max_tflops"] = float(d["max_tflops"])
        if "total_device_mbs" in d:
            t["total_device_mbs"] = float(d["total_device_mbs"])
            # Recalculate reward whenever total_device_mbs changes
            t["session"].config["CHECKPOINT_REWARD_RATE"] = _calc_reward(
                t["total_device_mbs"], t["session"].config["MAX_CLIENTS"])
        if "password" in d and d["password"]:
            t["password"] = str(d["password"])
        if "active" in d:
            t["active"] = bool(d["active"])
    _save_tenants()
    return jsonify({"success": True})


@app.route("/api/admin/tenant/<username>/delete", methods=["POST"])
@admin_required
def api_delete_tenant(username):
    # 1. Remove the tenant from active memory
    with _tenants_lock:
        if username not in _tenants:
            return jsonify({"error": "Not found."}), 404
        del _tenants[username]
        
    # 2. Update the persistent ledger (Explicit Firestore Delete or File-based sync)
    if not USE_FIRESTORE:
        _save_tenants()
    else:
        try:
            _db.collection("tenants").document(username).delete()
            _save_tenants() # Sync remaining states if needed
        except Exception as e:
            print(f"[!] FRACTAL OS: Failed to delete tenant doc '{username}' from Firestore: {e}")
    
    # 3. Physically wipe the tenant's data silo
    tenant_dir = os.path.join(DATA_DIR, "tenants", username)
    if os.path.exists(tenant_dir):
        try:
            shutil.rmtree(tenant_dir)
            print(f"[*] FRACTAL OS: Purged physical data silo for tenant '{username}'")
        except Exception as e:
            print(f"[!] FRACTAL OS: Failed to delete directory for '{username}': {e}")
            return jsonify({"success": True, "warning": "Tenant deleted from registry, but folder deletion failed."})

    return jsonify({"success": True})


@app.route("/api/admin/tenant/<username>/status")
@admin_required
def api_admin_tenant_status(username):
    with _tenants_lock:
        t = _tenants.get(username)
    if not t: return jsonify({"error": "Not found."}), 404
    st = t["session"].get_status()
    st["max_tflops"]       = t["max_tflops"]
    st["total_device_mbs"] = t["total_device_mbs"]
    return jsonify(st)


# ─────────────────────────────────────────────────────────────────────────────
#  Tenant routes
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/tenant")
def tenant_dashboard():
    # Page route — just serve the HTML.
    # The tenant.html JS sends X-Auth-Token on every API call; those are protected.
    return render_template("tenant.html")


@app.route("/api/tenant/status")
@tenant_required
def api_tenant_status():
    u = _cur_tenant()
    t = _tenants.get(u)
    if not t: return jsonify({"error": "Not found."}), 404
    st = t["session"].get_status()
    st["max_tflops"]       = t["max_tflops"]
    st["total_device_mbs"] = t["total_device_mbs"]
    st["username"]         = u
    return jsonify(st)


@app.route("/api/tenant/config", methods=["POST"])
@tenant_required
def api_tenant_config():
    u = _cur_tenant()
    t = _tenants.get(u)
    if not t: return jsonify({"error": "Not found."}), 404
    s: TenantSession = t["session"]
    payload  = request.get_json(force=True, silent=True) or {}
    proposed = dict(s.config)

    for k in ["MAX_CLIENTS","MAX_ROUNDS","N_BINS","ITEMS_PER_BIN",
              "NUM_EPOCHS","BATCH_SIZE","NUM_CLASSES"]:
        if k in payload: proposed[k] = int(payload[k])
    if "IMG_SIZE" in payload:
        proposed["INPUT_SHAPE"] = [int(payload["IMG_SIZE"])] * 2
    for k in ["REPETITIVE_TRAINING", "AUTO_DELETE_CHECKPOINTS"]:
        if k in payload: proposed[k] = bool(payload[k])
    for k in ["DATASET","ARCHITECTURE"]:
        if k in payload: proposed[k] = str(payload[k])

    # Reward always derived from admin's total_device_mbs / MAX_CLIENTS
    proposed["CHECKPOINT_REWARD_RATE"] = _calc_reward(
        t["total_device_mbs"], proposed["MAX_CLIENTS"])

    # TFLOPs guard
    projected = _tflops_full_session(proposed)
    if projected > t["max_tflops"] + 1e-9:
        return jsonify({
            "error": (f"Rejected — projected {projected:.4f} TFLOPs exceeds "
                      f"your budget of {t['max_tflops']:.4f} TFLOPs. "
                      f"Reduce Bins, Items/Bin, Epochs, or Rounds."),
            "projected_tflops": round(projected, 6),
            "max_tflops": t["max_tflops"],
        }), 400

    s.config = proposed
    _save_tenants()
    return jsonify({"success": True, "config": s.config,
                    "projected_tflops": round(projected, 6),
                    "max_tflops": t["max_tflops"]})


@app.route("/api/tenant/start", methods=["POST"])
@tenant_required
def api_tenant_start():
    u = _cur_tenant()
    t = _tenants.get(u)
    if not t: return jsonify({"error": "Not found."}), 404
    if not t["active"]: return jsonify({"error": "Account inactive."}), 403
    s: TenantSession = t["session"]
    if s.state.get("restarting"): return jsonify({"error": "Already starting."}), 409
    s.start(t["max_tflops"])
    return jsonify({"status": "Starting..."})


@app.route("/api/tenant/pause", methods=["POST"])
@tenant_required
def api_tenant_pause():
    u = _cur_tenant()
    t = _tenants.get(u)
    if not t: return jsonify({"error": "Not found."}), 404
    s: TenantSession = t["session"]
    s.state["paused"] = not s.state["paused"]
    s.add_log(f"Session {'paused' if s.state['paused'] else 'resumed'} by tenant.", "warn")
    return jsonify({"paused": s.state["paused"]})


# ─────────────────────────────────────────────────────────────────────────────
#  Device routes  (no auth — Android clients)
# ─────────────────────────────────────────────────────────────────────────────
@app.route("/api/task/current")
def get_current_task():
    device_id = request.args.get("device_id", "unknown")
    username, t = _get_next_tenant_rr()
    if not t:
        return jsonify({"error": "No active sessions available."}), 503
    s: TenantSession = t["session"]

    if len(s.assigned_devices) >= s.config["MAX_CLIENTS"]:
        s.add_log(f"Device {device_id} rejected — client cap.", "warn")
        return jsonify({"error": "Max client limit reached."}), 403
    if not s.config["REPETITIVE_TRAINING"] and device_id in s.assigned_devices:
        return jsonify({"error": "Already served this device."}), 403

    cost = _tflops_per_task(s.config)
    with s._tflops_lock:
        if s.remaining_tflops < cost:
            return jsonify({"error": "TFLOPs quota exhausted."}), 403
        s.remaining_tflops -= cost

    with s.task_queue_lock:
        if not s.task_queue:
            if s._bin_segment_cache:
                s.add_log("Queue empty — refilling circularly.", "info")
                s._build_task_queue()
            else:
                with s._tflops_lock: s.remaining_tflops += cost
                return jsonify({"error": "No tasks available."}), 404
        task = s.task_queue.popleft()

    s.assigned_devices.add(device_id)
    s.state["segments_dispatched"] += 1

    with s._task_device_map_lock:
        s._task_device_map[task["task_Id"]] = {
            "device_id":  device_id,
            "reward_mbs": float(task["reward_rate"]),
        }
    with _global_task_tenant_lock:
        _global_task_tenant_map[task["task_Id"]] = username
    with _task_download_lock:
        _task_download_map[task["task_Id"]] = username

    s.add_log(
        f"Task -> {device_id} | tenant={username} | "
        f"{len(s.assigned_devices)}/{s.config['MAX_CLIENTS']} | "
        f"{s.remaining_tflops:.4f} TF left | reward={task['reward_rate']} MB",
        "success",
    )
    return jsonify(task)


@app.route("/api/model/upload", methods=["POST"])
def upload_checkpoint():
    task_Id   = request.form.get("task_Id") or request.form.get("taskId") or "unknown"
    device_id = request.form.get("device_id", "unknown_device")

    tenant_id = None
    with _global_task_tenant_lock:
        tenant_id = _global_task_tenant_map.pop(task_Id, None)
    with _task_download_lock:
        _task_download_map.pop(task_Id, None)

    if not tenant_id:
        # Fallback: if server reloaded, identify tenant via first 4 chars of task_Id (model_id prefix)
        if len(task_Id) >= 4:
            prefix = task_Id[:4]
            with _tenants_lock:
                for username, t in _tenants.items():
                    s_tmp = t.get("session")
                    if s_tmp and s_tmp.model_id == prefix:
                        tenant_id = username
                        break

    if not tenant_id: return "Unknown task_Id.", 400

    with _tenants_lock:
        t = _tenants.get(tenant_id)
    if not t: return "Tenant not found.", 404
    s: TenantSession = t["session"]

    if "model_file" not in request.files: return "No file.", 400
    f = request.files["model_file"]
    if f.filename == "": return "No file selected.", 400

    with s._task_device_map_lock:
        di = s._task_device_map.pop(task_Id, None)
    
    target_device_id = di.get("device_id") if di else device_id
    reward_mbs       = di.get("reward_mbs") if di else float(s.config.get("CHECKPOINT_REWARD_RATE", 33.3333) or 33.3333)

    ts  = int(datetime.now().timestamp() * 1000)
    fn  = f"{task_Id}_{target_device_id}_{ts}.ckpt"
    f.save(os.path.join(s.upload_dir, fn))
    s.add_log(f"Received: {fn}", "success")

    if target_device_id and target_device_id.lower() not in ("unknown", "unknown_device", ""):
        threading.Thread(target=credit_mbs_for_device,
                         args=(target_device_id, reward_mbs, s),
                         daemon=True).start()
    else:
        s.add_log(f"Skipping reward: No valid device ID resolved for task {task_Id}.", "warn")

    received = [x for x in os.listdir(s.upload_dir) if x.endswith(".ckpt")]
    s.state["clients_uploaded"] = len(received)
    s.add_log(f"Checkpoints: {len(received)}/{s.config['MAX_CLIENTS']}", "info")

    if len(received) >= s.config["MAX_CLIENTS"]:
        s.add_log("All clients reported — triggering Federated Averaging.", "info")
        def _agg():
            s.do_federated_averaging()
            import time; time.sleep(2)
            s.assigned_devices.clear()
            s.state.update({"clients_uploaded": 0, "segments_dispatched": 0,
                            "aggregation_done": False})
            if s.config.get("AUTO_DELETE_CHECKPOINTS", False):
                for x in [x for x in os.listdir(s.upload_dir) if x.endswith(".ckpt")]:
                    try: os.remove(os.path.join(s.upload_dir, x))
                    except OSError: pass
            else:
                s.add_log("Auto-delete disabled: keeping round client checkpoints in uploads folder.", "info")
            if not s.state["finished"]:
                s.add_log("Round reset. Ready for next round.", "info")
        threading.Thread(target=_agg, daemon=True).start()

    return "Upload Successful", 200


def _resolve_tenant(tenant_id: str, task_id: str, filename: str = "") -> str:
    """Resolve the owning tenant for a download request.

    Resolution order:
      1. Explicit tenant_id param (if valid).
      2. task_id lookup in _task_download_map.
      3. Parse the tenant username prefix embedded in the filename
         (files are named  {username}_images_NNN.bin / {username}_labels_NNN.bin).
      4. Filesystem scan fallback (last resort).
    """
    if tenant_id:
        with _tenants_lock:
            if tenant_id in _tenants:
                return tenant_id

    if task_id:
        with _task_download_lock:
            found = _task_download_map.get(task_id, "")
        if found:
            return found

    # Parse username prefix:  "{username}_images_NNN.bin" or "{username}_labels_NNN.bin"
    if filename:
        # All tenant-prefixed files follow the pattern  <username>_images_NNN.bin
        for sep in ("_images_", "_labels_"):
            if sep in filename:
                candidate = filename.split(sep)[0]
                with _tenants_lock:
                    if candidate in _tenants:
                        return candidate
                break

        # Last-resort: scan every tenant's download dir
        with _tenants_lock:
            items = list(_tenants.items())
        for u, t in items:
            fp = os.path.join(t["session"].download_dir, filename)
            if os.path.exists(fp):
                return u

    return ""


@app.route("/download/images")
def download_images():
    fn  = request.args.get("filename", "").strip()
    if not fn:
        return "filename parameter required.", 400
    tid = _resolve_tenant(request.args.get("tenant_id", "").strip(),
                          request.args.get("task_id",   "").strip(), fn)
    if not tid:
        return f"Cannot resolve tenant for '{fn}'. Include tenant_id or task_id.", 400
    t = _tenants.get(tid)
    if not t:
        return f"Tenant '{tid}' not found.", 404
    fp = os.path.join(t["session"].download_dir, fn)
    if not os.path.exists(fp):
        return f"File '{fn}' not found in tenant '{tid}' bins dir.", 404
    return send_file(fp, as_attachment=True)


@app.route("/download/labels")
def download_labels():
    fn  = request.args.get("filename", "").strip()
    if not fn:
        return "filename parameter required.", 400
    tid = _resolve_tenant(request.args.get("tenant_id", "").strip(),
                          request.args.get("task_id",   "").strip(), fn)
    if not tid:
        return f"Cannot resolve tenant for '{fn}'. Include tenant_id or task_id.", 400
    t = _tenants.get(tid)
    if not t:
        return f"Tenant '{tid}' not found.", 404
    fp = os.path.join(t["session"].download_dir, fn)
    if not os.path.exists(fp):
        return f"File '{fn}' not found in tenant '{tid}' bins dir.", 404
    return send_file(fp, as_attachment=True)


@app.route("/download/model")
def download_model():
    fn = request.args.get("filename", SHARED_MODEL_FILENAME)
    fp = os.path.join(SHARED_MODEL_DIR, fn)
    return send_file(fp, as_attachment=True) if os.path.exists(fp) \
        else (f"Model '{fn}' not found.", 404)

    # ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    USE_FIRESTORE = True

    _load_tenants()   # Restore tenants from Firestore or local json on startup
    print("=" * 60)
    print("  FRACTAL  .  Multi-Tenant Federated Learning Server")
    print("  Admin  -> http://127.0.0.1:5000/admin")
    print("  Tenant -> http://127.0.0.1:5000/tenant")
    print(f"  Persistence: {'Firebase Firestore' if USE_FIRESTORE else 'Local JSON (secrets/tenants.json)'}")
    print(f"  Shared model: {SHARED_MODEL_FILENAME}")
    print(f"  Auth: per-login token in sessionStorage (tab-isolated)")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)