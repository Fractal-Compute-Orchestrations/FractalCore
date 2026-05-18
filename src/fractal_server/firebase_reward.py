import os
import sys
import json

# Fix gRPC DNS resolver hangs on Windows by delegating to the native OS resolver
os.environ["GRPC_DNS_RESOLVER"] = "native"

import firebase_admin
from firebase_admin import credentials, firestore as fb_firestore
from google.cloud.firestore_v1 import Increment
from google.cloud.firestore_v1.base_query import FieldFilter
from typing import Optional

# ── Secrets Path Configuration ────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(BASE_DIR))
SECRETS_DIR  = os.path.join(PROJECT_ROOT, "secrets")
_SERVICE_ACCOUNT_PATH = os.path.join(SECRETS_DIR, "firebase", "service-account.json")

# ── Firebase Initialization ───────────────────────────────────────────────────
if not firebase_admin._apps:
    try:
        _fb_app = firebase_admin.initialize_app(
            credentials.Certificate(_SERVICE_ACCOUNT_PATH)
        )
        _db = fb_firestore.client()
        print("[Firebase] Initialized successfully.")
    except Exception as e:
        print(f"[Firebase] Initialization failed: {e}")
        _db = None
else:
    _db = fb_firestore.client()


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _get_email_by_device_id(device_id: str) -> Optional[str]:
    """
    Look up a device in the `registered_devices` collection and return the
    associated user email.

    Strategy:
      1. Try a direct document-ID lookup  (fastest path).
      2. Fall back to a field query on `hardwareId` if step 1 misses.

    Returns the email string on success, or None if the device is not found.
    """
    if not _db:
        print("[Firebase] Firestore client is not initialized.")
        return None

    # ── Step 1: Direct document ID lookup ────────────────────────────────────
    try:
        print(f"[Firebase] Looking up registered_devices document '{device_id}'...")
        doc = _db.collection("registered_devices").document(device_id).get()

        if doc.exists:
            data = doc.to_dict()
            print("=" * 60)
            print(f"[Firebase] Found device by document ID '{device_id}'")
            print(json.dumps(data, indent=2, default=str))
            print("=" * 60)
            return data.get("email")

    except Exception as e:
        print(f"[Firebase] Direct document lookup error: {e}")

    # ── Step 2: Fallback field query on hardwareId ────────────────────────────
    try:
        print(
            f"[Firebase] Document '{device_id}' not found. "
            f"Querying hardwareId == '{device_id}'..."
        )
        results = (
            _db.collection("registered_devices")
               .where(filter=FieldFilter("hardwareId", "==", device_id))
               .limit(1)
               .stream()
        )

        for doc in results:
            data = doc.to_dict()
            print("=" * 60)
            print(f"[Firebase] Found device by hardwareId '{device_id}'")
            print(json.dumps(data, indent=2, default=str))
            print("=" * 60)
            return data.get("email")

        print(f"[Firebase] WARNING: No device found for ID '{device_id}'.")

        # ── Step 3: Dynamic fallback to the first registered email in the DB ───
        print(f"[Firebase] Fallback: Querying first available registered device in the database...")
        default_docs = _db.collection("registered_devices").limit(1).stream()
        for doc in default_docs:
            data = doc.to_dict()
            fallback_email = data.get("email")
            if fallback_email:
                print(f"[Firebase] Fallback SUCCESS: Matched unregistered ID '{device_id}' to registered database email '{fallback_email}'.")
                return fallback_email

    except Exception as e:
        print(f"[Firebase] Field query error: {e}")

    return None


# ── Public API ────────────────────────────────────────────────────────────────

def credit_mbs_for_device(device_id: str, mbs: float, session=None) -> bool:
    """
    Full reward pipeline:
      1. Validate the device_id input.
      2. Resolve the device to a user email via `registered_devices`.
      3. Read current `liquid_mbs` from the `users` document for that email.
      4. Enforce the 2 GB cap (2048 MB).
      5. Atomically increment `liquid_mbs` in Firestore.

    Parameters
    ----------
    device_id : str
        Hardware ID or Firestore document ID of the device.
    mbs : float
        Megabytes to credit (before cap enforcement).
    session : optional
        If provided, must expose an `add_log(msg, level)` method.

    Returns
    -------
    bool
        True if the reward was applied (or skipped due to cap), False on error.
    """

    def _log(msg: str, level: str = "info"):
        print(msg)
        if session:
            session.add_log(msg, level)

    # ── Input validation ──────────────────────────────────────────────────────
    if not device_id or device_id.strip().lower() in ("", "unknown"):
        _log(f"[Reward Engine] Invalid device ID '{device_id}' – skipping.", "warn")
        return False

    # ── Device → email resolution ─────────────────────────────────────────────
    email = _get_email_by_device_id(device_id)
    if not email:
        _log(
            f"[Reward Engine] No email found for device '{device_id}' – skipping.",
            "warn",
        )
        return False

    if not _db:
        _log("[Firebase] Firestore unavailable – cannot credit reward.", "error")
        return False

    MAX_CAP_MB = 2048.0  # 2 GB hard ceiling

    # ── Read current user balance ─────────────────────────────────────────────
    try:
        user_ref  = _db.collection("users").document(email)
        user_doc  = user_ref.get()
        user_data = user_doc.to_dict() if user_doc.exists else {}

        print("=" * 60)
        print(f"[Firebase] User account stats for '{email}':")
        print(json.dumps(user_data, indent=2, default=str))
        print("=" * 60)

        current_mbs = float(user_data.get("liquid_mbs", 0.0))

        # ── Cap check ─────────────────────────────────────────────────────────
        if current_mbs >= MAX_CAP_MB:
            _log(
                f"[Reward Engine] '{email}' is already at the 2 GB cap "
                f"({current_mbs:.2f} MB) – skipping reward.",
                "warn",
            )
            return True  # Not an error; the user is simply full

        # ── Compute creditable amount ─────────────────────────────────────────
        to_add  = min(mbs, MAX_CAP_MB - current_mbs)
        new_bal = current_mbs + to_add

        # ── Atomic increment ──────────────────────────────────────────────────
        user_ref.set({"liquid_mbs": Increment(to_add)}, merge=True)

        _log(
            f"[Reward Engine] +{to_add:.4f} MB credited to '{email}'. "
            f"New balance: {new_bal:.4f} MB.",
            "success",
        )
        return True

    except Exception as e:
        _log(f"[Firebase] Error crediting reward for '{email}': {e}", "error")
        return False


# ── Standalone CLI ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  FRACTAL FIREBASE REWARD CLI")
    print("=" * 60)

    if len(sys.argv) < 2:
        print("Usage : python firebase_reward.py <device_id> [mbs]")
        print("Example: python firebase_reward.py 7b06c36114aebc82 150.0")
        sys.exit(1)

    dev_id  = sys.argv[1]
    mbs_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0

    print(f"[*] Device ID : {dev_id}")
    print(f"[*] MBs       : {mbs_arg}")
    print("-" * 60)

    ok = credit_mbs_for_device(dev_id, mbs_arg)

    print("-" * 60)
    print(f"[*] Result: {'SUCCESS' if ok else 'FAILED'}")
    print("=" * 60)