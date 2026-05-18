"""
test_firebase_reward.py
=======================
Integration test suite for the Fractal Firebase Reward system.

Run (default suite):
    python test_firebase_reward.py

Run (single device via CLI):
    python test_firebase_reward.py <device_id> [mbs]
"""

import os
import sys
import importlib.util

# ── Dynamic module loader ─────────────────────────────────────────────────────
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODULE_PATH  = os.path.join(CURRENT_DIR, "src", "fractal_server", "firebase_reward.py")

# Allow override via environment variable for flexible project layouts
MODULE_PATH = os.environ.get("FIREBASE_REWARD_PATH", MODULE_PATH)

try:
    spec = importlib.util.spec_from_file_location("firebase_reward", MODULE_PATH)
    firebase_reward = importlib.util.module_from_spec(spec)
    sys.modules["firebase_reward"] = firebase_reward
    spec.loader.exec_module(firebase_reward)

    credit_mbs_for_device   = firebase_reward.credit_mbs_for_device
    _get_email_by_device_id = firebase_reward._get_email_by_device_id
    _db                     = firebase_reward._db

except Exception as e:
    print(f"[FATAL] Could not load firebase_reward from:\n  {MODULE_PATH}\n  Error: {e}")
    sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _section(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)

def _pass(msg: str):
    print(f"  [PASS] {msg}")

def _fail(msg: str):
    print(f"  [FAIL] {msg}")

def _info(msg: str):
    print(f"  [INFO] {msg}")


# ── Test Cases ────────────────────────────────────────────────────────────────

def test_firestore_connection():
    """Test 1 – Firestore client is alive."""
    _section("Test 1 · Firestore Connection")
    if _db:
        _pass("Firestore client initialized.")
        return True
    else:
        _fail("Firestore client is None – check service-account.json path.")
        return False


def test_device_lookup(device_id: str):
    """Test 2 – Resolve device_id → email from registered_devices."""
    _section(f"Test 2 · Device Lookup  [{device_id}]")
    _info(f"Querying registered_devices for '{device_id}'...")

    email = _get_email_by_device_id(device_id)
    if email:
        _pass(f"Device resolved to email: '{email}'")
        return True, email
    else:
        _fail(f"No email found for device '{device_id}'.")
        return False, None


def test_reward_crediting(device_id: str, mbs: float, email: str = None):
    """Test 3 – Credit mbs to the user linked with device_id."""
    _section(f"Test 3 · Reward Crediting  [{device_id}  +{mbs} MB]")

    if email:
        _info(f"Expected target user : '{email}'")
    _info(f"Crediting {mbs} MB...")

    success = credit_mbs_for_device(device_id, mbs)
    if success:
        _pass(f"Reward transaction completed for device '{device_id}'.")
    else:
        _fail(f"Reward transaction failed for device '{device_id}'.")
    return success


def test_invalid_device_ids():
    """Test 4 – Invalid / empty device IDs must be rejected gracefully."""
    _section("Test 4 · Invalid Device ID Handling")

    bad_ids = ["", "unknown", "UNKNOWN", "   ", None]
    all_ok  = True

    for bad_id in bad_ids:
        result = credit_mbs_for_device(bad_id, 10.0)  # type: ignore[arg-type]
        if not result:
            _pass(f"Correctly rejected invalid ID: {repr(bad_id)}")
        else:
            _fail(f"Accepted an invalid ID (should have rejected): {repr(bad_id)}")
            all_ok = False

    return all_ok


def test_cap_enforcement(device_id: str):
    """
    Test 5 – Cap enforcement.
    Credits an absurdly large value; the function must clamp it to 2048 MB.
    NOTE: This writes to Firestore. Run against a test account only.
    """
    _section(f"Test 5 · 2 GB Cap Enforcement  [{device_id}]")
    _info("Attempting to credit 999,999 MB (should be capped at 2048 MB total)...")

    success = credit_mbs_for_device(device_id, 999_999.0)
    if success:
        _pass("Cap enforcement executed without error (check Firestore balance ≤ 2048 MB).")
    else:
        _fail("Unexpected failure during cap-enforcement test.")
    return success


# ── Test Suite Runner ─────────────────────────────────────────────────────────

def run_test_suite(target_device: str = "7b06c36114aebc82", reward_mbs: float = 250.0):
    _section("FRACTAL FIREBASE REWARD — INTEGRATION TEST SUITE")

    results = {}

    # 1. Connection
    results["connection"] = test_firestore_connection()
    if not results["connection"]:
        print("\n[!] Aborting: no Firestore connection.")
        _print_summary(results)
        return

    # 2. Device lookup
    ok, email = test_device_lookup(target_device)
    results["device_lookup"] = ok

    # 3. Reward crediting
    results["reward_credit"] = test_reward_crediting(target_device, reward_mbs, email)

    # 4. Invalid ID rejection
    results["invalid_ids"] = test_invalid_device_ids()

    # 5. Cap enforcement  (comment out if you don't want to write large values)
    # results["cap_enforcement"] = test_cap_enforcement(target_device)

    _print_summary(results)


def _print_summary(results: dict):
    _section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total  = len(results)

    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")

    print()
    print(f"  Result: {passed}/{total} tests passed.")
    print("=" * 60)


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # CLI mode: python test_firebase_reward.py <device_id> [mbs]
        dev_id  = sys.argv[1]
        mbs_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
        _section(f"CLI MODE  ·  device={dev_id}  mbs={mbs_arg}")
        credit_mbs_for_device(dev_id, mbs_arg)
    else:
        # Default: full test suite with the known test device
        run_test_suite(
            target_device="7b06c36114aebc82",
            reward_mbs=250.0,
        )