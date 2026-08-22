"""
test_firebase_reward.py
=======================
Integration test suite for the Fractal Firebase Reward system.

Run (default suite):
    python scripts/test_firebase_reward.py

Run (single device via CLI):
    python scripts/test_firebase_reward.py <device_id> [mbs]
"""

import os
import sys
import importlib.util

# Dynamic module loader
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == "scripts" else CURRENT_DIR
MODULE_PATH = os.path.join(PROJECT_ROOT, "src", "fractal_server", "firebase_reward.py")

# Allow override via environment variable for flexible project layouts
MODULE_PATH = os.environ.get("FIREBASE_REWARD_PATH", MODULE_PATH)

credit_mbs_for_device = None
_get_email_by_device_id = None
_db = None

try:
    if os.path.exists(MODULE_PATH):
        spec = importlib.util.spec_from_file_location("firebase_reward", MODULE_PATH)
        if spec and spec.loader:
            firebase_reward = importlib.util.module_from_spec(spec)
            sys.modules["firebase_reward"] = firebase_reward
            spec.loader.exec_module(firebase_reward)

            credit_mbs_for_device = getattr(
                firebase_reward, "credit_mbs_for_device", None
            )
            _get_email_by_device_id = getattr(
                firebase_reward, "_get_email_by_device_id", None
            )
            _db = getattr(firebase_reward, "_db", None)
except Exception as e:
    print(f"[WARN] Could not load firebase_reward from: {MODULE_PATH} (Error: {e})")


# Helpers


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


# Test Cases (Integration Checks)


def check_firestore_connection():
    """Check 1 - Firestore client is alive."""
    _section("Test 1 - Firestore Connection")
    if _db:
        _pass("Firestore client initialized.")
        return True
    else:
        _fail("Firestore client is None - check service-account.json path.")
        return False


def check_device_lookup(device_id: str):
    """Check 2 - Resolve device_id -> email from registered_devices."""
    _section(f"Test 2 - Device Lookup  [{device_id}]")
    _info(f"Querying registered_devices for '{device_id}'...")

    if not _get_email_by_device_id:
        _fail("Module _get_email_by_device_id is not loaded.")
        return False, None

    email = _get_email_by_device_id(device_id)
    if email:
        _pass(f"Device resolved to email: '{email}'")
        return True, email
    else:
        _fail(f"No email found for device '{device_id}'.")
        return False, None


def check_reward_crediting(device_id: str, mbs: float, email: str | None = None):
    """Check 3 - Credit mbs to the user linked with device_id."""
    _section(f"Test 3 - Reward Crediting  [{device_id}  +{mbs} MB]")

    if email:
        _info(f"Expected target user : '{email}'")
    _info(f"Crediting {mbs} MB...")

    if not credit_mbs_for_device:
        _fail("Module credit_mbs_for_device is not loaded.")
        return False

    success = credit_mbs_for_device(device_id, mbs)
    if success:
        _pass(f"Reward transaction completed for device '{device_id}'.")
    else:
        _fail(f"Reward transaction failed for device '{device_id}'.")
    return success


def check_invalid_device_ids():
    """Check 4 - Invalid / empty device IDs must be rejected gracefully."""
    _section("Test 4 - Invalid Device ID Handling")

    if not credit_mbs_for_device:
        _fail("Module credit_mbs_for_device is not loaded.")
        return False

    bad_ids = ["", "unknown", "UNKNOWN", "   ", None]
    all_ok = True

    for bad_id in bad_ids:
        result = credit_mbs_for_device(bad_id, 10.0)  # type: ignore[arg-type]
        if not result:
            _pass(f"Correctly rejected invalid ID: {bad_id!r}")
        else:
            _fail(f"Accepted an invalid ID (should have rejected): {bad_id!r}")
            all_ok = False

    return all_ok


def check_cap_enforcement(device_id: str):
    """Check 5 - Cap enforcement."""
    _section(f"Test 5 - 2 GB Cap Enforcement  [{device_id}]")
    _info("Attempting to credit 999,999 MB (should be capped at 2048 MB total)...")

    if not credit_mbs_for_device:
        _fail("Module credit_mbs_for_device is not loaded.")
        return False

    success = credit_mbs_for_device(device_id, 999_999.0)
    if success:
        _pass(
            "Cap enforcement executed without error (check Firestore balance <= 2048 MB)."
        )
    else:
        _fail("Unexpected failure during cap-enforcement test.")
    return success


# Test Suite Runner


def run_test_suite(target_device: str = "7b06c36114aebc82", reward_mbs: float = 250.0):
    _section("FRACTAL FIREBASE REWARD - INTEGRATION TEST SUITE")

    results = {}

    # 1. Connection
    results["connection"] = check_firestore_connection()
    if not results["connection"]:
        print("\n[!] Aborting: no Firestore connection.")
        _print_summary(results)
        return

    # 2. Device lookup
    ok, email = check_device_lookup(target_device)
    results["device_lookup"] = ok

    # 3. Reward crediting
    results["reward_credit"] = check_reward_crediting(target_device, reward_mbs, email)

    # 4. Invalid ID rejection
    results["invalid_ids"] = check_invalid_device_ids()

    # 5. Cap enforcement
    # results["cap_enforcement"] = check_cap_enforcement(target_device)

    _print_summary(results)


def _print_summary(results: dict):
    _section("TEST SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, ok in results.items():
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}]  {name}")

    print()
    print(f"  Result: {passed}/{total} tests passed.")
    print("=" * 60)


# Entry Point

if __name__ == "__main__":
    if not credit_mbs_for_device:
        print("[FATAL] Could not load firebase_reward module. Exiting.")
        sys.exit(1)

    if len(sys.argv) > 1:
        # CLI mode: python test_firebase_reward.py <device_id> [mbs]
        dev_id = sys.argv[1]
        mbs_arg = float(sys.argv[2]) if len(sys.argv) > 2 else 100.0
        _section(f"CLI MODE - device={dev_id}  mbs={mbs_arg}")
        credit_mbs_for_device(dev_id, mbs_arg)
    else:
        # Default: full test suite with the known test device
        run_test_suite(
            target_device="7b06c36114aebc82",
            reward_mbs=250.0,
        )
