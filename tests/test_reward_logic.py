import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SERVER_DIR = os.path.join(PROJECT_ROOT, "src", "fractal_server")
if SERVER_DIR not in sys.path:
    sys.path.insert(0, SERVER_DIR)

from firebase_reward import credit_mbs_for_device, _get_email_by_device_id  # noqa: E402


def test_invalid_device_id_rejection():
    # Empty, whitespace, None, or 'unknown' must return False
    assert credit_mbs_for_device("", 10.0) is False
    assert credit_mbs_for_device("   ", 10.0) is False
    assert credit_mbs_for_device("unknown", 10.0) is False
    assert credit_mbs_for_device("UNKNOWN", 10.0) is False
    assert credit_mbs_for_device(None, 10.0) is False  # type: ignore[arg-type]


def test_lookup_uninitialized_db():
    # When _db is None or invalid ID, lookup gracefully returns None without crashing
    email = _get_email_by_device_id("non_existent_test_device_12345")
    assert email is None
