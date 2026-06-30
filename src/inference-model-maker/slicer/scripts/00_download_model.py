"""
slicer/scripts/00_download_model.py
===================================
Asset Provisioner for Project Fractal.

This script runs exactly once to securely authenticate and download the monolithic
16GB Meta Llama 3 (8B) model checkpoint from Hugging Face to local storage.
It ensures robust download resumption and enforces strict weight format selection
to avoid downloading redundant format weights.

Expected Environment:
--------------------
HF_TOKEN: Must be set in environment variables or a .env file.
"""

from __future__ import annotations

import argparse
import os
import sys
import socket

# Prevent infinite hangs on unstable connections by timing out sockets after 30 seconds.
# This triggers huggingface_hub's native auto-retry mechanism.
socket.setdefaulttimeout(30.0)

# Force stdout and stderr to flush immediately so that progress bars update in real-time.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Try loading env variables from .env file if dotenv is installed.
# Resolve the .env path relative to the project root (two levels up from slicer/scripts/).
try:
    from pathlib import Path
    from dotenv import load_dotenv
    _PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # inference-model-maker/
    _ENV_FILE = _PROJECT_ROOT / ".env"
    if _ENV_FILE.exists():
        load_dotenv(_ENV_FILE)
except ImportError:
    pass

# Set Hugging Face Hub timeouts before importing huggingface_hub to prevent infinite hangs.
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "30")
os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "30")

try:
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError
except ImportError:
    print("[ERROR] The 'huggingface_hub' library is required to run this script.")
    print("Please install it by running: pip install huggingface_hub")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
DEFAULT_MODEL_ID: str = "meta-llama/Meta-Llama-3-8B"
DEFAULT_TARGET_DIR: str = os.getenv("RAW_MODEL_DIR", "./assets/raw_models/Meta-Llama-3-8B")

# Ignore patterns to prevent downloading duplicate or redundant formats.
# We explicitly target Safetensors (.safetensors) and ignore PyTorch .bin,
# GGUF, or other binary formats to save bandwidth and disk space.
IGNORE_PATTERNS: list[str] = [
    "*.bin",
    "*.gguf",
    "*.h5",
    "*.ot",
    "*.msgpack",
    "*.pth",
    "original/*",  # Skip native llama weights if downloading standard HF format
]


def resolve_auth_token() -> str:
    """Resolve the Hugging Face authentication token from env or CLI login.

    Returns
    -------
    str
        The resolved token string.

    Raises
    ------
    RuntimeError
        If no token can be resolved.
    """
    # 1. Check local environment variable
    token: str | None = os.getenv("HF_TOKEN")
    if token:
        return token

    # 2. Check token from huggingface-cli caching
    try:
        from huggingface_hub import HfFolder
        cached_token = HfFolder.get_token()
        if cached_token:
            return cached_token
    except Exception:
        pass

    # 3. Raise helpful diagnostic error
    raise RuntimeError(
        "Hugging Face Authentication Token (HF_TOKEN) not found.\n\n"
        "Llama 3 is a gated model. To resolve this error:\n"
        "  1. Request access on Hugging Face: https://huggingface.co/meta-llama/Meta-Llama-3-8B\n"
        "  2. Obtain a User Access Token: https://huggingface.co/settings/tokens\n"
        "  3. Set it in your environment: e.g. export HF_TOKEN=your_token_here (Linux/macOS) "
        "or $env:HF_TOKEN=\"your_token_here\" (Windows PowerShell)\n"
        "     Alternatively, create a '.env' file in the root workspace containing: HF_TOKEN=your_token_here\n"
    )


def download_model(model_id: str, output_path: str, max_workers: int = 4) -> None:
    """Securely downloads the Hugging Face model to the target directory.

    Parameters
    ----------
    model_id : str
        The Hugging Face Repository ID.
    output_path : str
        Target local directory where files will be stored.
    max_workers : int
        Maximum number of parallel downloader threads.
    """
    token = resolve_auth_token()
    abs_output_path = os.path.abspath(output_path)

    print("[PROVISIONER] -- Initializing Secure Monolithic Asset Provisioning --")
    print(f"[PROVISIONER] Repository ID:   {model_id}")
    print(f"[PROVISIONER] Local target:    {abs_output_path}")
    print(f"[PROVISIONER] Dtype Focus:     Safetensors (.safetensors)")
    print(f"[PROVISIONER] Ignoring:        {', '.join(IGNORE_PATTERNS)}")
    print("[PROVISIONER] ------------------------------------------------------------\n")

    os.makedirs(abs_output_path, exist_ok=True)

    try:
        # snapshot_download handles resumes and multi-part downloads natively
        downloaded_path = snapshot_download(
            repo_id=model_id,
            local_dir=abs_output_path,
            ignore_patterns=IGNORE_PATTERNS,
            token=token,
            max_workers=max_workers,  # Parallel download streams
        )

        print("\n[PROVISIONER] -- Provisioning Completed Successfully --")
        print(f"[PROVISIONER] [OK] Monolithic asset cached at absolute path:")
        print(f"              {downloaded_path}")

    except GatedRepoError:
        print(
            f"\n[PROVISIONER] [ERROR] Gated repository access rejected for '{model_id}'.\n"
            f"Ensure you have approved the license agreement on HF and your HF_TOKEN is valid.",
            file=sys.stderr,
        )
        sys.exit(1)
    except RepositoryNotFoundError:
        print(
            f"\n[PROVISIONER] [ERROR] Repository '{model_id}' does not exist on HF.",
            file=sys.stderr,
        )
        sys.exit(1)
    except Exception as exc:
        print(
            f"\n[PROVISIONER] [ERROR] Download process failed: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Asset Provisioner: Download Llama 3 8B")
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"HF Repository ID (default: {DEFAULT_MODEL_ID})",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_TARGET_DIR,
        help=f"Target output directory (default: {DEFAULT_TARGET_DIR})",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="Maximum parallel download threads (use 1 for very unstable connections) (default: 2)",
    )

    args = parser.parse_args()
    try:
        download_model(model_id=args.model, output_path=args.output, max_workers=args.workers)
    except KeyboardInterrupt:
        print("\n\n[PROVISIONER] [INFO] Download interrupted by user. Run the script again to resume.")
        sys.exit(0)
    except RuntimeError as rerr:
        print(f"\n[PROVISIONER] [ERROR] {rerr}", file=sys.stderr)
        sys.exit(1)
