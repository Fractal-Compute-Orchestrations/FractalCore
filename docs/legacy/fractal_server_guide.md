# Legacy Server Operations and Migration Guide

This document provides operational procedures, migration instructions, and debugging workflows for standalone and legacy instances of the Fractal server.

---

## 1. Single-User vs Multi-Tenant Architecture

The Fractal server has transitioned from a single-user prototype to a production multi-tenant orchestration engine:

- **Legacy Engine (`src/legacy/single_user_server/`)**: Global file paths, monolithic configuration dictionary, single shared upload directory.
- **Production Engine (`src/fractal_server/`)**: Dynamic tenant routing, per-user data silos (`data/tenants/{username}/`), `X-Auth-Token` session isolation, and automated Firebase reward crediting.

---

## 2. Standalone Server Execution

To run the server in a standalone environment without Docker:

```bash
# 1. Activate Python virtual environment
source venv/bin/activate

# 2. Install production dependencies
pip install -r requirements.txt

# 3. Export environment configuration
export FLASK_APP=src/fractal_server/server.py
export FLASK_ENV=production
export PORT=5000

# 4. Launch the application
python -m flask run --host=0.0.0.0 --port=5000
```

---

## 3. Background Process Management

For standalone cloud virtual machines (e.g. Ubuntu instances on AWS, GCP, or Oracle Cloud):

```bash
# Start server in background with logging
nohup python src/fractal_server/server.py > server.log 2>&1 &

# Start the midnight reset scheduler
nohup python scripts/midnight_reset.py > reset.log 2>&1 &

# Monitor active process logs
tail -f server.log
```

---

## 4. Diagnostics and Health Checks

| Check | Command | Expected Output |
| :--- | :--- | :--- |
| **Verify Port Binding** | `lsof -i :5000` or `fuser 5000/tcp` | Process ID listening on TCP port 5000 |
| **Check Background Jobs** | `ps aux \| grep python` | Active PID for server.py and midnight_reset.py |
| **Verify Health API** | `curl -I http://localhost:5000/api/task/current` | HTTP 200 or 204 response |
| **Inspect System Logs** | `cat server.log \| grep ERROR` | Empty (or isolated non-fatal errors) |