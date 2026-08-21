# Fractal Server Operations and Command Reference

This document lists operational commands for deploying, monitoring, and managing standalone Fractal server instances on remote Linux infrastructure.

---

## 1. Remote Synchronization

Synchronize project files from a local development machine to the remote Linux instance:

```bash
scp -i "/path/to/ssh_key.key" -r "./FractalCore" ubuntu@161.118.184.57:~/
```

---

## 2. Process Control

### 2.1 Start Services (Ubuntu / Linux Host)

Connect to the instance and launch the API server and midnight maintenance sweeper:

```bash
# 1. Connect via SSH
ssh -i "/path/to/ssh_key.key" ubuntu@161.118.184.57

# 2. Navigate to project directory and activate environment
cd FractalCore
source ~/fractal_env/bin/activate

# 3. Start the Midnight Sweeper worker in background
nohup python scripts/midnight_reset.py > reset_log.txt 2>&1 &

# 4. Start the FractalCore API Server in background
nohup python src/fractal_server/server.py > server_log.txt 2>&1 &

# 5. Stream active server logs
tail -f server_log.txt
```

### 2.2 Stop Services

Gracefully terminate background services:

```bash
# Stop API Server (Port 5000)
fuser -k 5000/tcp

# Stop Midnight Reset Script
pkill -f midnight_reset.py
```

---

## 3. Maintenance and Diagnostics

| Objective | Command |
| :--- | :--- |
| **Check Active Python Services** | `ps aux \| grep .py` |
| **Inspect Reset Log History** | `cat reset_log.txt` |
| **Check System Timezone** | `timedatectl` |
| **Clear Stale Runtime Logs** | `rm -f *_log.txt` |
| **Inspect Open Ports** | `netstat -tuln \| grep 5000` |
