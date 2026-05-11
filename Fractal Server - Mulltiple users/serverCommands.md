# Fractal Server Deployment Guide (`Fractal Server - exp`)

This guide outlines the process for syncing, starting, and maintaining the **Fractal Federated Learning Server** on the Oracle Ubuntu instance.

---

## 🟢 Upload Files to Server

Run this command from your **local Windows machine** (PowerShell or CMD) to sync the project folder.

```bash
scp -i "P:\Fractal\Fractal Server - exp\ssh-key-2026-03-06.key" -r "P:\Fractal\Fractal Server - exp" ubuntu@161.118.184.57:~/
```

---

## 1. Start the System (Ubuntu)

Connect to the instance and initialize the background workers.

```bash
# 1. Connect via SSH
ssh -i "P:\Fractal\Fractal Server - exp\ssh-key-2026-03-06.key" ubuntu@161.118.184.57

# 2. Enter Project Directory & Activate Environment
cd "Fractal Server - exp"
source ~/fractal_env/bin/activate

# 3. Start the Global Midnight Sweeper (Background)
# This handles daily TFLOPs and state resets
nohup python midnight_reset.py > reset_log.txt 2>&1 &

# 4. Start the Fractal API Server (Background)
# TF logs and Flask output will go to server-exp_log.txt
nohup python server.py > server-exp_log.txt 2>&1 &

# 5. Monitor the Live Logs
tail -f server-exp_log.txt
```

---

## 2. Stop / Kill the System (Ubuntu)

Use these commands to shut down the processes safely.

### Stop the API Server (Port 5000)
```bash
fuser -k 5000/tcp
```

### Stop the Midnight Reset Script
```bash
pkill -f midnight_reset.py
```

---

## 3. Maintenance & Health Checks

| Goal                                   | Command              |
| :------------------------------------- | :------------------- |
| **Check if both processes are active** | `ps aux \| grep .py` |
| **View Reset/Cleanup History**         | `cat reset_log.txt`  |
| **Check Current Server Time**          | `timedatectl`        |
| **Wipe all log files**                 | `rm *_log.txt`       |

---

## ⚠️ Deployment Notes
* **Folder Name**: Ensure the folder name `"Fractal Server - exp"` is always quoted in commands due to the spaces.
* **Firestore**: Ensure the `firebase_service_account.json` is present in the root of the folder before syncing.
* **Port 5000**: Ensure the Oracle Cloud Ingress Rules allow TCP traffic on port 5000.
```
