
# Fractal Server Deployment Guide

## 🟢 Upload Files to Server
Run this on your **Windows machine** to sync your local project folder with the Oracle instance.

```bash
scp -i "P:\Fractal\Fractal Server\ssh-key-2026-03-06.key" -r "P:\Fractal\Fractal Server" ubuntu@161.118.184.57:~/
```

---

## 1. Start the System (Run on Ubuntu)
Connect to your instance and fire up both the **Flask API** and the **Midnight Reset** worker.

```bash
# 1. Connect via SSH
ssh -i "P:\Fractal\Fractal Server\ssh-key-2026-03-06.key" ubuntu@161.118.184.57

# 2. Enter Project & Activate Env
cd "Fractal Server"
source ~/fractal_env/bin/activate

# 3. Start the Global Midnight Sweeper (Background)
nohup python midnight_reset.py > reset_log.txt 2>&1 &

# 4. Start the Fractal API Server (Background)
nohup python server.py > server_log.txt 2>&1 &

# 5. Monitor the Logs
tail -f server_log.txt
```

---

## 2. Stop / Kill the System (Run on Ubuntu)
Use these commands to gracefully shut down the components.

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
Useful commands to ensure everything is running perfectly in the background.

| Goal                          | Command              |
| :---------------------------- | :------------------- |
| **Check if both are running** | `ps aux \| grep .py` |
| **View Reset History**        | `cat reset_log.txt`  |
| **Check Server Timezone**     | `timedatectl`        |
| **Clear all stale logs**      | `rm *_log.txt`       |
