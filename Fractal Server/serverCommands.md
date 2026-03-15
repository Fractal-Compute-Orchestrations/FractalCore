# Fractal Server Deployment Guide

## Upload Files to Server

Run this on your Windows machine to upload the server folder to the remote machine.

```bash
scp -i "P:\\Fractal\\Fractal Server\\ssh-key-2026-03-06.key" -r "P:\\Fractal\\Fractal Server" ubuntu@161.118.184.57:~/
```

---

## 1. Log Into the Server (Run on Windows)

Open Command Prompt or PowerShell and connect to the remote machine.

```bash
ssh -i "P:\\Fractal\\Fractal Server\\ssh-key-2026-03-06.key" ubuntu@161.118.184.57

cd "Fractal Server"

source ~/fractal_env/bin/activate

nohup python server.py > server_log.txt 2>&1 &

tail -f server_log.txt
```


## 2. Stop / Kill the Server (Run on Ubuntu)

```bash
fuser -k 5000/tcp
```

This stops any process using port 5000.