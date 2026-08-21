# FractalCore Deployment and Operations Guide

This guide describes production deployment, scaling, and maintenance procedures for the FractalCore orchestration server.

---

## Containerized Deployment (Production Standard)

Docker Compose provides the standard production deployment stack with automated process management and storage volume isolation.

### 1. Prerequisites
- Docker Engine 24.0+ and Docker Compose v2+
- Linux host with at least 4 vCPUs and 8GB RAM (recommended for FedAvg tensor operations)
- Valid `firebase_service_account.json` credential file

### 2. Configuration Setup
Create and configure your `.env` file in the `FractalCore` root directory:

```bash
# Core Server Configuration
PORT=5000
FLASK_ENV=production
SECRET_KEY=generate_a_64_character_hex_secret_here

# Storage Mounts
DATA_DIR=/var/lib/fractal/data
UPLOAD_DIR=/var/lib/fractal/uploads

# Firebase Service Account
FIREBASE_CREDENTIALS_PATH=/etc/fractal/firebase_service_account.json
```

### 3. Launching the Stack

```bash
# Build images and start services in background
docker-compose up --build -d

# Verify container health
docker-compose ps

# Stream application logs
docker-compose logs -f app
```

---

## High-Throughput Scaling Configuration

For enterprise deployments handling thousands of concurrent Android nodes:

### 1. Gunicorn Multi-Worker Configuration
Configure Gunicorn WSGI with synchronous/asynchronous workers matching host CPU core allocation:

```bash
gunicorn \
  --workers 4 \
  --threads 2 \
  --bind 0.0.0.0:5000 \
  --worker-class gthread \
  --timeout 120 \
  --access-logfile - \
  --error-logfile - \
  src.fractal_server.server:app
```

### 2. NGINX Reverse Proxy and SSL Termination
Deploy NGINX in front of Gunicorn to manage TLS encryption, gzip compression, and client body buffers for large `.ckpt` uploads:

```nginx
server {
    listen 443 ssl http2;
    server_name core.fractalcompute.io;

    ssl_certificate /etc/letsencrypt/live/core.fractalcompute.io/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/core.fractalcompute.io/privkey.pem;

    client_max_body_size 250M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Automated Maintenance Operations

### 1. Midnight Sweeper (`scripts/midnight_reset.py`)
Cleans expired task reservations, resets daily client rate limit buckets, and synchronizes state with Firestore:

```bash
# Run standalone maintenance sweep
python scripts/midnight_reset.py
```

### 2. Global Model Accuracy Tester (`scripts/Global_Model_Tester.py`)
Validates global model convergence and classification accuracy across validation splits:

```bash
python scripts/Global_Model_Tester.py --model_path data/global_model/model.tflite --dataset_dir data/validation/
```
