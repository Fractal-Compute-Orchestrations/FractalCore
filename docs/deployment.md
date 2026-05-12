# Deployment and Infrastructure Guide

FractalCore is designed to be highly portable, whether running on a local workstation for research or a cloud-based cluster for production orchestration.

## Containerized Deployment (Recommended)

The most reliable way to deploy FractalCore is using Docker. This ensures that all dependencies (TensorFlow, Flask, etc.) are consistently managed.

### 1. Production Stack
The production environment utilizes `gunicorn` as a high-performance WSGI server.

```bash
# Build and launch the stack
docker-compose up --build -d
```

### 2. Infrastructure Scaling
For large-scale deployments handling thousands of concurrent devices, the following infrastructure changes are recommended:

- **Reverse Proxy**: Use NGINX or Traefik to handle SSL termination and load balancing.
- **Persistent Storage**: Ensure that the `uploads/`, `downloads/`, and `global_model/` directories are mapped to persistent volumes (e.g., AWS EBS, Google Persistent Disk).
- **Compute Resources**: TensorFlow weight aggregation is CPU-intensive. Allocate at least 4 vCPUs and 8GB RAM for the aggregation service.

## Manual Installation

For environments where Docker is not available:

1. **Environment Setup**:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Database Connectivity**:
   Ensure the `firebase_service_account.json` is correctly placed and the path is exported in your environment.

3. **Execution**:
   ```bash
   gunicorn --bind 0.0.0.0:5000 src.fractal_server.server:app
   ```
