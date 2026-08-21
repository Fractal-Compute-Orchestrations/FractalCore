# FractalCore REST API Reference Specification

This document provides the complete API specification for the FractalCore server. The API supports two client classes:
1. **Administrative and Tenant Clients**: Web dashboard and programmatic management.
2. **Edge Node Clients (`FractalAndroid`)**: Task polling, asset downloads, and checkpoint uploads.

---

## Authentication and Headers

Administrative and Tenant routes require authentication via the `X-Auth-Token` header.

```http
X-Auth-Token: <64-character hex string>
```

- **Admin Scope**: Full access to global system state, tenant provisioning, and global logs.
- **Tenant Scope**: Isolated access scoped strictly to the authenticated tenant's data silo and session.
- **Client Scope**: Android node endpoints (`/api/task/current`, `/api/model/upload`, `/download/*`) are public or authenticated via unique hardware identifiers (`device_id`).

---

## Administrative and Tenant Endpoints

### 1. Tenant Authentication
Authenticates a tenant or administrator and generates a session token.

- **Method**: `POST`
- **Endpoint**: `/api/admin/login`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "username": "tenant_alice",
    "password": "secure_password"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "status": "success",
    "token": "7f8b9a1c2d3e4f5a6b7c8d9e0f1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a",
    "role": "tenant",
    "username": "tenant_alice"
  }
  ```
- **Error Responses**:
  - `401 Unauthorized`: Invalid credentials.

---

### 2. List All Tenants (Admin Only)
Retrieves the status and resource allocation of all registered tenants.

- **Method**: `GET`
- **Endpoint**: `/api/admin/tenants`
- **Headers**: `X-Auth-Token: <admin_token>`
- **Response (200 OK)**:
  ```json
  {
    "tenants": [
      {
        "username": "tenant_alice",
        "max_tflops": 10.0,
        "total_device_mbs": 500.0,
        "active_round": 3,
        "completed_tasks": 42
      }
    ]
  }
  ```

---

### 3. Provision New Tenant (Admin Only)
Provisions a new tenant with isolated filesystem storage and a compute budget.

- **Method**: `POST`
- **Endpoint**: `/api/admin/tenant`
- **Headers**:
  - `X-Auth-Token: <admin_token>`
  - `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "username": "tenant_bob",
    "password": "temporary_password",
    "max_tflops": 5.0,
    "total_device_mbs": 250.0
  }
  ```
- **Response (201 Created)**:
  ```json
  {
    "status": "created",
    "username": "tenant_bob",
    "silo_path": "data/tenants/tenant_bob/"
  }
  ```

---

## Edge Node Endpoints (`FractalAndroid`)

### 4. Fetch Current Task
Polled by Android nodes to receive an active compute assignment.

- **Method**: `GET`
- **Endpoint**: `/api/task/current`
- **Parameters**:
  - `device_id` (string, required): Hardware identifier of the requesting node.
- **Response (200 OK)**:
  ```json
  {
    "task_Id": "task_9876543210_abcdef",
    "taskType": "ActiveTask",
    "tenant": "tenant_alice",
    "model_url": "/download/model?filename=model_round_2.tflite",
    "image_bin_url": "/download/images?filename=batch_014.bin",
    "label_bin_url": "/download/labels?filename=batch_014_labels.bin",
    "hyperparameters": {
      "batch_size": 32,
      "epochs": 5,
      "learning_rate": 0.001
    },
    "reward_rate_mb": 5.0,
    "task_expire_date": "2026-08-29"
  }
  ```
- **Response (204 No Content)**: No active tasks available in the queue.

---

### 5. Upload Model Checkpoint
Transmits locally trained weight deltas back to the orchestrator.

- **Method**: `POST`
- **Endpoint**: `/api/model/upload`
- **Headers**: `Content-Type: multipart/form-data`
- **Form Data Fields**:
  - `task_Id` (string, required): Assigned task ID received during task fetch.
  - `device_id` (string, required): Hardware ID of the client device.
  - `model_file` (file binary, required): Checkpoint weights (`.ckpt` format).
- **Response (200 OK)**:
  ```json
  {
    "status": "success",
    "message": "Checkpoint received and registered.",
    "reward_credited_mb": 5.0
  }
  ```
- **Error Responses**:
  - `400 Bad Request`: Missing form field or invalid `task_Id`.
  - `409 Conflict`: Task ID has already been consumed or expired.

---

## Binary Asset Download Endpoints

### 6. Download Dataset Image Bin
- **Method**: `GET`
- **Endpoint**: `/download/images`
- **Query Parameters**: `filename=<string>`
- **Response**: Binary byte stream (`application/octet-stream`).

### 7. Download Dataset Label Bin
- **Method**: `GET`
- **Endpoint**: `/download/labels`
- **Query Parameters**: `filename=<string>`
- **Response**: Binary byte stream (`application/octet-stream`).

### 8. Download Global Model Checkpoint
- **Method**: `GET`
- **Endpoint**: `/download/model`
- **Query Parameters**: `filename=<string>`
- **Response**: Binary model file (`application/octet-stream`, `.tflite` or `.pte`).
