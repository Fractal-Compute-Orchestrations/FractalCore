# FractalCore API Reference

The FractalCore server exposes a RESTful API for both Administrative management and Client (Mobile) interaction.

## Authentication

### Administrative / Tenant Auth
Administrative routes require the `X-Auth-Token` header.
- **Admin**: Full access to all tenants and system metadata.
- **Tenant**: Access limited to their specific session data.

---

## Administrative API

### `GET /api/admin/tenants`
Returns a list of all registered tenants and their current status.
**Auth**: Admin Required

### `POST /api/admin/tenant`
Creates a new tenant.
**Payload**:
```json
{
  "username": "new_user",
  "password": "secure_password",
  "max_tflops": 1.0,
  "total_device_mbs": 150.0
}
```
**Auth**: Admin Required

---

## Client API (Android)

### `GET /api/task/current`
Requests a training task from the server.
**Parameters**:
- `device_id`: Unique hardware identifier for the mobile device.

**Response**:
Returns a JSON task object containing model paths, data bin names, and training hyperparameters.

### `POST /api/model/upload`
Uploads a trained checkpoint.
**Form Data**:
- `task_Id`: The ID of the assigned task.
- `device_id`: Hardware ID of the client.
- `model_file`: Binary checkpoint file (.ckpt).

---

## Download Routes

- `GET /download/images?filename=...`: Fetch a binary image bin.
- `GET /download/labels?filename=...`: Fetch a binary label bin.
- `GET /download/model?filename=...`: Fetch the latest TFLite model.
