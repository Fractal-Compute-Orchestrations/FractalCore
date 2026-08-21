# FractalCore Security Architecture Specification

Security and privacy are the primary design constraints of the Fractal ecosystem. This document outlines the cryptographic controls, network boundaries, and threat mitigation models implemented across FractalCore.

---

| Threat Category | System Mitigation Strategy |
| :--- | :--- |
| **Data Privacy Breach** | Zero raw dataset ingress; client data never leaves device |
| **Session Leakage** | Header-bound `X-Auth-Token` without cookie fallback |
| **Cross-Tenant Data Access** | Sandboxed directory trees (`data/tenants/{username}/`) |
| **Model Poisoning / Replay** | Single-use `task_Id` bound to dispatch records in memory |
| **Credential Exposure** | Service account isolation via environment variables |

---

## 1. Data Privacy Principles

- **Zero Ingress Policy**: The server never receives raw training samples (e.g. photos, voice, telemetry logs). Edge nodes train exclusively on local storage partitions and export mathematical weight deltas (`.ckpt`).
- **Transient Checkpoint Retention**: Client checkpoint files uploaded to `uploads/` are retained only until the Federated Averaging (`FedAvg`) threshold is met. Upon global model summation, raw client checkpoint files are permanently deleted.

---

## 2. Authentication & Session Security

### 2.1 Cryptographic Token Issuance
- Authentication uses cryptographically secure pseudorandom generators (`secrets.token_hex(32)`).
- Session tokens are valid strictly via the `X-Auth-Token` HTTP request header.
- **Absence of Cookie Fallback**: Browser storage uses `sessionStorage` rather than persistent cookies. This architecture guarantees that multiple tabs logged into different tenant accounts cannot bleed or overwrite session states.

### 2.2 Role-Based Access Control (RBAC)
- **Admin**: Full authority to create, inspect, and revoke tenant budgets, view server logs, and trigger maintenance scripts.
- **Tenant**: Scoped strictly to the tenant's own silo (`data/tenants/{username}/`). All file path resolutions are sanitized to prevent directory traversal (`../`) attacks.
- **Edge Node**: Access restricted to public task acquisition (`/api/task/current`) and single-use upload verification (`/api/model/upload`).

---

## 3. Storage and Process Hardening

- **Filesystem Sandboxing**: Physical separation of tenant directories prevents cross-tenant file read/write operations.
- **Single-Use Task Descriptors**: Every task issued to an edge node contains a cryptographic `task_Id`. Checkpoints uploaded with invalid, already-used, or unissued task IDs are rejected with HTTP 400.
- **Environment Isolation**: Private keys (`firebase_service_account.json`, `private.envs/`) are excluded from version control via `.gitignore` and `.dockerignore`.
