# Security Architecture

Security in FractalCore is not an afterthought; it is the foundation of the project's philosophy.

## Data Privacy

The core value proposition of FractalCore is **privacy-by-design**. 

- **Edge Isolation**: Raw training data (images/labels) never leaves the mobile device. The server only receives model weights (.ckpt).
- **Transient Checkpoints**: Uploaded checkpoints are stored only until aggregation is complete, after which they are purged to maintain a minimal data footprint.

## Access Control

### 1. Token-Based Authentication
FractalCore uses an `X-Auth-Token` system for both administrative and tenant access.
- Tokens are generated using cryptographically secure random strings.
- Sessions are isolated via `sessionStorage` in the browser, preventing cross-tenant leakage in multi-tab environments.

### 2. Multi-Tenant Isolation
Tenants are physically isolated at the filesystem level.
- Each tenant has a unique data silo (`tenants/{username}/`).
- Requests are validated against the authenticated tenant context to ensure no cross-access to data segments or models.

## Infrastructure Hardening

- **Environment Isolation**: Sensitive configuration (Firebase keys, admin credentials) is managed via environment variables and never committed to source control.
- **Request Validation**: The server validates hardware identifiers and task sequences to prevent spoofing or replay attacks during weight uploads.
