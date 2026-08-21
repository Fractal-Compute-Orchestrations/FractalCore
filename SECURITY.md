# Security Policy

## Supported Versions

The following table summarizes the support status for FractalCore releases:

| Version | Supported |
| :--- | :--- |
| v1.0.x | Yes |
| < v1.0 | No |

---

## Reporting a Vulnerability

Security is a fundamental priority for FractalCore. If you discover a potential security vulnerability, please follow the coordinated disclosure process:

1. **Do not disclose the issue publicly** via GitHub Issues, Discussions, or pull requests.
2. Send a confidential report to `security@fractalcompute.io` with full technical details.
3. Include the following in your disclosure:
   - Vulnerability class (e.g. authentication bypass, privilege escalation, path traversal).
   - Exact affected file paths, endpoints, and line numbers.
   - Step-by-step reproduction instructions and proof-of-concept payload if available.
   - Assessed impact on multi-tenant isolation or node telemetry security.

### Response Timeline
- **Initial Acknowledgment**: Within 48 hours of receipt.
- **Triage and Impact Assessment**: Within 7 business days.
- **Patch Release & Security Advisory**: Coordinated upon validation and regression testing.
