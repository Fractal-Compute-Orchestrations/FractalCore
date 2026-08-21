# FractalCore Technical Roadmap

This roadmap outlines scheduled engineering milestones and capability enhancements for FractalCore.

---

## Q3 2026: Performance and Aggregation Reliability

- [ ] **Dynamic Quorum Adjustment**: Automatically adapt client upload thresholds ($N$) based on active network latency and device drop-out rates.
- [ ] **Asynchronous FedAvg Support**: Allow continuous streaming weight updates with decay weighting for straggler nodes.
- [ ] **Database Abstraction Layer**: Support pluggable backends (PostgreSQL with asyncpg, MongoDB, Redis) alongside Firestore.
- [ ] **Automated Benchmark Harness**: Automated regression testing measuring FedAvg tensor aggregation latency across large parameter matrices.

---

## Q4 2026: Privacy Engineering and Mesh Slicing

- [ ] **Differential Privacy (DP)**: Introduce adaptive gradient clipping and Gaussian noise injection into the aggregation pipeline.
- [ ] **Secure Multi-Party Computation (SMPC)**: Experimental support for additive secret sharing to prevent plaintext checkpoint inspection.
- [ ] **Automated Mesh Slicer**: Dynamic layer splitting based on real-time device VRAM announcements from connected Android nodes.
- [ ] **OpenID Connect (OIDC)**: Enterprise tenant authentication integration with Okta, Keycloak, and AWS IAM.

---

## 2027: Distributed Scaling & Edge Framework Support

- [ ] **Kubernetes Operator**: Native CRD-based deployment of FractalCore clusters with auto-scaling WSGI pods.
- [ ] **Real-Time Telemetry Dashboard 2.0**: WebSocket-driven visualization of global model loss curves, node geographic distribution, and thermal profiles.
- [ ] **Multi-Framework Edge Export**: Expand Slicer export targets to include CoreML (Apple Silicon), PyTorch Mobile, and ONNX Runtime Mobile.
