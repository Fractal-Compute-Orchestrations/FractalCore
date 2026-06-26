# FRACTAL: ARCHITECTURAL MOTIVE & SYSTEM MANIFESTO

## 1. The Core Problem

The current trajectory of Generative AI is physically unsustainable for the edge. State-of-the-art Large Language Models (LLMs) like Llama 3 (8B) require massive VRAM pools, restricting their execution to expensive cloud clusters or high-end desktop hardware.

Standard mobile hardware cannot load these models without triggering the OS-level Low Memory Killer (LMK), resulting in immediate application crashes. Consequently, mobile AI is currently trapped in a centralized paradigm: reliant on cloud APIs, sacrificing user privacy, enduring internet-bound latency, and incurring continuous server costs.

## 2. The Fractal Thesis

**Compute is abundant, but highly fragmented.** The average local environment (a household, a classroom, an office) contains gigabytes of dormant mobile RAM and teraflops of unused computational power distributed across various smartphones.

**Project Fractal** exists to harvest this fragmented compute. By shattering a massive neural network into its atomic structural layers and chaining devices together over a local network, we bypass the physical memory wall of any single device. Fractal transforms a collection of mid-range Android phones into a unified, decentralized intelligence cluster capable of running state-of-the-art foundation models locally, privately, and sustainably.

## 3. The Architectural Solution: Master-Worker Pipeline Parallelism

Fractal discards complex, unstable Peer-to-Peer (P2P) consensus models in favor of a robust, deterministic **Master-Worker Architecture**.

- **The Control Plane (The Master):** A lightweight central orchestrator maintains a real-time topology snapshot of available Android devices. It maps the devices and sequentially assigns computational roles (e.g., "Device A hosts Layers 0-3").
- **The Data Plane (The Workers):** Android edge devices execute the mathematical payload. Each device loads its assigned chunk and passes the resulting intermediate mathematical tensor directly to the next device in the chain via local network sockets.

## 4. Ironclad Engineering Constraints

To make this computationally and physically viable, Fractal is governed by three uncompromising constraints:

1. **The Memory Constraint (LMK Immunity):** \* Android Dalvik/ART garbage-collected heaps cannot handle gigabyte-scale weights.
   - **The Solution:** All model partitions (e.g., ~150MB `.pte` or `.tflite` chunks) MUST be loaded using zero-copy memory mapping (`mmap`). Weights remain in the virtual address space and are paged into physical RAM by the Linux kernel only when actively computing, guaranteeing immunity from the Android Low Memory Killer.
2. **The Network Payload Constraint (Bandwidth Survival):**
   - Transmitting gigabytes of activations over Wi-Fi will stall the pipeline and destroy the inference token generation rate.
   - **The Solution:** The activation tensors (hidden states) passed between devices MUST be aggressively quantized to INT8 before transmission. A 4096-dimension tensor must never exceed a strict ~4.1KB payload threshold on the wire.
3. **The Contract-Driven Constraint (Modularity):**
   - The pipeline is a factory. A module must not care how the previous module operates; it must only care that the input data matches the exact expected mathematical shape.
   - **The Solution:** Every phase of the architecture (Slicing, Loading, Executing) must be separated by strict Input/Output data contracts and automated validation assertions.

## 5. The Ultimate Objective

Fractal is not an experiment in routing protocols; it is an applied execution engine. The objective is to achieve stable, fluid, and interactive decentralized inference of Llama 3 (8B) across an ad-hoc mesh of Android smartphones, proving that true decentralized edge intelligence is a physical reality.
