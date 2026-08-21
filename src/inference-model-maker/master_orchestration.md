# Fractal Master Orchestration Pipeline Specification

**Architecture Philosophy**: Contract-Driven Manufacturing Pipeline  
**Target Execution**: Distributed Foundation Model Inference across Android Mesh  

---

## 1. System Overview

The Fractal orchestration system operates as an automated distributed manufacturing pipeline:

- **Raw Material**: Monolithic Foundation Model checkpoint (FP16/BF16).
- **Workstations (Modules)**: Isolated environments (Python server-side compiler, Kotlin on-device runtime) that perform singular deterministic transformations.
- **Conveyor Belt (Contracts)**: Strict input/output data shapes and memory constraints.
- **Quality Assurance (Validators)**: Programmatic validation assertions executed at stage boundaries. If a validator fails, compilation halts and an exception is raised.

---

## 2. Global System Contracts

### 2.1 The Weight Contract (Static Storage Asset)
- **Definition**: The serialized binary representation of a single neural network layer.
- **Format**: FlatBuffer binary (`.pte` or `.tflite`).
- **Size Constraint**: Strictly $\le 150\text{MB}$.

### 2.2 The Activation Contract (Dynamic Network Payload)
- **Definition**: The hidden state activation tensor moving between Android mesh nodes.
- **Format**: 1-dimensional byte array representing 4,096 elements + 1 Float32 scale.
- **Size Constraint**: Strictly 4,100 bytes.

---

## 3. Pipeline Module Specifications

### Module 1: The Slicer (Server-Side Graph Surgery)
- **Role**: Ingests monolithic models, splits them into atomic layer chunks, applies INT4 weight quantization, and packages them for mobile execution.
- **Input Contract**: Hugging Face FP16/BF16 Model Checkpoint.
- **Output Contract**: Sequential `layer_[N].pte` binary files.
- **Validator (`SlicerQA`)**:
  - Validates all output files match extension `.pte`.
  - Asserts every file satisfies $\text{size} \le 150\text{MB}$.

### Module 2: The Loader (On-Device Memory Mapping)
- **Role**: Running on the Android client, ingests binary files and maps them to virtual memory without triggering the Android Low Memory Killer (LMK).
- **Input Contract**: `layer_[N].pte` file residing on local Android filesystem.
- **Output Contract**: Initialized ExecuTorch module mapped into process virtual memory via `mmap`.
- **Validator (`LoaderQA`)**:
  - Samples `Debug.MemoryInfo()`.
  - Asserts physical Resident Set Size (RSS) spike is $\le 50\text{MB}$.

### Module 3: The Engine (On-Device Inference Execution)
- **Role**: Receives an incoming network activation payload, dequantizes it, executes the XNNPACK forward pass, quantizes the output tensor, and dispatches it to the next node in the mesh.
- **Input Contract**: Byte array [4,096 elements] + Float32 scale.
- **Output Contract**: Byte array [4,096 elements] + Float32 scale.
- **Validator (`EngineQA`)**:
  - Asserts input and output buffer lengths equal 4,096.
  - Asserts output values do not contain NaN or Infinity.

---

## 4. Engineering Directives for Implementation

1. **Subsystem Isolation**: Do not couple compilation logic with on-device runtime logic. The Server Slicer (Python) and Android Client (Kotlin) communicate strictly via file and network contracts.
2. **Validator-First Implementation**: When writing a pipeline workstation or execution engine, write the validator assertion test before implementing transformation logic.
3. **Deterministic Contracts**: Every workstation must expose a `verify()` method validating preconditions and postconditions.
