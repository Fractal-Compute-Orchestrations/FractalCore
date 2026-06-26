# FRACTAL MASTER ORCHESTRATION PIPELINE

**Architecture Philosophy:** Contract-Driven Manufacturing Pipeline
**Target Execution:** Llama 3 (8B) across Android Master-Worker Mesh

## 1. The Factory Analogy (System Overview)

Project Fractal operates as an automated manufacturing pipeline.

- **The Raw Material:** A monolithic 16GB LLM.
- **The Workstations (Modules):** Isolated software environments (Python off-device, Kotlin on-device) that perform exactly one transformation.
- **The Conveyor Belt (Contracts):** Strict input/output data shapes. A module knows absolutely nothing about the internal logic of the module before it. It only knows what data shape it requires to start working.
- **Quality Assurance (Validators):** Automated programmatic checks that run immediately after a module completes its task. If a validator fails, the conveyor belt stops, and an exception is thrown.

---

## 2. Global System Contracts

### 2.1 The Weight Contract (Static Asset)

- **Definition:** The physical representation of a single neural network layer.
- **Shape Constraint:** A binary file (e.g., `.pte` or `.tflite`).
- **Size Constraint:** Strictly ≤ 150MB.

### 2.2 The Activation Contract (Dynamic Payload)

- **Definition:** The hidden state tensor moving between Android devices.
- **Shape Constraint:** A 1-dimensional byte array representing 4,096 elements + 1 Float (Scale).
- **Size Constraint:** Strictly 4,100 bytes.

---

## 3. The Pipeline Stages (Module Specifications)

### MODULE 1: The Slicer (Off-Device Graph Surgery)

- **Role:** Acts as the intake factory. Ingests the monolithic 16GB model, chops it into 32 atomic layer chunks, applies INT4 weight quantization, and packages them for mobile execution.
- **Input Contract:** Hugging Face `LlamaForCausalLM` FP16 Checkpoint.
- **Output Contract:** 32 individual `layer_[N].pte` binary files.
- **The Validator (`SlicerQA`):**
  - Iterates through all 32 output files.
  - Asserts `file.extension == ".pte"`.
  - Asserts `file.size <= 150MB`.
  - _Halt Condition:_ If any file exceeds 150MB, the export is rejected.

### MODULE 2: The Loader (On-Device Memory Mapping)

- **Role:** Acts as the receiving dock on the Android phone. Ingests the 150MB binary file and maps it to virtual memory without triggering the Android Low Memory Killer (LMK).
- **Input Contract:** `layer_[N].pte` file residing on local Android storage.
- **Output Contract:** An initialized `ExecuTorch.Module` mapped to virtual memory.
- **The Validator (`LoaderQA`):**
  - Samples Android OS `Debug.MemoryInfo()`.
  - Asserts `Resident Set Size (RSS) spike <= 50MB`.
  - _Halt Condition:_ If physical RAM spikes indicating a heap allocation instead of a virtual map, the app safely terminates the process to prevent an OS-level crash.

### MODULE 3: The Engine (On-Device Execution)

- **Role:** Acts as the computational press. Ingests an incoming network payload, dequantizes it, executes the XNNPACK forward pass, quantizes the result, and prepares it for the network socket.
- **Input Contract:** `ByteArray` [4096 elements] + `Float` [1 element].
- **Output Contract:** `ByteArray` [4096 elements] + `Float` [1 element].
- **The Validator (`EngineQA`):**
  - Asserts `InputArray.size == 4096`.
  - Asserts `OutputArray.size == 4096`.
  - Asserts `OutputArray` does not contain `NaN` or `Infinity` values.
  - _Halt Condition:_ If array sizes mismatch or math corruption occurs, the pipeline requests a retry from the Master Server instead of passing corrupted data to the next phone.

---

## 4. Agent Directives (Instructions for AI Implementation)

If you are an AI agent tasked with writing code for this repository, you must adhere to the following rules:

1. **Isolation:** Do not merge Modules. Module 2 (Kotlin Loader) must not contain the logic for Module 3 (Kotlin Engine). They must be separate classes communicating via interfaces.
2. **Validator-First Coding:** When writing a module, write the Validator (the unit test / runtime check) _before_ writing the execution logic.
3. **No Black Boxes:** Every module must implement a `verify()` method that executes its respective QA Validator before passing its Output Contract to the next stage.
