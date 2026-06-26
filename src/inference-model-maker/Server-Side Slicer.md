# MODULE 1: THE SLICER (SERVER-SIDE ORCHESTRATION)

**Architectural Role:** The Offline Compiler & Graph Surgeon
**Target Environment:** Host Server/Desktop (Python Environment)
**Target Asset:** Llama 3 (8B) Monolithic Checkpoint
**Design Pattern:** Strict Contract-Driven Pipeline

## 1. Module Philosophy

This module operates entirely off-device. Its sole purpose is to ingest a massive, un-executable 16GB PyTorch model and surgically compile it into 32 independent, hardware-optimized, memory-safe binary payloads (`.pte` files).

The Android clients must never perform graph compilation, quantization, or graph tracing. The Slicer absorbs 100% of the preprocessing complexity to ensure the Android edge nodes act as purely "dumb," hyper-fast execution runtimes.

---

## 2. Step-by-Step Contract Pipeline

### Step 1.1: Monolithic Ingestion & Structural Isolation

- **Objective:** Mount the Llama 3 (8B) model into host memory safely and sever a single transformer decoder block from the sequential stack without breaking its parameter references.
- **Input Contract:**
  - **Asset:** Hugging Face model identifier or local directory path.
  - **Data Type:** FP16 or BF16 weights.
- **Output Contract:**
  - **Asset:** A single, isolated `torch.nn.Module` representing exactly one layer (e.g., `model.layers[5]`).
- **Architectural Directive:** The ingestor must map the model to the host CPU (`device_map="cpu"`) to prevent VRAM exhaustion. The module must retain its layer normalization and self-attention topological boundaries.
- **Validator (`IngestQA`):**
  - Assert the isolated asset is a subclass of `torch.nn.Module`.
  - Calculate total parameter count of the isolated block. Assert it is mathematically $\approx 1/32$ of the 8-billion total.

### Step 1.2: Weight-Only Quantization (INT4)

- **Objective:** Compress the FP16 weight matrices of the isolated layer to hit the strict mobile RAM safety boundaries while configuring the activations to dynamically quantize to INT8 at runtime.
- **Input Contract:**
  - **Asset:** Unquantized `torch.nn.Module` (Size: $\approx 450$MB).
- **Output Contract:**
  - **Asset:** Mutated `torch.nn.Module` with INT4 weights (Target Size: $\approx 115$MB).
- **Architectural Directive:** Apply Grouped INT4 Quantization (group size 128). This must specifically target the `q_proj`, `k_proj`, `v_proj`, and `o_proj` linear layers of the attention mechanism, alongside the MLP projection layers.
- **Validator (`QuantQA`):**
  - Traverse the module's state dictionary.
  - Assert that the byte size of the target linear layers reflects a 4-bit footprint (e.g., `param.nelement() / 2` bytes).
  - Assert the total in-memory footprint of the layer is strictly $\le 150$MB.

### Step 1.3: Static Graph Tracing (ATen Dialect Export)

- **Objective:** Eliminate the Python runtime environment entirely. Freeze the dynamic PyTorch code into a static, statically-shaped computational graph using strictly Core ATen operators.
- **Input Contract:**
  - **Asset:** INT4 Quantized `torch.nn.Module`.
  - **State:** Dummy Input Tensor (Hidden States: `[1, 1, 4096]`, FP32).
  - **State:** Dummy KV Cache Tensor (KV States: `[1, 8, 4096]`, FP32).
- **Output Contract:**
  - **Asset:** `ExportedProgram` object (Pre-Autograd ATen Dialect).
- **Architectural Directive:** The tracing mechanism must use exact static shapes. The graph must not contain any data-dependent control flow (no dynamic `if/else` statements that rely on tensor values) to ensure execution determinism on Android.
- **Validator (`TraceQA`):**
  - Inspect the Abstract Syntax Tree (AST) of the traced graph.
  - Assert the graph node count $> 0$.
  - Assert the graph contains exactly zero `call_function` nodes that map back to Python-native operators.

### Step 1.4: Hardware Delegation (XNNPACK Lowering)

- **Objective:** Map the generic ATen mathematical operators to highly optimized, ARM-specific microkernels (XNNPACK) so the Android CPU can execute the INT4 matrix multiplications natively.
- **Input Contract:**
  - **Asset:** `ExportedProgram` (ATen Dialect).
- **Output Contract:**
  - **Asset:** `EdgeProgramManager` targeting the XNNPACK backend.
- **Architectural Directive:** Lower the graph to the ExecuTorch Edge Dialect. Apply the `XnnpackPartitioner`. This step bakes the hardware acceleration directly into the file, meaning the Kotlin client requires zero configuration to leverage the Android CPU's NEON instructions.
- **Validator (`DelegateQA`):**
  - Parse the compiled graph nodes.
  - Assert the presence of `executorch_call_delegate` operations, confirming that the heavy matrix multiplication layers have been successfully routed to the XNNPACK subsystem.

### Step 1.5: Binary Serialization & Export

- **Objective:** Package the hardware-delegated graph and the INT4 weights into a FlatBuffer format engineered specifically for zero-copy `mmap` ingestion on mobile devices.
- **Input Contract:**
  - **Asset:** `EdgeProgramManager`.
- **Output Contract:**
  - **Asset:** `layer_[N].pte` binary file on the host filesystem.
- **Architectural Directive:** Extract the buffer from the delegated program and write it as a continuous byte stream to disk. Ensure the output directory is cleanly structured to hold 32 sequential files.
- **Validator (`ExportQA`):**
  - Query the host operating system file metrics.
  - Assert `file.exists == True`.
  - Assert `file.size <= 150,000,000` bytes.
