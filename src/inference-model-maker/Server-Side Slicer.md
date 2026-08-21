# Module 1: The Slicer (Server-Side Graph Surgery Pipeline)

**Architectural Role**: The Offline Compiler & Graph Surgeon  
**Target Environment**: Host Server / Desktop (Python Environment)  
**Target Asset**: Foundation Model (e.g. Llama 3 8B, TinyLlama) Monolithic Checkpoint  
**Design Pattern**: Strict Contract-Driven Pipeline  

---

## 1. Module Philosophy

This module operates entirely off-device on the FractalCore server. Its sole purpose is to ingest a massive, un-executable 16GB PyTorch model and compile it into independent, hardware-optimized, memory-safe binary payloads (`.pte` files) $\le 150\text{MB}$.

Android clients must never perform graph compilation, quantization, or graph tracing. The Slicer absorbs 100% of the preprocessing complexity to ensure that Android edge nodes act as lightweight, hyper-fast execution runtimes.

---

## 2. Step-by-Step Contract Pipeline

### Step 1.1: Monolithic Ingestion and Structural Isolation

- **Objective**: Mount the model into host memory safely and sever a single transformer decoder block from the sequential stack without breaking parameter references.
- **Input Contract**:
  - **Asset**: Hugging Face model identifier or local directory path.
  - **Data Type**: FP16 or BF16 weights.
- **Output Contract**:
  - **Asset**: A single, isolated `torch.nn.Module` representing exactly one layer (e.g., `model.layers[5]`).
- **Architectural Directive**: The ingestor maps the model to host CPU (`device_map="cpu"`) to prevent VRAM exhaustion. The module retains its layer normalization and self-attention topological boundaries.
- **Validator (`IngestQA`)**:
  - Assert the isolated asset is an instance of `torch.nn.Module`.
  - Calculate total parameter count of the isolated block and assert it is $\approx 1/N$ of the total model parameter count.

### Step 1.2: Weight-Only Quantization (INT4)

- **Objective**: Compress the FP16 weight matrices of the isolated layer to hit strict mobile RAM safety boundaries while configuring activations to dynamically quantize to INT8 at runtime.
- **Input Contract**:
  - **Asset**: Unquantized `torch.nn.Module` (Size: $\approx 450\text{MB}$).
- **Output Contract**:
  - **Asset**: Mutated `torch.nn.Module` with INT4 weights (Target Size: $\approx 115\text{MB}$).
- **Architectural Directive**: Apply Grouped INT4 Quantization (group size 128) via `torchao`. This specifically targets the `q_proj`, `k_proj`, `v_proj`, and `o_proj` linear attention layers alongside MLP projection layers.
- **Validator (`QuantQA`)**:
  - Traverse the module state dictionary.
  - Assert that the byte size of target linear layers reflects a 4-bit footprint.
  - Assert the total in-memory footprint of the layer is strictly $\le 150\text{MB}$.

### Step 1.3: Static Graph Tracing (ATen Dialect Export)

- **Objective**: Eliminate the Python runtime environment entirely. Freeze dynamic PyTorch execution into a static computational graph using strictly Core ATen operators.
- **Input Contract**:
  - **Asset**: INT4 Quantized `torch.nn.Module`.
  - **State**: Dummy Input Tensor (Hidden States: `[1, 1, 4096]`, FP32).
  - **State**: Dummy KV Cache Tensor (KV States: `[1, 8, 4096]`, FP32).
- **Output Contract**:
  - **Asset**: `ExportedProgram` object (Pre-Autograd ATen Dialect).
- **Architectural Directive**: The tracing mechanism uses exact static shapes. The graph must not contain data-dependent control flow to guarantee execution determinism on Android.
- **Validator (`TraceQA`)**:
  - Inspect the traced graph nodes.
  - Assert graph node count $> 0$.
  - Assert the graph contains zero `call_function` nodes that map back to Python-native operators.

### Step 1.4: Hardware Delegation (XNNPACK Lowering)

- **Objective**: Map generic ATen mathematical operators to optimized ARM-specific microkernels (XNNPACK) so the Android CPU executes INT4 matrix multiplications natively via ARM NEON.
- **Input Contract**:
  - **Asset**: `ExportedProgram` (ATen Dialect).
- **Output Contract**:
  - **Asset**: `EdgeProgramManager` targeting the XNNPACK backend.
- **Architectural Directive**: Lower the graph to ExecuTorch Edge Dialect using the `XnnpackPartitioner`.
- **Validator (`DelegateQA`)**:
  - Parse compiled graph nodes.
  - Assert the presence of `executorch_call_delegate` operations, confirming that matrix multiplication layers route to the XNNPACK subsystem.

### Step 1.5: Binary Serialization & Export

- **Objective**: Package the hardware-delegated graph and INT4 weights into a FlatBuffer format engineered for zero-copy `mmap` ingestion on mobile devices.
- **Input Contract**:
  - **Asset**: `EdgeProgramManager`.
- **Output Contract**:
  - **Asset**: `layer_[N].pte` binary file on the host filesystem.
- **Architectural Directive**: Extract the buffer from the delegated program and write it as a contiguous byte stream to disk.
- **Validator (`ExportQA`)**:
  - Query the host operating system file metrics.
  - Assert `file.exists == True`.
  - Assert `file.size <= 150_000_000` bytes.
