import torch
from torch.export import export
from slicer.contracts import QuantizeOutput, TraceOutput


def _infer_sample_input(module: torch.nn.Module) -> tuple[int, torch.dtype]:
    """Infer transformer hidden dimension and dtype directly from module parameters."""
    hidden_size = getattr(module, "hidden_size", None)
    dtype = torch.float32
    params = list(module.parameters())
    if params:
        dtype = params[0].dtype
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        for name, param in module.named_parameters():
            if param.ndim == 2 and any(
                tag in name for tag in ("q_proj", "in_features", "c_attn")
            ):
                hidden_size = int(param.shape[1])
                break
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            for p in params:
                if p.ndim == 1 and p.shape[0] >= 64:
                    hidden_size = int(p.shape[0])
                    break
    if not isinstance(hidden_size, int) or hidden_size <= 0:
        hidden_size = 4096
    return hidden_size, dtype


def export_to_aten(
    quantized_layer: QuantizeOutput, dry_run: bool = False
) -> TraceOutput:
    """Trace the quantized layer module into core ATen graph representations."""
    layer_idx = quantized_layer.layer_index
    module = quantized_layer.module
    module.eval()

    hidden_size, dtype = _infer_sample_input(module)
    print(
        f"[Workstation 3] Tracing layer {layer_idx} (hidden={hidden_size}, dtype={dtype}) to ATen dialect..."
    )
    x = torch.randn(1, 1, hidden_size, dtype=dtype)

    try:
        print("[Workstation 3] Executing torch.export.export()...")
        exported_program = export(module, (x,))
        print(
            f"[Workstation 3] Graph trace successful. Nodes captured: {len(exported_program.graph.nodes)}"
        )
    except Exception as exc:
        if not dry_run:
            raise RuntimeError(
                f"ATen tracing failed for layer {layer_idx} (pass dry_run=True to simulate): {exc}"
            ) from exc
        print(
            f"[Workstation 3] dry_run=True: Tracing failed ({exc}); returning simulation graph."
        )

        class MockGraphNode:
            def __init__(self, op, target):
                self.op = op
                self.target = target

        class MockGraph:
            def __init__(self):
                self.nodes = [
                    MockGraphNode("placeholder", "x"),
                    MockGraphNode("call_function", "torch.ops.aten.add.Tensor"),
                    MockGraphNode("output", "out"),
                ]

        class MockExportedProgram:
            def __init__(self):
                self.graph = MockGraph()

        exported_program = MockExportedProgram()

    return TraceOutput(layer_index=layer_idx, exported_program=exported_program)
