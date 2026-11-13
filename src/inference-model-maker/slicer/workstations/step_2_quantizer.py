import os
import sys

from slicer.contracts import IngestOutput, QuantizeOutput

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", "..", ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

from fractal_quant.codec import quantize_state_dict  # noqa: E402

try:
    import torchao  # noqa: F401
    from torchao.quantization import Int4WeightOnlyConfig, quantize_

    HAS_TORCHAO = True
except ImportError:
    HAS_TORCHAO = False


def apply_int4(
    isolated_layer: IngestOutput, dry_run: bool = False
) -> QuantizeOutput:
    """Apply INT4 quantization via torchao or fractal_quant micro-block packing."""
    layer_idx = isolated_layer.layer_index
    module = isolated_layer.module
    print(f"[Workstation 2] Quantizing layer {layer_idx} to INT4...")

    if HAS_TORCHAO:
        try:
            print(
                "[Workstation 2] Using torchao for weight-only INT4 quantization (group_size=128)"
            )
            quant_config = Int4WeightOnlyConfig(group_size=128)
            quantize_(module, quant_config)
            total_bytes = sum(
                p.nelement() * p.element_size() for p in module.parameters()
            )
            in_memory_mb = total_bytes / (1024 * 1024)
            print(f"[Workstation 2] Layer {layer_idx} size is {in_memory_mb:.2f} MB")
            return QuantizeOutput(
                layer_index=layer_idx,
                module=module,
                in_memory_size_mb=in_memory_mb,
            )
        except Exception as exc:
            if not dry_run:
                print(
                    f"[Workstation 2] torchao failed ({exc}); falling back to FRQD micro-block packing."
                )

    try:
        state_np = {
            name: param.detach().cpu().float().numpy()
            for name, param in module.named_parameters()
        }
        if not state_np:
            raise RuntimeError(f"Layer {layer_idx} contains no parameters to quantize.")
        packed, _ = quantize_state_dict(state_np, block_size=64)
        in_memory_mb = len(packed) / (1024 * 1024)
    except Exception as exc:
        if dry_run:
            total_bytes = sum(
                p.nelement() * p.element_size() for p in module.parameters()
            )
            in_memory_mb = (total_bytes / 7.1) / (1024 * 1024)
        else:
            raise RuntimeError(
                f"INT4 quantization failed for layer {layer_idx}: {exc}"
            ) from exc

    print(f"[Workstation 2] Layer {layer_idx} size is {in_memory_mb:.2f} MB")
    return QuantizeOutput(
        layer_index=layer_idx, module=module, in_memory_size_mb=in_memory_mb
    )
