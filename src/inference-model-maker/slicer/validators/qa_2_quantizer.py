import torch
from slicer.contracts import QuantizeOutput

MAX_SIZE_MB = 150.0


def verify(output: QuantizeOutput) -> bool:
    """
    Quality Assurance for Step 2 Quantizer.
    Asserts in-memory layer size limit (<= 150MB).
    """
    print(f"[QA Quantizer] Running quality checks for Layer {output.layer_index}...")

    # 1. Type Assertion
    assert isinstance(output, QuantizeOutput), "Output must be of type QuantizeOutput"
    assert isinstance(
        output.module, torch.nn.Module
    ), "Quantized asset must be a torch.nn.Module"

    # 2. Size limit assertion
    total_mb = output.in_memory_size_mb
    assert total_mb <= MAX_SIZE_MB, (
        f"Quantized layer {output.layer_index} size of {total_mb:.2f}MB "
        f"exceeds strict contract limit of {MAX_SIZE_MB}MB"
    )

    print(
        f"[QA Quantizer] QuantQA contract check PASSED. Size in RAM: {total_mb:.2f} MB"
    )
    return True
