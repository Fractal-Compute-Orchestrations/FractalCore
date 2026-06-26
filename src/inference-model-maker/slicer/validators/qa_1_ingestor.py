import torch
from slicer.contracts import IngestOutput

TARGET_PARAMS = 250_000_000
TOLERANCE = 0.15

def verify(output: IngestOutput) -> bool:
    """
    Quality Assurance for Step 1 Ingestor.
    Asserts output is a valid isolated PyTorch decoder block
    with parameter count ~1/32nd of Llama 3 (8B).
    """
    print(f"[QA Ingestor] Running quality checks for Layer {output.layer_index}...")
    
    # 1. Type Assertion
    assert isinstance(output, IngestOutput), "Output must be of type IngestOutput"
    assert isinstance(output.module, torch.nn.Module), "Isolated asset must be a torch.nn.Module"
    
    # 2. Parameter Count Constraint (~250M parameters)
    num_params = output.original_params
    lower_bound = TARGET_PARAMS * (1.0 - TOLERANCE)
    upper_bound = TARGET_PARAMS * (1.0 + TOLERANCE)
    
    assert lower_bound <= num_params <= upper_bound, (
        f"Layer {output.layer_index} parameter count {num_params} "
        f"violates contract target of {TARGET_PARAMS} +/- {TOLERANCE*100}%"
    )
    
    print(f"[QA Ingestor] IngestQA contract check PASSED. Parameter count: {num_params}")
    return True
