import torch
from ..contracts import IngestOutput

class IngestQA:
    """
    Quality Assurance for Step 1 Ingestor.
    Validates that the output is a valid isolated PyTorch decoder block
    with approximately 1/32nd of Llama 3 (8B) parameter count.
    """
    def __init__(self, target_params: int = 250_000_000, tolerance: float = 0.15):
        self.target_params = target_params
        self.tolerance = tolerance

    def verify(self, output: IngestOutput) -> bool:
        print(f"[QA Ingestor] Running quality checks for Layer {output.layer_index}...")
        
        # 1. Type Assertion
        assert isinstance(output, IngestOutput), "Output must be of type IngestOutput"
        assert isinstance(output.module, torch.nn.Module), "Isolated asset must be a torch.nn.Module"
        
        # 2. Parameter Count Constraint (~250M parameters)
        num_params = output.original_params
        lower_bound = self.target_params * (1.0 - self.tolerance)
        upper_bound = self.target_params * (1.0 + self.tolerance)
        
        assert lower_bound <= num_params <= upper_bound, (
            f"Layer {output.layer_index} parameter count {num_params} "
            f"violates contract target of {self.target_params} +/- {self.tolerance*100}%"
        )
        
        print(f"[QA Ingestor] IngestQA contract check PASSED. Parameter count: {num_params}")
        return True
