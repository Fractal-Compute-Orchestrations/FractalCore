import os
from ..contracts import ExportOutput

class ExportQA:
    """
    Quality Assurance for Step 5 Exporter.
    Validates that the final serialized edge binary file exists
    on disk and respects the strict 150MB maximum size limit.
    """
    def __init__(self, max_bytes: int = 150_000_000):
        self.max_bytes = max_bytes

    def verify(self, output: ExportOutput) -> bool:
        print(f"[QA Exporter] Running quality checks for Layer {output.layer_index}...")
        
        # 1. Type Assertion
        assert isinstance(output, ExportOutput), "Output must be of type ExportOutput"
        
        # 2. File existence check
        file_path = output.file_path
        assert os.path.exists(file_path), f"PTE binary file not found at: {file_path}"
        
        # 3. File size constraint (<= 150MB)
        file_size = output.file_size_bytes
        assert file_size <= self.max_bytes, (
            f"Layer {output.layer_index} serialized binary is {file_size} bytes, "
            f"exceeding strict mobile budget of {self.max_bytes} bytes (~150MB)"
        )
        
        print(f"[QA Exporter] ExportQA contract check PASSED. File size: {file_size} bytes.")
        return True
