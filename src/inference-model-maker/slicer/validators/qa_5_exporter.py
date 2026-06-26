import os

MAX_BYTES = 150_000_000

def verify(output_path: str) -> bool:
    """
    Quality Assurance for Step 5 Exporter.
    Asserts final PTE binary file exists and fits within strict 150MB budget.
    """
    print(f"[QA Exporter] Running quality checks for file: '{output_path}'...")
    
    # 1. File existence check
    assert os.path.exists(output_path), f"PTE binary file not found at: {output_path}"
    
    # 2. Size assertion (<= 150MB)
    file_size = os.path.getsize(output_path)
    assert file_size <= MAX_BYTES, (
        f"Serialized binary size {file_size} bytes exceeds limit of {MAX_BYTES} bytes"
    )
    
    print(f"[QA Exporter] ExportQA contract check PASSED. File size: {file_size} bytes.")
    return True
