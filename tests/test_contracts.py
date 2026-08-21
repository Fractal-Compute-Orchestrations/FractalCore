import os
import sys
import pytest

# Add contracts directory directly
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SLICER_DIR = os.path.join(PROJECT_ROOT, "src", "inference-model-maker", "slicer")
if SLICER_DIR not in sys.path:
    sys.path.insert(0, SLICER_DIR)

from contracts import (  # noqa: E402
    DelegateInput,
    DelegateOutput,
    ExportInput,
    ExportOutput,
    IngestInput,
    IngestOutput,
    QuantizeInput,
    QuantizeOutput,
    TraceInput,
    TraceOutput,
)


def test_ingest_contracts():
    inp = IngestInput(model_id_or_path="test/model", layer_index=0)
    assert inp.model_id_or_path == "test/model"
    assert inp.layer_index == 0

    out = IngestOutput(layer_index=0, module=None, original_params=1000)
    assert out.layer_index == 0
    assert out.original_params == 1000

    # Verify frozen immutability
    with pytest.raises(Exception):
        inp.layer_index = 1  # type: ignore[misc]


def test_quantize_contracts():
    inp = QuantizeInput(layer_index=2, module=None)
    assert inp.layer_index == 2
    assert inp.module is None

    out = QuantizeOutput(layer_index=2, module=None, in_memory_size_mb=115.0)
    assert out.in_memory_size_mb == 115.0


def test_trace_contracts():
    inp = TraceInput(layer_index=3, module=None)
    assert inp.layer_index == 3

    out = TraceOutput(layer_index=3, exported_program="mock_program")
    assert out.exported_program == "mock_program"


def test_delegate_contracts():
    inp = DelegateInput(layer_index=4, exported_program="mock_program")
    assert inp.layer_index == 4

    out = DelegateOutput(layer_index=4, delegated_program="mock_delegated")
    assert out.delegated_program == "mock_delegated"


def test_export_contracts():
    inp = ExportInput(
        layer_index=5,
        delegated_program="mock_delegated",
        output_path="/tmp/layer_5.pte",
    )
    assert inp.output_path == "/tmp/layer_5.pte"

    out = ExportOutput(
        layer_index=5, file_path="/tmp/layer_5.pte", file_size_bytes=1024
    )
    assert out.file_size_bytes == 1024
