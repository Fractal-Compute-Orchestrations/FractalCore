import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAKER_DIR = os.path.join(PROJECT_ROOT, "src", "inference-model-maker")
if MAKER_DIR not in sys.path:
    sys.path.insert(0, MAKER_DIR)

from slicer.contracts import IngestOutput  # noqa: E402
from slicer.validators import qa_1_ingestor  # noqa: E402
from slicer.workstations import (  # noqa: E402
    step_2_quantizer,
    step_3_tracer,
    step_4_delegator,
    step_5_exporter,
)


class ToyDecoderBlock(torch.nn.Module):
    def __init__(self, hidden_size: int = 128):
        super().__init__()
        self.q_proj = torch.nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.q_proj(x)


def test_qa1_ingestor_supports_custom_model_bounds():
    layer = ToyDecoderBlock(128).to(torch.float16)
    with pytest.raises(RuntimeError):
        qa_1_ingestor.verify(layer)
    assert qa_1_ingestor.verify(layer, min_params=1000, max_params=100_000)


def test_quantizer_measures_real_size_not_mock_constant():
    layer = ToyDecoderBlock(256).to(torch.float16)
    ingest = IngestOutput(
        layer_index=0, module=layer, original_params=sum(p.numel() for p in layer.parameters())
    )
    q_out = step_2_quantizer.apply_int4(ingest)
    assert q_out.in_memory_size_mb > 0.0
    assert q_out.in_memory_size_mb != 115.0


def test_tracer_infers_hidden_size_and_respects_dry_run(tmp_path):
    layer = ToyDecoderBlock(128).to(torch.float16)
    ingest = IngestOutput(
        layer_index=0, module=layer, original_params=sum(p.numel() for p in layer.parameters())
    )
    q_out = step_2_quantizer.apply_int4(ingest)
    trace_out = step_3_tracer.export_to_aten(q_out, dry_run=False)
    assert len(list(trace_out.exported_program.graph.nodes)) > 0

    if not step_4_delegator.HAS_EXECUTORCH:
        with pytest.raises(RuntimeError):
            step_4_delegator.lower_to_xnnpack(trace_out, dry_run=False)
        del_out = step_4_delegator.lower_to_xnnpack(trace_out, dry_run=True)

        with pytest.raises(RuntimeError):
            step_5_exporter.serialize(del_out, 0, str(tmp_path), dry_run=False)
        out_file = step_5_exporter.serialize(del_out, 0, str(tmp_path), dry_run=True)
        assert os.path.exists(out_file)
