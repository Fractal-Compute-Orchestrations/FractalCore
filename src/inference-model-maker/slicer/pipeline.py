"""Factory Conveyor Belt Orchestrator.

Sequences workstations, invokes QA validators between stages, and packages
outputs into immutable contracts for downstream steps.
"""

from __future__ import annotations

from typing import Optional

from slicer.contracts import IngestOutput
from slicer.validators import (
    qa_1_ingestor,
    qa_2_quantizer,
    qa_3_tracer,
    qa_4_delegator,
    qa_5_exporter,
)
from slicer.workstations import (
    step_1_ingestor,
    step_2_quantizer,
    step_3_tracer,
    step_4_delegator,
    step_5_exporter,
)


def _infer_param_bounds(isolated_layer, min_params: Optional[int], max_params: Optional[int]) -> tuple[int, int]:
    if min_params is not None and max_params is not None:
        return min_params, max_params
    count = sum(p.numel() for p in isolated_layer.parameters())
    lo = min_params if min_params is not None else max(1, int(count * 0.5))
    hi = max_params if max_params is not None else max(count + 1, int(count * 1.5))
    return lo, hi


def run_factory_pipeline(
    model_path: str = "./assets/raw_models/Meta-Llama-3-8B",
    total_layers: int = 32,
    output_dir: str = "./output",
    dry_run: bool = False,
    min_layer_params: Optional[int] = None,
    max_layer_params: Optional[int] = None,
):
    """Run the full Fractal layer slicing pipeline across *total_layers*."""
    monolithic_model = step_1_ingestor.load_model(model_path)

    for layer_idx in range(total_layers):
        print(f"\n{'=' * 60}")
        print(f"  PROCESSING LAYER {layer_idx}")
        print(f"{'=' * 60}")

        isolated_layer = step_1_ingestor.isolate_layer(monolithic_model, layer_idx)
        lo, hi = _infer_param_bounds(isolated_layer, min_layer_params, max_layer_params)
        qa_1_ingestor.verify(isolated_layer, min_params=lo, max_params=hi)

        param_count = sum(p.numel() for p in isolated_layer.parameters())
        ingest_output = IngestOutput(
            layer_index=layer_idx,
            module=isolated_layer,
            original_params=param_count,
        )

        quantized_layer = step_2_quantizer.apply_int4(ingest_output, dry_run=dry_run)
        qa_2_quantizer.verify(quantized_layer)

        aten_graph = step_3_tracer.export_to_aten(quantized_layer, dry_run=dry_run)
        qa_3_tracer.verify(aten_graph)

        delegated_graph = step_4_delegator.lower_to_xnnpack(aten_graph, dry_run=dry_run)
        qa_4_delegator.verify(delegated_graph)

        output_path = step_5_exporter.serialize(
            delegated_graph, layer_idx, output_dir, dry_run=dry_run
        )
        qa_5_exporter.verify(output_path)

    print("\n" + "=" * 60)
    print("  FACTORY PIPELINE COMPLETE: ALL ASSETS VALIDATED")
    print("=" * 60)
