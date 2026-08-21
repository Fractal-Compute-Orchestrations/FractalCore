# slicer/pipeline.py
"""
Factory Conveyor Belt Orchestrator.
====================================
This module is the factory floor manager. It sequences workstations,
invokes QA validators between them, and packages raw outputs into
contract data classes for the next station. It performs **no** logic
of its own — it only moves parts along the belt.
"""

from slicer.contracts import IngestOutput
from slicer.workstations import (
    step_1_ingestor,
    step_2_quantizer,
    step_3_tracer,
    step_4_delegator,
    step_5_exporter,
)
from slicer.validators import (
    qa_1_ingestor,
    qa_2_quantizer,
    qa_3_tracer,
    qa_4_delegator,
    qa_5_exporter,
)


def run_factory_pipeline(
    model_path: str = "./assets/raw_models/Meta-Llama-3-8B",
    total_layers: int = 32,
    output_dir: str = "./output",
):
    """
    Main Conveyor Belt Orchestrator loop.
    Acts exclusively as the factory floor manager, handing outputs of one workstation
    to the inputs of the next and verifying constraints at each boundary.
    """
    # 1. Ingest the raw material (Runs once)
    monolithic_model = step_1_ingestor.load_model(model_path)

    for layer_idx in range(total_layers):
        print(f"\n{'='*60}")
        print(f"  PROCESSING LAYER {layer_idx}")
        print(f"{'='*60}")

        # STATION 1: Isolate — returns raw torch.nn.Module
        isolated_layer = step_1_ingestor.isolate_layer(monolithic_model, layer_idx)
        qa_1_ingestor.verify(isolated_layer)

        # Package the validated module into the IngestOutput contract
        # for handoff to Station 2. The pipeline is the contract packager.
        param_count = sum(p.numel() for p in isolated_layer.parameters())
        ingest_output = IngestOutput(
            layer_index=layer_idx,
            module=isolated_layer,
            original_params=param_count,
        )

        # STATION 2: Quantize
        quantized_layer = step_2_quantizer.apply_int4(ingest_output)
        qa_2_quantizer.verify(quantized_layer)

        # STATION 3: Trace Graph
        aten_graph = step_3_tracer.export_to_aten(quantized_layer)
        qa_3_tracer.verify(aten_graph)

        # STATION 4: Delegate to Hardware
        delegated_graph = step_4_delegator.lower_to_xnnpack(aten_graph)
        qa_4_delegator.verify(delegated_graph)

        # STATION 5: Export Payload
        output_path = step_5_exporter.serialize(delegated_graph, layer_idx, output_dir)
        qa_5_exporter.verify(output_path)

    print("\n" + "=" * 60)
    print("  FACTORY PIPELINE COMPLETE — ALL ASSETS VALIDATED")
    print("=" * 60)
