# slicer/pipeline.py

from slicer.workstations import step_1_ingestor, step_2_quantizer, step_3_tracer, step_4_delegator, step_5_exporter
from slicer.validators import qa_1_ingestor, qa_2_quantizer, qa_3_tracer, qa_4_delegator, qa_5_exporter

def run_factory_pipeline(model_id: str, total_layers: int = 32, output_dir: str = "./output"):
    """
    Main Conveyor Belt Orchestrator loop.
    Acts exclusively as the factory floor manager, handing outputs of one workstation
    to the inputs of the next and verifying constraints at each boundary.
    """
    # 1. Ingest the raw material (Runs once)
    monolithic_model = step_1_ingestor.load_model(model_id)
    
    for layer_idx in range(total_layers):
        print(f"--- Processing Layer {layer_idx} ---")
        
        # STATION 1: Isolate
        isolated_layer = step_1_ingestor.isolate_layer(monolithic_model, layer_idx)
        qa_1_ingestor.verify(isolated_layer)
        
        # STATION 2: Quantize
        quantized_layer = step_2_quantizer.apply_int4(isolated_layer)
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
        
    print("Factory Pipeline Complete. All assets validated.")
