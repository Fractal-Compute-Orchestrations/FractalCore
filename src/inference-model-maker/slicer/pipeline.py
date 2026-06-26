import os
from .contracts import (
    IngestInput, QuantizeInput, TraceInput, DelegateInput, ExportInput
)
from .workstations import (
    ModelIngestor, ModelQuantizer, ModelTracer, ModelDelegator, ModelExporter
)
from .validators import (
    IngestQA, QuantizeQA, TraceQA, DelegateQA, ExportQA
)

class SlicerPipeline:
    """
    The Slicer Conveyor Belt (Main Orchestrator loop).
    Sequentially routes inputs and outputs through isolated workstation modules,
    validating structural data shapes and constraints at each boundary.
    """
    def __init__(self, model_id_or_path: str, output_dir: str):
        self.model_id_or_path = model_id_or_path
        self.output_dir = output_dir
        
        # Initialize isolated workstation modules
        self.ingestor = ModelIngestor()
        self.quantizer = ModelQuantizer()
        self.tracer = ModelTracer()
        self.delegator = ModelDelegator()
        self.exporter = ModelExporter()
        
        # Initialize Quality Assurance validators
        self.ingest_qa = IngestQA()
        self.quant_qa = QuantizeQA()
        self.trace_qa = TraceQA()
        self.delegate_qa = DelegateQA()
        self.export_qa = ExportQA()

    def slice_layer(self, layer_index: int) -> str:
        """Runs a single layer partition through all compiler workstations and QA validators."""
        print(f"\n==================== [CONVEYOR BELT] Slicing Layer {layer_index} Start ====================")
        
        # --- Workstation 1: Ingest & Isolate ---
        ingest_in = IngestInput(model_id_or_path=self.model_id_or_path, layer_index=layer_index)
        ingest_out = self.ingestor.execute(ingest_in)
        self.ingest_qa.verify(ingest_out)  # QA checks. Throws AssertionError on failure.
        
        # --- Workstation 2: Quantize ---
        quant_in = QuantizeInput(layer_index=layer_index, module=ingest_out.module)
        quant_out = self.quantizer.execute(quant_in)
        self.quant_qa.verify(quant_out)
        
        # --- Workstation 3: Trace ---
        trace_in = TraceInput(layer_index=layer_index, module=quant_out.module)
        trace_out = self.tracer.execute(trace_in)
        self.trace_qa.verify(trace_out)
        
        # --- Workstation 4: Delegate ---
        delegate_in = DelegateInput(layer_index=layer_index, exported_program=trace_out.exported_program)
        delegate_out = self.delegator.execute(delegate_in)
        self.delegate_qa.verify(delegate_out)
        
        # --- Workstation 5: Serialize & Export ---
        output_path = os.path.join(self.output_dir, f"layer_{layer_index}.pte")
        export_in = ExportInput(
            layer_index=layer_index,
            delegated_program=delegate_out.delegated_program,
            output_path=output_path
        )
        export_out = self.exporter.execute(export_in)
        self.export_qa.verify(export_out)
        
        print(f"==================== [CONVEYOR BELT] Layer {layer_index} Completed: {export_out.file_path} ====================\n")
        return export_out.file_path

    def run(self, num_layers: int = 32):
        """Orchestrates sequential slicing across the full model decoders stack."""
        print(f"Starting Slicer Pipeline Orchestrator on model: {self.model_id_or_path}")
        print(f"Export target directory: {self.output_dir}")
        os.makedirs(self.output_dir, exist_ok=True)
        
        summary = []
        for idx in range(num_layers):
            try:
                saved_path = self.slice_layer(idx)
                summary.append((idx, True, saved_path))
            except Exception as e:
                print(f"[FATAL ERROR] Slicing layer {idx} failed: {e}")
                summary.append((idx, False, str(e)))
                print("[HALT CONDITION] Stopping the conveyor belt to prevent system corruption.")
                raise e
                
        print("\nSlicing Pipeline Complete. Results:")
        for idx, success, detail in summary:
            status = "SUCCESS" if success else "FAILED"
            print(f"  Layer {idx:02d}: {status} - {detail}")
            
        return summary
