from ..contracts import DelegateInput, DelegateOutput

try:
    from executorch.exir import to_edge
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False

class ModelDelegator:
    """
    Delegator workstation. Compiles Edge IR operators and partitions the graph
    to route compatible matrix operations to the ARM XNNPACK backend.
    """
    def execute(self, config: DelegateInput) -> DelegateOutput:
        print(f"[Workstation 4] Delegating layer {config.layer_index} to XNNPACK backend...")
        
        exported_program = config.exported_program
        
        if HAS_EXECUTORCH:
            try:
                print("[Workstation 4] Converting ATen ExportedProgram to Edge Dialect...")
                edge_program = to_edge(exported_program)
                
                print("[Workstation 4] Applying XNNPACK Partitioner...")
                # Lower to XNNPACK backend
                delegated_program = edge_program.to_backend(XnnpackPartitioner())
            except Exception as e:
                print(f"[Workstation 4] Delegation failed: {e}. Falling back to simulation delegated program.")
                class MockEdgeProgramManager:
                    def __init__(self, exported_program):
                        self.exported_program = exported_program
                delegated_program = MockEdgeProgramManager(exported_program)
        else:
            print("[Workstation 4] executorch not found. Creating simulated EdgeProgramManager wrapper.")
            class MockEdgeProgramManager:
                def __init__(self, exported_program):
                    self.exported_program = exported_program
            delegated_program = MockEdgeProgramManager(exported_program)

        return DelegateOutput(
            layer_index=config.layer_index,
            delegated_program=delegated_program
        )
