from slicer.contracts import TraceOutput, DelegateOutput

try:
    from executorch.exir import to_edge
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )

    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False


def lower_to_xnnpack(aten_graph: TraceOutput) -> DelegateOutput:
    """Delegates core operators in the ATen graph to the ARM XNNPACK subsystem."""
    layer_idx = aten_graph.layer_index
    exported_program = aten_graph.exported_program
    print(f"[Workstation 4] Delegating layer {layer_idx} to XNNPACK backend...")

    if HAS_EXECUTORCH:
        try:
            print("[Workstation 4] Converting ATen ExportedProgram to Edge Dialect...")
            edge_program = to_edge(exported_program)

            print("[Workstation 4] Applying XNNPACK Partitioner...")
            delegated_program = edge_program.to_backend(XnnpackPartitioner())
        except Exception as e:
            print(
                f"[Workstation 4] Delegation failed: {e}. Falling back to simulation delegated program."
            )

            class MockEdgeProgramManager:
                def __init__(self, exported_program):
                    self.exported_program = exported_program

            delegated_program = MockEdgeProgramManager(exported_program)
    else:
        print(
            "[Workstation 4] executorch not found. Creating simulated EdgeProgramManager wrapper."
        )

        class MockEdgeProgramManager:
            def __init__(self, exported_program):
                self.exported_program = exported_program

        delegated_program = MockEdgeProgramManager(exported_program)

    return DelegateOutput(layer_index=layer_idx, delegated_program=delegated_program)
