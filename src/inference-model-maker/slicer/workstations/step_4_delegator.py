from slicer.contracts import DelegateOutput, TraceOutput

try:
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
        XnnpackPartitioner,
    )
    from executorch.exir import to_edge

    HAS_EXECUTORCH = True
except ImportError:
    HAS_EXECUTORCH = False


def lower_to_xnnpack(
    aten_graph: TraceOutput, dry_run: bool = False
) -> DelegateOutput:
    """Delegate operators in the ATen graph to the ARM XNNPACK backend."""
    layer_idx = aten_graph.layer_index
    exported_program = aten_graph.exported_program
    print(f"[Workstation 4] Delegating layer {layer_idx} to XNNPACK backend...")

    if HAS_EXECUTORCH:
        try:
            print("[Workstation 4] Converting ATen ExportedProgram to Edge Dialect...")
            edge_program = to_edge(exported_program)
            print("[Workstation 4] Applying XNNPACK Partitioner...")
            delegated_program = edge_program.to_backend(XnnpackPartitioner())
            return DelegateOutput(
                layer_index=layer_idx, delegated_program=delegated_program
            )
        except Exception as exc:
            if not dry_run:
                raise RuntimeError(
                    f"ExecuTorch XNNPACK lowering failed for layer {layer_idx}: {exc}"
                ) from exc
            print(
                f"[Workstation 4] dry_run=True: Delegation failed ({exc}); creating wrapper."
            )
    elif not dry_run:
        raise RuntimeError(
            "executorch is required to delegate ATen graphs to XNNPACK (pass dry_run=True for simulation)."
        )
    else:
        print(
            "[Workstation 4] dry_run=True: executorch unavailable, using simulation wrapper."
        )

    class MockEdgeProgramManager:
        def __init__(self, exported_program):
            self.exported_program = exported_program

    return DelegateOutput(
        layer_index=layer_idx,
        delegated_program=MockEdgeProgramManager(exported_program),
    )
