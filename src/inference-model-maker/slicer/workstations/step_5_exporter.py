import os

from slicer.contracts import DelegateOutput


def serialize(
    delegated_graph: DelegateOutput,
    layer_idx: int,
    output_dir: str,
    dry_run: bool = False,
) -> str:
    """Serialize the delegated graph to an ExecuTorch FlatBuffer segment (.pte)."""
    output_path = os.path.join(output_dir, f"layer_{layer_idx}.pte")
    print(
        f"[Workstation 5] Exporting layer {layer_idx} to PTE format at '{output_path}'..."
    )

    delegated_program = delegated_graph.delegated_program
    os.makedirs(output_dir, exist_ok=True)

    if hasattr(delegated_program, "to_executorch"):
        print(
            "[Workstation 5] Calling delegated_program.to_executorch() and saving buffer..."
        )
        et_program = delegated_program.to_executorch()
        with open(output_path, "wb") as f:
            f.write(et_program.buffer)
    elif dry_run:
        print("[Workstation 5] dry_run=True: Writing simulated PTE flatbuffer header.")
        _write_mock_pte(output_path)
    else:
        raise RuntimeError(
            f"Cannot export non-ExecuTorch delegated object for layer {layer_idx} without dry_run=True."
        )

    file_size = os.path.getsize(output_path)
    print(
        f"[Workstation 5] Layer {layer_idx} exported successfully. Size: {file_size} bytes."
    )
    return output_path


def _write_mock_pte(file_path: str):
    """Write a dry-run placeholder PTE header."""
    with open(file_path, "wb") as f:
        f.write(b"FRACTAL_PTE_DRYRUN_V1\x00")
