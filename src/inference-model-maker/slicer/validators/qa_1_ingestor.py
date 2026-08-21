"""
slicer/validators/qa_1_ingestor.py
===================================
Quality Assurance Gate for Workstation 1: Monolithic Ingestion & Structural Isolation.

This validator is the first line of defense in the Fractal factory pipeline.
It programmatically guarantees that the isolated decoder layer conforms to the
Step 1 Output Contract before it is handed off to Workstation 2 (Quantization).

Contract Assertions
--------------------
1. The isolated layer MUST be an instance of ``torch.nn.Module``.
2. Its total parameter count MUST fall within the mathematically sound range
   for a single Llama 3 8B decoder block (150M < count < 350M).
3. Every learnable parameter tensor MUST have dtype ``torch.float16``.

Halt Condition
---------------
If any assertion fails, a descriptive ``RuntimeError`` is raised, halting the
factory pipeline immediately. This module performs zero mutation of the
artifact under test — it is strictly read-only.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Contract Constants
# ---------------------------------------------------------------------------
# Llama 3 8B has ~8.03B parameters spread across 32 decoder layers.
# A single layer is therefore ~250M parameters. The bounds below are a firm,
# but deliberately generous, engineering envelope around that expectation.
PARAM_COUNT_LOWER_BOUND: int = 150_000_000
PARAM_COUNT_UPPER_BOUND: int = 350_000_000
REQUIRED_DTYPE: torch.dtype = torch.float16


def _count_parameters(module: torch.nn.Module) -> int:
    """Return the total scalar parameter count of *module*.

    Counts every parameter tensor, including those with
    ``requires_grad=False``, since the contract cares about structural
    (memory) footprint, not trainability.
    """
    return sum(p.numel() for p in module.parameters())


def _validate_module_type(isolated_layer: object) -> None:
    """Assert *isolated_layer* is a ``torch.nn.Module``.

    Raises
    ------
    RuntimeError
        If the object is not a ``torch.nn.Module`` instance.
    """
    if not isinstance(isolated_layer, torch.nn.Module):
        raise RuntimeError(
            f"[QA-1 FAIL | TYPE] Expected torch.nn.Module, received "
            f"{type(isolated_layer).__qualname__}. Workstation 1 returned "
            f"an invalid artifact -- halting pipeline."
        )


def _validate_parameter_count(module: torch.nn.Module) -> int:
    """Assert the parameter count sits inside the contract envelope.

    Returns
    -------
    int
        The validated parameter count, for downstream logging.

    Raises
    ------
    RuntimeError
        If the count falls outside
        ``(PARAM_COUNT_LOWER_BOUND, PARAM_COUNT_UPPER_BOUND)``.
    """
    param_count: int = _count_parameters(module)

    if not (PARAM_COUNT_LOWER_BOUND < param_count < PARAM_COUNT_UPPER_BOUND):
        raise RuntimeError(
            f"[QA-1 FAIL | PARAM COUNT] Isolated layer has {param_count:,} "
            f"parameters. Contract requires strictly between "
            f"{PARAM_COUNT_LOWER_BOUND:,} and {PARAM_COUNT_UPPER_BOUND:,} "
            f"(expected ~250M for a single Llama 3 8B decoder block). "
            f"This indicates an incorrect layer extraction, an off-by-one "
            f"`layer_index`, or a model/architecture mismatch."
        )

    return param_count


def _validate_dtype(module: torch.nn.Module) -> None:
    """Assert every parameter tensor is ``torch.float16``.

    Raises
    ------
    RuntimeError
        If any parameter has a dtype other than ``torch.float16``.
    """
    for name, param in module.named_parameters():
        if param.dtype != REQUIRED_DTYPE:
            raise RuntimeError(
                f"[QA-1 FAIL | DTYPE] Parameter '{name}' has dtype "
                f"{param.dtype}, but the contract mandates "
                f"{REQUIRED_DTYPE}. Workstation 1 must load/cast all "
                f"weights to float16 prior to handoff."
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def verify(isolated_layer: torch.nn.Module) -> bool:
    """Run all Step 1 Output Contract checks against *isolated_layer*.

    This is the sole entry point called by the orchestrator between
    Workstation 1 (Ingestion) and Workstation 2 (Quantization).

    Parameters
    ----------
    isolated_layer : torch.nn.Module
        A single, surgically extracted Llama 3 8B decoder block.

    Returns
    -------
    bool
        ``True`` if every assertion passes.

    Raises
    ------
    RuntimeError
        If any contract assertion fails, identifying the exact violation.
    """
    print("[QA-1] -- Running Step 1 Ingestor Quality Gate --")

    _validate_module_type(isolated_layer)
    print("[QA-1]  [OK] Type check PASSED -- object is torch.nn.Module")

    param_count: int = _validate_parameter_count(isolated_layer)
    print(
        f"[QA-1]  [OK] Param count PASSED -- {param_count:,} parameters "
        f"(bounds: {PARAM_COUNT_LOWER_BOUND:,} < n < {PARAM_COUNT_UPPER_BOUND:,})"
    )

    _validate_dtype(isolated_layer)
    print(f"[QA-1]  [OK] Dtype check PASSED -- all parameters are {REQUIRED_DTYPE}")

    print("[QA-1] -- All Step 1 contract checks PASSED --")
    return True
