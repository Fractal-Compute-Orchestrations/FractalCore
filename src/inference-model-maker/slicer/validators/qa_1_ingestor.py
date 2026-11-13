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


def _validate_parameter_count(
    module: torch.nn.Module,
    min_params: int = PARAM_COUNT_LOWER_BOUND,
    max_params: int = PARAM_COUNT_UPPER_BOUND,
) -> int:
    """Assert the parameter count sits inside the configured contract envelope."""
    param_count: int = _count_parameters(module)

    if not (min_params < param_count < max_params):
        raise RuntimeError(
            f"[QA-1 FAIL | PARAM COUNT] Isolated layer has {param_count:,} "
            f"parameters. Contract requires strictly between "
            f"{min_params:,} and {max_params:,}. "
            f"Pass min_params/max_params for non-Llama3-8B architectures."
        )

    return param_count


def _validate_dtype(
    module: torch.nn.Module, required_dtype: torch.dtype = REQUIRED_DTYPE
) -> None:
    """Assert every parameter tensor matches *required_dtype*."""
    for name, param in module.named_parameters():
        if param.dtype != required_dtype:
            raise RuntimeError(
                f"[QA-1 FAIL | DTYPE] Parameter '{name}' has dtype "
                f"{param.dtype}, but the contract mandates {required_dtype}."
            )


def verify(
    isolated_layer: torch.nn.Module,
    min_params: int = PARAM_COUNT_LOWER_BOUND,
    max_params: int = PARAM_COUNT_UPPER_BOUND,
    required_dtype: torch.dtype = REQUIRED_DTYPE,
) -> bool:
    """Run all Step 1 Output Contract checks against *isolated_layer*."""
    print("[QA-1] Running Step 1 Ingestor Quality Gate")

    _validate_module_type(isolated_layer)
    print("[QA-1] [OK] Type check PASSED: object is torch.nn.Module")

    param_count: int = _validate_parameter_count(
        isolated_layer, min_params=min_params, max_params=max_params
    )
    print(
        f"[QA-1] [OK] Param count PASSED: {param_count:,} parameters "
        f"(bounds: {min_params:,} < n < {max_params:,})"
    )

    _validate_dtype(isolated_layer, required_dtype=required_dtype)
    print(f"[QA-1] [OK] Dtype check PASSED: all parameters are {required_dtype}")

    print("[QA-1] All Step 1 contract checks PASSED")
    return True
