"""
slicer/workstations/step_1_ingestor.py
=======================================
Workstation 1: Monolithic Ingestion & Structural Isolation.

This workstation has exactly two responsibilities and zero validation logic:

1. ``load_model``    -- Ingest the full monolithic Hugging Face checkpoint onto
                        the host CPU in ``torch.float16``, preventing VRAM
                        exhaustion on machines without a high-VRAM GPU.
2. ``isolate_layer`` -- Navigate the model's module tree and sever a single
                        transformer decoder block, returning it as an
                        independent ``torch.nn.Module`` that retains its full
                        topological boundaries and parameter references.

All quality checks (type, parameter count, dtype) are performed exclusively
by ``qa_1_ingestor.verify()`` downstream in the pipeline. This module makes
no assertions of its own and contains no simulated/mocked fallback path --
it operates strictly against real checkpoints, on real hardware. If the
``transformers`` dependency is missing, or the checkpoint cannot be
resolved, this module fails loudly and immediately rather than masking the
failure behind synthetic data.

Architecture Notes
-------------------
* Llama 3 / Mistral / Qwen store decoder blocks at ``model.model.layers[i]``.
* GPT-NeoX / Falcon variants store them at ``model.transformer.h[i]``.
  Both paths are resolved for portability, though Project Fractal's current
  target asset is the Llama 3 (8B) monolithic checkpoint.
"""

from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM


def load_model(model_path: str = "T:\\models\\Meta-Llama-3-8B") -> torch.nn.Module:
    """Ingest the monolithic local model checkpoint onto the host CPU.

    Loads the full ``AutoModelForCausalLM`` in ``torch.float16`` with
    ``device_map="cpu"``, guaranteeing the checkpoint never touches
    GPU VRAM. ``low_cpu_mem_usage`` is enabled to reduce peak RAM.
    ``local_files_only=True`` is enforced so the workstation never hits
    the network -- the model MUST already be provisioned on disk by
    ``slicer/scripts/00_download_model.py``.

    .. note::
        Uses ``dtype=`` (not the deprecated ``torch_dtype=``) to avoid
        the deprecation warning introduced in transformers 4.56+/5.x.
        Requires ``accelerate`` to be installed for ``device_map`` support.

    Parameters
    ----------
    model_path : str
        A local filesystem path to the pre-downloaded checkpoint
        (default: ``"./assets/raw_models/Meta-Llama-3-8B"``).

    Returns
    -------
    torch.nn.Module
        The full, monolithic causal-LM model resident on CPU in float16.

    Raises
    ------
    OSError
        If the checkpoint cannot be resolved at ``model_path``.
    ImportError
        If ``accelerate`` is not installed (required by ``device_map``).
    Exception
        Any other error raised by ``from_pretrained`` propagates unmodified.
    """
    print(f"[Workstation-1] Loading monolithic checkpoint from: '{model_path}'")
    print("[Workstation-1] Enforcing device_map='cpu', dtype=float16, local_files_only=True")

    model: torch.nn.Module = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.float16,
        device_map="cpu",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )

    print(
        f"[Workstation-1] Model loaded. Type: {type(model).__qualname__} | "
        f"device=cpu | dtype=float16"
    )
    return model


def isolate_layer(model: torch.nn.Module, layer_index: int) -> torch.nn.Module:
    """Sever a single transformer decoder block from the monolithic model.

    Navigates the model's internal module tree to locate the decoder layer
    list, extracts the block at ``layer_index``, and returns it as a
    standalone ``torch.nn.Module``. The returned module retains its full
    topological boundaries (all sub-modules, parameters, and buffers) -- it
    is a live reference into the original parameter storage, not a copy,
    so its weights remain those learned during pretraining.

    Parameters
    ----------
    model : torch.nn.Module
        The full monolithic model returned by ``load_model``.
    layer_index : int
        Zero-based index of the decoder block to extract.

    Returns
    -------
    torch.nn.Module
        The isolated decoder block, in ``torch.float16``, ready for handoff
        to Workstation 2 (INT4 Quantization).

    Raises
    ------
    RuntimeError
        If the decoder block list cannot be located in the model's module
        tree (unsupported/unrecognized architecture).
    IndexError
        If ``layer_index`` is out of range for the resolved decoder stack.
    """
    print(f"[Workstation-1] Isolating decoder block at index {layer_index} ...")

    decoder_layers: torch.nn.ModuleList | None = None

    # Llama / Mistral / Qwen: model.model.layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        decoder_layers = model.model.layers
    # GPT-NeoX / Falcon: model.transformer.h
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        decoder_layers = model.transformer.h

    if decoder_layers is None:
        raise RuntimeError(
            f"[Workstation-1 FAIL] Could not locate the decoder block list "
            f"in model of type {type(model).__qualname__}. Expected "
            f"'model.model.layers' or 'model.transformer.h'."
        )

    # Native list indexing -- raises IndexError on out-of-range access with
    # no custom contract-checking logic involved.
    isolated_layer: torch.nn.Module = decoder_layers[layer_index]
    isolated_layer = isolated_layer.to(torch.float16)

    param_count: int = sum(p.numel() for p in isolated_layer.parameters())
    print(
        f"[Workstation-1] Layer {layer_index} isolated. "
        f"Parameters: {param_count:,} | Dtype: torch.float16"
    )

    return isolated_layer