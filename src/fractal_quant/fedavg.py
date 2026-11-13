"""Deterministic and Byzantine-resilient federated aggregation operators."""

from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence

import numpy as np

from fractal_quant.codec import _to_numpy, unpack_state_dict


def _extract_state_dict(update) -> Dict[str, np.ndarray]:
    """Normalize either an FRQD wire bundle or a float tensor map into float32 arrays."""
    if isinstance(update, (bytes, bytearray)):
        return unpack_state_dict(bytes(update))
    if isinstance(update, Mapping):
        return {k: _to_numpy(v) for k, v in update.items()}
    raise TypeError(f"Unsupported client update type: {type(update).__name__}")


def federated_average(
    client_updates: Sequence[Mapping[str, object] | bytes],
    weights: Optional[Sequence[float]] = None,
) -> Dict[str, np.ndarray]:
    """Compute weighted FedAvg across dense or FRQD-compressed client state dicts."""
    if not client_updates:
        raise ValueError("client_updates must not be empty")

    decoded = [_extract_state_dict(u) for u in client_updates]
    keys = list(decoded[0].keys())
    if weights is None:
        raw_weights = [1.0] * len(decoded)
    else:
        if len(weights) != len(decoded):
            raise ValueError("weights length must match client_updates length")
        raw_weights = [float(w) for w in weights]

    total_w = sum(raw_weights)
    if total_w <= 0.0:
        raise ValueError("Sum of client aggregation weights must be positive")
    norm_weights = [w / total_w for w in raw_weights]

    aggregated: Dict[str, np.ndarray] = {}
    for key in keys:
        acc = np.zeros_like(decoded[0][key], dtype=np.float64)
        for upd, w in zip(decoded, norm_weights):
            if key not in upd:
                raise KeyError(f"Missing parameter '{key}' in client update")
            acc += w * upd[key].astype(np.float64)
        aggregated[key] = acc.astype(np.float32)
    return aggregated


def coordinate_median_aggregate(
    client_updates: Sequence[Mapping[str, object] | bytes],
) -> Dict[str, np.ndarray]:
    """Byzantine-resilient coordinate-wise median aggregation across clients."""
    if not client_updates:
        raise ValueError("client_updates must not be empty")
    decoded = [_extract_state_dict(u) for u in client_updates]
    keys = list(decoded[0].keys())
    aggregated: Dict[str, np.ndarray] = {}
    for key in keys:
        stacked = np.stack([d[key] for d in decoded], axis=0)
        aggregated[key] = np.median(stacked, axis=0).astype(np.float32)
    return aggregated


def krum_aggregate(
    client_updates: Sequence[Mapping[str, object] | bytes],
    num_byzantine: int = 1,
) -> Dict[str, np.ndarray]:
    """Select the client update with minimum L2 distance sum to nearest neighbors (Krum)."""
    n = len(client_updates)
    if n == 0:
        raise ValueError("client_updates must not be empty")
    if n <= 2 * num_byzantine + 2:
        # Multi-Krum / small fleet fallback to coordinate median when N <= 2f + 2
        return coordinate_median_aggregate(client_updates)

    decoded = [_extract_state_dict(u) for u in client_updates]
    keys = list(decoded[0].keys())
    vectors: List[np.ndarray] = [
        np.concatenate([d[k].ravel().astype(np.float64) for k in keys]) for d in decoded
    ]

    keep = n - num_byzantine - 2
    best_idx = 0
    best_score = float("inf")
    for i in range(n):
        dists = sorted(
            float(np.sum((vectors[i] - vectors[j]) ** 2)) for j in range(n) if i != j
        )
        score = sum(dists[:keep])
        if score < best_score:
            best_score = score
            best_idx = i
    return {k: decoded[best_idx][k].copy() for k in keys}


def fedprox_proximal_step(
    local_weights: Mapping[str, object],
    global_weights: Mapping[str, object],
    gradients: Mapping[str, object],
    lr: float = 0.01,
    mu: float = 0.01,
) -> Dict[str, np.ndarray]:
    """Apply one FedProx proximal update step: w <- w - lr * (grad + mu * (w - w_global))."""
    updated: Dict[str, np.ndarray] = {}
    for key in local_weights:
        w = _to_numpy(local_weights[key])
        w0 = _to_numpy(global_weights[key])
        g = _to_numpy(gradients[key])
        updated[key] = (w - lr * (g + mu * (w - w0))).astype(np.float32)
    return updated
