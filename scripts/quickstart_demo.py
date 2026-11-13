"""FRACTAL 30-second quickstart demonstration.

Runs two reproducible demonstrations with zero external cloud services:
1. Micro-block INT4 compression (FRQ4) on outlier-heavy LLM weights vs Global INT4.
2. End-to-end 3-client FRQD Federated Averaging round with Byzantine Krum check.
"""

from __future__ import annotations

import numpy as np

from fractal_quant import (
    ACTIVATION_FRAME_LIMIT_BYTES,
    Int4BlockQuantizer,
    cosine_similarity,
    dequantize_tensor_int4,
    federated_average,
    krum_aggregate,
    quantize_activation_int8,
    quantize_state_dict,
    quantize_tensor_int4,
)


def main() -> None:
    rng = np.random.default_rng(2026)

    # 1. Outlier-heavy transformer projection quantization
    weights = (rng.normal(0.0, 0.02, size=(1024, 256))).astype(np.float32)
    weights[0, :4] = 8.5

    max_abs = float(np.max(np.abs(weights)))
    global_scale = max(max_abs / 7.0, 1e-8)
    global_deq = (
        np.clip(np.round(weights / global_scale), -8, 7) * global_scale
    ).astype(np.float32)
    global_cos = cosine_similarity(weights, global_deq)

    packet, fractal_deq = quantize_tensor_int4(weights, block_size=64)
    fractal_cos = cosine_similarity(weights, fractal_deq)
    raw_bytes = weights.nbytes
    ratio = raw_bytes / len(packet)

    print("=== Part 1: Outlier-Resilient Micro-Block INT4 (FRQ4) ===")
    print(f"FP32 Baseline      : {raw_bytes:,} bytes")
    print(f"FRQ4 Compressed    : {len(packet):,} bytes ({ratio:.2f}x)")
    print(f"Global INT4 Cosine : {global_cos:.4f} (collapses around attention outliers)")
    print(f"FRQ4 Micro Cosine  : {fractal_cos:.4f} | Re-decoded Cosine : {cosine_similarity(weights, dequantize_tensor_int4(packet)):.4f}")

    # 2. INT8 activation frame transport
    activations = rng.normal(0.0, 1.0, size=(4096,)).astype(np.float32)
    act_frame = quantize_activation_int8(activations)
    print(f"Activation Frame   : {len(act_frame)} / {ACTIVATION_FRAME_LIMIT_BYTES} bytes limit")

    # 3. 3-client FRQD Federated Averaging + Byzantine check
    true_target = {"proj.weight": np.full((64, 64), 1.5, dtype=np.float32)}
    q = Int4BlockQuantizer(block_size=64, error_feedback=True)
    packets = []
    for scale in (1.4, 1.5, 1.6):
        client_sd = {"proj.weight": np.full((64, 64), scale, dtype=np.float32)}
        blob, _ = q.compress_state_dict(client_sd)
        packets.append(blob)

    merged = federated_average(packets)
    merged_cos = cosine_similarity(true_target["proj.weight"], merged["proj.weight"])

    poisoned, _ = quantize_state_dict({"proj.weight": np.full((64, 64), 999.0, dtype=np.float32)})
    robust = krum_aggregate(packets + [poisoned], num_byzantine=1)
    robust_cos = cosine_similarity(true_target["proj.weight"], robust["proj.weight"])

    print("\n=== Part 2: FRQD Federated Averaging & Byzantine Resilience ===")
    print(f"FedAvg (3x FRQD) Cosine : {merged_cos:.4f}")
    print(f"Krum (with 1 Byzantine) : {robust_cos:.4f}")


if __name__ == "__main__":
    main()
