import numpy as np
import pytest

from fractal_quant.codec import (
    ACTIVATION_FRAME_LIMIT_BYTES,
    FRQD_MAGIC,
    FRQ4_MAGIC,
    Int4BlockQuantizer,
    cosine_similarity,
    dequantize_activation_int8,
    dequantize_state_dict,
    dequantize_tensor_int4,
    quantize_activation_int8,
    quantize_state_dict,
    quantize_tensor_int4,
    unpack_int4_packet,
    unpack_state_dict,
)


def _global_int4(arr: np.ndarray) -> np.ndarray:
    max_abs = float(np.max(np.abs(arr)))
    scale = max(max_abs / 7.0, 1e-8)
    q = np.clip(np.round(arr / scale), -8, 7)
    return (q * scale).astype(np.float32)


def test_frq4_roundtrip_outlier_resilience():
    rng = np.random.default_rng(7)
    weights = (rng.normal(0.0, 0.02, size=(512, 256))).astype(np.float32)
    weights[0, 0] = 0.85

    q = Int4BlockQuantizer(block_size=64, error_feedback=False)
    packet, stats = q.compress("attn.q_proj", weights)
    assert packet.startswith(FRQ4_MAGIC)
    assert stats.compression_ratio >= 6.8

    recovered = dequantize_tensor_int4(packet)
    assert recovered.shape == weights.shape

    cos_fractal = cosine_similarity(weights, recovered)
    cos_global = cosine_similarity(weights, _global_int4(weights))
    assert cos_fractal >= 0.985
    assert cos_fractal > cos_global + 0.15

    direct_pkt, direct_rec = quantize_tensor_int4(weights, block_size=64)
    assert direct_pkt == packet
    assert np.allclose(direct_rec, recovered)


def test_frqd_state_dict_pack_and_unpack():
    rng = np.random.default_rng(13)
    state = {
        "layer0.weight": rng.normal(0.0, 0.05, size=(128, 128)).astype(np.float32),
        "layer0.bias": rng.normal(0.0, 0.05, size=(128,)).astype(np.float32),
    }
    blob, stats = quantize_state_dict(state, block_size=64)
    assert blob.startswith(FRQD_MAGIC)
    assert stats.compression_ratio > 6.5

    restored = dequantize_state_dict(blob)
    assert set(restored.keys()) == set(state.keys())
    for k, arr in state.items():
        assert restored[k].shape == arr.shape
        assert cosine_similarity(arr, restored[k]) > 0.98


def test_error_feedback_bounds_multi_round_drift():
    rng = np.random.default_rng(99)
    delta = (rng.normal(0.0, 0.01, size=(256, 64))).astype(np.float32)
    q = Int4BlockQuantizer(block_size=64, error_feedback=True)

    accum_true = np.zeros_like(delta)
    accum_quant = np.zeros_like(delta)
    for _ in range(8):
        accum_true += delta
        packet, _ = q.compress("w", delta)
        accum_quant += dequantize_tensor_int4(packet)

    assert cosine_similarity(accum_true, accum_quant) >= 0.998


def test_int8_activation_frame_size_limit():
    rng = np.random.default_rng(42)
    acts = rng.normal(0.0, 1.0, size=(4096,)).astype(np.float32)
    frame = quantize_activation_int8(acts)
    assert len(frame) == ACTIVATION_FRAME_LIMIT_BYTES
    assert len(frame) <= 4100

    restored = dequantize_activation_int8(frame, acts.shape)
    assert restored.shape == acts.shape
    assert cosine_similarity(acts, restored) > 0.999


def test_invalid_packet_headers_raise():
    with pytest.raises(ValueError):
        unpack_int4_packet(b"BAD!" + b"\x00" * 32)
    with pytest.raises(ValueError):
        unpack_state_dict(b"BAD!" + b"\x00" * 32)
