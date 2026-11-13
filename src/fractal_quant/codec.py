"""Micro-block INT4 and INT8 codecs with deterministic binary framing.

Wire format FRQ4 (little-endian):
  4B  magic ("FRQ4")
  1B  version (1)
  1B  flags (bit 0 = stochastic rounding, bit 1 = error feedback)
  2B  block_size (uint16)
  1B  ndim (uint8)
  4*ndim B  shape dimensions (uint32 each)
  4B  num_blocks (uint32)
  4*num_blocks B  per-block float32 scales
  4B  packed_len (uint32)
  packed_len B  sub-byte nibble data (low nibble even index, high nibble odd)
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

FRQ4_MAGIC = b"FRQ4"
FRQD_MAGIC = b"FRQD"
FORMAT_VERSION = 1
FLAG_STOCHASTIC = 0x01
FLAG_ERROR_FEEDBACK = 0x02
ACTIVATION_FRAME_LIMIT_BYTES = 4100


@dataclass(frozen=True)
class CompressionStats:
    """Summary telemetry for a quantized tensor or checkpoint transfer."""

    original_bytes: int
    compressed_bytes: int
    num_elements: int
    cosine_similarity: float
    mse: float

    @property
    def compression_ratio(self) -> float:
        """Return FP32-equivalent compression ratio."""
        if self.compressed_bytes <= 0:
            return 0.0
        return self.original_bytes / self.compressed_bytes

    @property
    def bandwidth_reduction_pct(self) -> float:
        """Return percentage reduction in wire payload."""
        if self.original_bytes <= 0:
            return 0.0
        return (1.0 - (self.compressed_bytes / self.original_bytes)) * 100.0


def _to_numpy(array_or_tensor) -> np.ndarray:
    """Convert NumPy arrays or PyTorch tensors cleanly to contiguous float32."""
    if hasattr(array_or_tensor, "detach"):
        arr = array_or_tensor.detach().cpu().numpy()
    else:
        arr = np.asarray(array_or_tensor)
    return np.ascontiguousarray(arr, dtype=np.float32)


def cosine_similarity(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute vector cosine similarity between original and reconstructed tensors."""
    a = _to_numpy(original).ravel().astype(np.float64)
    b = _to_numpy(reconstructed).ravel().astype(np.float64)
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 and norm_b == 0.0:
        return 1.0
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def quantize_tensor_int4(
    tensor,
    block_size: int = 64,
    stochastic: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[bytes, np.ndarray]:
    """Quantize a tensor into FRQ4 wire bytes using per-block INT4 dynamic scaling.

    Returns a tuple of ``(packet_bytes, dequantized_float32_array)`` so callers
    can immediately compute local error-feedback residuals without a separate unpack pass.
    """
    if block_size <= 0 or block_size > 65535:
        raise ValueError(f"block_size must be in [1, 65535], got {block_size}")

    arr = _to_numpy(tensor)
    shape = arr.shape
    flat = arr.ravel()
    numel = flat.size

    if numel == 0:
        header = _encode_header(shape, block_size, stochastic=stochastic, error_feedback=False, num_blocks=0, packed_len=0)
        return header, arr.copy()

    num_blocks = math.ceil(numel / block_size)
    padded_len = num_blocks * block_size
    if padded_len != numel:
        padded = np.pad(flat, (0, padded_len - numel), mode="constant")
    else:
        padded = flat

    blocks = padded.reshape(num_blocks, block_size)
    max_abs = np.max(np.abs(blocks), axis=1, keepdims=True)
    scales = (max_abs / 7.0).astype(np.float32)
    safe_scales = np.where(scales == 0.0, 1.0, scales)
    scaled = np.where(scales == 0.0, 0.0, blocks / safe_scales)

    if stochastic:
        generator = rng if rng is not None else np.random.default_rng()
        noise = generator.uniform(-0.5, 0.5, size=scaled.shape).astype(np.float32)
        rounded = np.floor(scaled + 0.5 + noise)
    else:
        rounded = np.round(scaled)

    q_signed = np.clip(rounded, -8.0, 7.0).astype(np.int8)
    dequant_blocks = (q_signed.astype(np.float32) * scales).ravel()[:numel]
    reconstructed = dequant_blocks.reshape(shape)

    q_flat = q_signed.ravel()[:numel]
    packed_bytes = _pack_nibbles(q_flat)
    scales_le = np.asarray(scales.ravel(), dtype="<f4").tobytes()

    header = _encode_header(
        shape=shape,
        block_size=block_size,
        stochastic=stochastic,
        error_feedback=False,
        num_blocks=num_blocks,
        packed_len=len(packed_bytes),
    )
    return header + scales_le + packed_bytes, reconstructed


def _pack_nibbles(q_signed: np.ndarray) -> bytes:
    """Pack signed INT4 integers in [-8, 7] as two 4-bit unsigned nibbles per uint8."""
    u = (q_signed.astype(np.int16) + 8).astype(np.uint8)
    if u.size % 2 != 0:
        u = np.append(u, np.uint8(8))
    even = u[0::2] & 0x0F
    odd = (u[1::2] & 0x0F) << 4
    return np.bitwise_or(even, odd).astype(np.uint8).tobytes()


def _unpack_nibbles(packed: bytes, numel: int) -> np.ndarray:
    """Unpack sub-byte nibbles back into signed int8 values in [-8, 7]."""
    raw = np.frombuffer(packed, dtype=np.uint8)
    unpacked = np.empty(raw.size * 2, dtype=np.int16)
    unpacked[0::2] = raw & 0x0F
    unpacked[1::2] = (raw >> 4) & 0x0F
    return (unpacked[:numel] - 8).astype(np.int8)


def _encode_header(
    shape: Sequence[int],
    block_size: int,
    stochastic: bool,
    error_feedback: bool,
    num_blocks: int,
    packed_len: int,
) -> bytes:
    flags = (FLAG_STOCHASTIC if stochastic else 0) | (FLAG_ERROR_FEEDBACK if error_feedback else 0)
    ndim = len(shape)
    parts = [
        struct.pack("<4sBBHB", FRQ4_MAGIC, FORMAT_VERSION, flags, block_size, ndim),
        struct.pack(f"<{ndim}I", *shape) if ndim > 0 else b"",
        struct.pack("<I", num_blocks),
    ]
    if num_blocks == 0:
        parts.append(struct.pack("<I", packed_len))
    return b"".join(parts)


def unpack_int4_packet(packet: bytes) -> Tuple[np.ndarray, dict]:
    """Decode an FRQ4 wire packet back into a float32 NumPy array and header metadata."""
    if len(packet) < 13 or packet[:4] != FRQ4_MAGIC:
        raise ValueError("Invalid FRQ4 packet header")

    magic, version, flags, block_size, ndim = struct.unpack_from("<4sBBHB", packet, 0)
    if magic != FRQ4_MAGIC or version != FORMAT_VERSION:
        raise ValueError(f"Unsupported FRQ4 version: {version}")

    offset = 9
    if ndim > 0:
        shape = struct.unpack_from(f"<{ndim}I", packet, offset)
        offset += 4 * ndim
    else:
        shape = ()

    (num_blocks,) = struct.unpack_from("<I", packet, offset)
    offset += 4

    numel = int(math.prod(shape)) if shape else 1
    if ndim == 0 and num_blocks == 0:
        numel = 0
    if numel == 0 or num_blocks == 0:
        return np.zeros(shape, dtype=np.float32), {
            "block_size": block_size,
            "flags": flags,
            "shape": shape,
        }

    scales_size = num_blocks * 4
    scales = np.frombuffer(packet, dtype="<f4", count=num_blocks, offset=offset).astype(np.float32)
    offset += scales_size

    packed_len = math.ceil(numel / 2)
    packed = packet[offset : offset + packed_len]
    if len(packed) != packed_len:
        raise ValueError(f"Truncated FRQ4 payload: expected {packed_len} bytes, got {len(packed)}")

    q_signed = _unpack_nibbles(packed, numel)
    padded_len = num_blocks * block_size
    if q_signed.size != padded_len:
        q_padded = np.pad(q_signed, (0, padded_len - numel), mode="constant")
    else:
        q_padded = q_signed

    blocks = q_padded.reshape(num_blocks, block_size).astype(np.float32)
    reconstructed = (blocks * scales[:, None]).ravel()[:numel].reshape(shape)
    return np.ascontiguousarray(reconstructed, dtype=np.float32), {
        "block_size": block_size,
        "flags": flags,
        "shape": shape,
        "num_blocks": num_blocks,
    }


def dequantize_tensor_int4(packet: bytes) -> np.ndarray:
    """Convenience wrapper returning only the reconstructed float32 array."""
    arr, _ = unpack_int4_packet(packet)
    return arr


def pack_state_dict(packets: Mapping[str, bytes]) -> bytes:
    """Serialize a dictionary of named FRQ4 packets into a single FRQD binary bundle."""
    chunks = [struct.pack("<4sI", FRQD_MAGIC, len(packets))]
    for name in sorted(packets.keys()):
        name_bytes = name.encode("utf-8")
        payload = packets[name]
        chunks.append(struct.pack("<H", len(name_bytes)))
        chunks.append(name_bytes)
        chunks.append(struct.pack("<I", len(payload)))
        chunks.append(payload)
    return b"".join(chunks)


def unpack_state_dict(bundle: bytes) -> Dict[str, np.ndarray]:
    """Deserialize an FRQD binary bundle into a mapping of parameter name to float32 array."""
    if len(bundle) < 8 or bundle[:4] != FRQD_MAGIC:
        raise ValueError("Invalid FRQD state bundle magic header")
    _, count = struct.unpack_from("<4sI", bundle, 0)
    offset = 8
    restored: Dict[str, np.ndarray] = {}
    for _ in range(count):
        (name_len,) = struct.unpack_from("<H", bundle, offset)
        offset += 2
        name = bundle[offset : offset + name_len].decode("utf-8")
        offset += name_len
        (pkt_len,) = struct.unpack_from("<I", bundle, offset)
        offset += 4
        pkt = bundle[offset : offset + pkt_len]
        offset += pkt_len
        restored[name] = dequantize_tensor_int4(pkt)
    return restored


def quantize_activation_int8(tensor, max_frame_bytes: int = ACTIVATION_FRAME_LIMIT_BYTES) -> bytes:
    """Encode an inter-node activation tensor as an INT8 frame strictly <= 4,100 bytes.

    Frame layout (little-endian): [4B float32 scale] + [N int8 quantized elements].
    For a 4096-dim hidden state, total length is exactly 4 + 4096 = 4100 bytes.
    """
    arr = _to_numpy(tensor).ravel()
    expected_bytes = 4 + arr.size
    if expected_bytes > max_frame_bytes:
        raise ValueError(
            f"Activation frame size {expected_bytes} bytes exceeds strict contract limit "
            f"of {max_frame_bytes} bytes"
        )
    if arr.size == 0:
        return struct.pack("<f", 0.0)
    max_abs = float(np.max(np.abs(arr)))
    scale = max_abs / 127.0 if max_abs > 0.0 else 0.0
    if scale == 0.0:
        q = np.zeros(arr.size, dtype=np.int8)
    else:
        q = np.clip(np.round(arr / scale), -127.0, 127.0).astype(np.int8)
    return struct.pack("<f", scale) + q.tobytes()


def dequantize_activation_int8(frame: bytes, shape: Tuple[int, ...]) -> np.ndarray:
    """Decode an INT8 activation frame back into the target tensor shape."""
    if len(frame) < 4:
        raise ValueError("Truncated INT8 activation frame")
    (scale,) = struct.unpack_from("<f", frame, 0)
    q = np.frombuffer(frame, dtype=np.int8, offset=4)
    return (q.astype(np.float32) * scale).reshape(shape)


class Int4BlockQuantizer:
    """Stateful micro-block INT4 compressor with optional error-feedback residuals."""

    def __init__(
        self,
        block_size: int = 64,
        stochastic: bool = False,
        error_feedback: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        self.block_size = block_size
        self.stochastic = stochastic
        self.error_feedback = error_feedback
        self._rng = np.random.default_rng(seed) if stochastic else None
        self._residuals: Dict[str, np.ndarray] = {}

    def reset_residuals(self) -> None:
        """Clear accumulated quantization error buffers."""
        self._residuals.clear()

    def compress(self, name: str, tensor) -> Tuple[bytes, CompressionStats]:
        """Compress a tensor and optionally accumulate quantization residual error."""
        arr = _to_numpy(tensor)
        effective = arr
        if self.error_feedback and name in self._residuals:
            effective = arr + self._residuals[name]

        pkt, reconstructed = quantize_tensor_int4(
            effective,
            block_size=self.block_size,
            stochastic=self.stochastic,
            rng=self._rng,
        )

        if self.error_feedback:
            self._residuals[name] = (effective - reconstructed).astype(np.float32)

        stats = CompressionStats(
            original_bytes=max(arr.nbytes, 1),
            compressed_bytes=len(pkt),
            num_elements=arr.size,
            cosine_similarity=cosine_similarity(effective, reconstructed),
            mse=float(np.mean((effective - reconstructed) ** 2)) if arr.size > 0 else 0.0,
        )
        return pkt, stats

    def compress_state_dict(
        self, state_dict: Mapping[str, object]
    ) -> Tuple[bytes, CompressionStats]:
        """Compress an entire named weight dictionary into an FRQD wire bundle."""
        packets: Dict[str, bytes] = {}
        total_orig = 0
        orig_parts = []
        rec_parts = []

        for key, val in state_dict.items():
            arr = _to_numpy(val)
            pkt, _ = self.compress(key, arr)
            packets[key] = pkt
            total_orig += arr.nbytes
            orig_parts.append(arr.ravel())
            rec_parts.append(dequantize_tensor_int4(pkt).ravel())

        bundle = pack_state_dict(packets)
        if orig_parts:
            all_orig = np.concatenate(orig_parts)
            all_rec = np.concatenate(rec_parts)
            cos = cosine_similarity(all_orig, all_rec)
            mse = float(np.mean((all_orig - all_rec) ** 2))
            numel = all_orig.size
        else:
            cos, mse, numel = 1.0, 0.0, 0

        return bundle, CompressionStats(
            original_bytes=total_orig,
            compressed_bytes=len(bundle),
            num_elements=numel,
            cosine_similarity=cos,
            mse=mse,
        )


def quantize_state_dict(
    state_dict: Mapping[str, object],
    block_size: int = 64,
    stochastic: bool = False,
) -> Tuple[bytes, CompressionStats]:
    """Stateless helper to compress a state dictionary into an FRQD bundle."""
    return Int4BlockQuantizer(
        block_size=block_size,
        stochastic=stochastic,
        error_feedback=False,
    ).compress_state_dict(state_dict)


def dequantize_state_dict(bundle: bytes) -> Dict[str, np.ndarray]:
    """Decode an FRQD bundle back into float32 NumPy weight arrays."""
    return unpack_state_dict(bundle)
