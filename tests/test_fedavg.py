import numpy as np

from fractal_quant.codec import cosine_similarity, quantize_state_dict
from fractal_quant.fedavg import (
    coordinate_median_aggregate,
    federated_average,
    fedprox_proximal_step,
    krum_aggregate,
)


def test_federated_average_frqd_packets():
    w1 = {"proj": np.full((64, 64), 1.0, dtype=np.float32)}
    w2 = {"proj": np.full((64, 64), 3.0, dtype=np.float32)}
    p1, _ = quantize_state_dict(w1)
    p2, _ = quantize_state_dict(w2)

    merged = federated_average([p1, p2], weights=[1.0, 3.0])
    expected = np.full((64, 64), 2.5, dtype=np.float32)
    assert np.allclose(merged["proj"], expected, atol=0.1)


def test_byzantine_median_and_krum_reject_poison():
    honest = [
        {"w": np.full((64, 64), 0.1 * i, dtype=np.float32)} for i in (1, 2, 1, 2)
    ]
    poisoned = [{"w": np.full((64, 64), 5000.0, dtype=np.float32)}]
    all_updates = honest + poisoned

    med = coordinate_median_aggregate(all_updates)
    krum = krum_aggregate(all_updates, num_byzantine=1)

    target = np.full((64, 64), 0.15, dtype=np.float32)
    assert float(np.max(np.abs(med["w"]))) < 1.0
    assert cosine_similarity(target, krum["w"]) > 0.99


def test_fedprox_proximal_step_shrinks_drift():
    local_w = {"w": np.full((16,), 2.0, dtype=np.float32)}
    global_w = {"w": np.zeros((16,), dtype=np.float32)}
    grads = {"w": np.zeros((16,), dtype=np.float32)}
    updated = fedprox_proximal_step(local_w, global_w, grads, lr=0.2, mu=0.5)
    assert np.allclose(updated["w"], np.full((16,), 1.8, dtype=np.float32))
