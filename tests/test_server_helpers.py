from fractal_server import server


def test_calc_reward_normal():
    assert server._calc_reward(150.0, 10) == 15.0


def test_calc_reward_zero_clients():
    assert server._calc_reward(150.0, 0) == 0.0


def test_tflops_calculation():
    cfg = {
        "N_BINS": 10,
        "ITEMS_PER_BIN": 100,
        "NUM_EPOCHS": 5,
        "MAX_ROUNDS": 2,
    }
    task_tflops = server._tflops_per_task(cfg)
    assert task_tflops > 0.0
    assert isinstance(task_tflops, float)

    session_tflops = server._tflops_full_session(cfg)
    assert session_tflops == task_tflops * cfg["N_BINS"] * cfg["MAX_ROUNDS"]


def test_id_registry_formatting():
    reg = server.IDRegistry()
    seg_id, seg_seq = reg.generateDataSegmentId("1001")
    task_id, _ = reg.generateTaskId("0001", "1001", seg_seq)
    assert seg_id == "10010000001"
    assert len(task_id) == 22
    assert server.IDRegistry().readable(task_id) == f"0001_1001_{seg_seq}_0000001"


def test_safe_join_blocks_traversal(tmp_path):
    base = str(tmp_path)
    assert server._safe_join(base, "batch_001.bin") is not None
    for bad in ("../secrets/tenants.json", "..\\tokens.json", "/etc/passwd", "a/b.bin", ""):
        assert server._safe_join(base, bad) is None
