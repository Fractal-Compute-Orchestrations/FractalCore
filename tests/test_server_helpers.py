def calc_reward(total_device_mbs: float, max_clients: int) -> float:
    return round(total_device_mbs / max_clients, 4) if max_clients > 0 else 0.0


def tflops_per_task(cfg: dict, flops_per_image: float = 12.6e6) -> float:
    return cfg["ITEMS_PER_BIN"] * cfg["NUM_EPOCHS"] * flops_per_image / 1e12


def tflops_full_session(cfg: dict, flops_per_image: float = 12.6e6) -> float:
    return (
        cfg["N_BINS"]
        * cfg["ITEMS_PER_BIN"]
        * cfg["NUM_EPOCHS"]
        * cfg["MAX_ROUNDS"]
        * flops_per_image
        / 1e12
    )


def test_calc_reward_normal():
    reward = calc_reward(150.0, 10)
    assert reward == 15.0


def test_calc_reward_zero_clients():
    reward = calc_reward(150.0, 0)
    assert reward == 0.0


def test_tflops_calculation():
    cfg = {
        "N_BINS": 10,
        "ITEMS_PER_BIN": 100,
        "NUM_EPOCHS": 5,
        "MAX_ROUNDS": 2,
    }
    task_tflops = tflops_per_task(cfg)
    assert task_tflops > 0.0
    assert isinstance(task_tflops, float)

    session_tflops = tflops_full_session(cfg)
    assert session_tflops == task_tflops * cfg["N_BINS"] * cfg["MAX_ROUNDS"]
