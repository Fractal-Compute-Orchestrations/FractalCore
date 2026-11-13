import io
import os

import numpy as np
import pytest

from fractal_quant.codec import cosine_similarity, quantize_state_dict, unpack_state_dict
from fractal_server import server


@pytest.fixture
def isolated_server(tmp_path, monkeypatch):
    monkeypatch.setattr(server, "DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(server, "SECRETS_DIR", str(tmp_path / "secrets"))
    monkeypatch.setattr(
        server, "TENANTS_JSON", str(tmp_path / "secrets" / "tenants.json")
    )
    monkeypatch.setattr(
        server, "tokens_json", str(tmp_path / "secrets" / "tokens.json")
    )
    monkeypatch.setattr(server, "SHARED_MODEL_DIR", str(tmp_path / "downloads"))
    monkeypatch.setattr(server, "ALLOW_ANON_DEMO", False)
    monkeypatch.setattr(server, "SEED_DEMO_TENANTS", False)
    os.makedirs(server.SHARED_MODEL_DIR, exist_ok=True)
    server._tenants.clear()
    server._token_store.clear()
    server._global_task_tenant_map.clear()
    server._task_download_map.clear()
    server._initialized = True
    yield server.app.test_client()
    server._tenants.clear()
    server._token_store.clear()


def test_health_and_auth_enforcement(isolated_server):
    res = isolated_server.get("/api/health")
    assert res.status_code == 200
    assert res.get_json()["frqd_enabled"] is True

    # Anonymous calls to protected API routes must receive 401 instead of admin access
    assert isolated_server.get("/api/admin/tenants").status_code == 401
    assert isolated_server.get("/api/tenant/status").status_code == 401

    # Valid admin token works, tenant token is rejected from admin routes (403)
    admin_tok = server._issue_token(server.ADMIN_USERNAME, "admin")
    ok, _ = server.create_tenant("uni_lab", "pw123", 1.0, 100.0, active=False)
    assert ok
    tenant_tok = server._issue_token("uni_lab", "tenant")

    assert (
        isolated_server.get(
            "/api/admin/tenants", headers={"X-Auth-Token": admin_tok}
        ).status_code
        == 200
    )
    assert (
        isolated_server.get(
            "/api/admin/tenants", headers={"X-Auth-Token": tenant_tok}
        ).status_code
        == 403
    )
    assert (
        isolated_server.get(
            "/api/tenant/status", headers={"X-Auth-Token": tenant_tok}
        ).status_code
        == 200
    )


def test_download_blocks_path_traversal(isolated_server):
    res = isolated_server.get("/download/model?filename=../../secrets/tenants.json")
    assert res.status_code == 400

    res2 = isolated_server.get(
        "/download/images?tenant_id=missing&filename=..\\tenants.json"
    )
    assert res2.status_code == 400


def test_frqd_upload_and_federated_averaging_round(isolated_server):
    ok, _ = server.create_tenant("edge_lab", "pw", 5.0, 200.0, active=True)
    assert ok
    sess = server._tenants["edge_lab"]["session"]
    sess.config["MAX_CLIENTS"] = 2

    # Register two task IDs for deterministic upload mapping
    for idx, dev in enumerate(("dev_a", "dev_b"), start=1):
        tid = f"{sess.model_id}{sess.data_id}0000001000000{idx}"
        with server._global_task_tenant_lock:
            server._global_task_tenant_map[tid] = "edge_lab"
        w = {"layer.weight": np.full((64, 64), float(idx * 2), dtype=np.float32)}
        payload, _ = quantize_state_dict(w, block_size=64)
        resp = isolated_server.post(
            "/api/model/upload",
            data={
                "task_Id": tid,
                "device_id": dev,
                "model_file": (io.BytesIO(payload), f"client_{idx}.frqd"),
            },
            content_type="multipart/form-data",
        )
        assert resp.status_code == 200

    # Directly verify aggregation writes global.frqd cleanly without TensorFlow
    sess.do_federated_averaging()
    global_frqd = sess.global_ckpt_path.replace(".ckpt", ".frqd")
    assert os.path.exists(global_frqd)
    with open(global_frqd, "rb") as fh:
        merged = unpack_state_dict(fh.read())

    expected = np.full((64, 64), 3.0, dtype=np.float32)
    assert cosine_similarity(expected, merged["layer.weight"]) > 0.99
    assert sess.state["aggregation_backend"] == "fractal_quant.frqd"
    assert sess.state["compression_ratio"] > 4.0
