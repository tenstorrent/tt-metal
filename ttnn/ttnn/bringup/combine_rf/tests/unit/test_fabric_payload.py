# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""sparse_combine_rf's fabric packets vs the mesh's fabric payload (deepseek_v41_flash_d_p K.1, 2026-10-27).

Every row piece goes as one fabric packet of piece + 32 B (its trailer). DeepSeek-V4.1's 10240 B rows (H 5120, one
piece: the L1 hand-off) need a 10272 B packet; a mesh opened without the model's fabric_router_config (router default
4352 B) hung the weighted combine on every chip (the flat expert's compute stalled behind full rings). The prepare
now refuses such a mesh with the payload it needs.

- test_payload_refused (mesh opened with the router default payload): refused, the message names the 10272 B packet.
- test_payload_fits (mesh opened with the model's 10880 B payload): the same prepare builds.

    BRINGUP_TEST_MESH=8x4 scripts/run_safe_pytest.sh --run-all ttnn/ttnn/bringup/combine_rf/tests/unit/test_fabric_payload.py
"""

import os

import pytest

import ttnn
from ttnn.bringup.combine_rf.sparse_combine_rf import sparse_combine_rf_prepare

_MESH = tuple(int(v) for v in os.environ.get("BRINGUP_TEST_MESH", "2x2").split("x"))
W = 5120  # DeepSeek-V4.1 hidden: 10240 B bf16 rows
PAYLOAD = 10880  # the model's (deepseek_v41 tt/weighted_combine.py FABRIC_PAYLOAD)


def _params(payload):
    p = {"fabric_config": ttnn.FabricConfig.FABRIC_2D, "l1_small_size": 24576}
    if payload:
        rc = ttnn._ttnn.fabric.FabricRouterConfig()
        rc.max_packet_payload_size_bytes = payload
        p["fabric_router_config"] = rc
    return p


def _prepare(mesh):
    R, C = tuple(mesh.shape)
    chips = [(r, c) for r in range(R) for c in range(C)]
    # (the model's column combine, one piece a row as with its L1 hand-off; no rows: only the cores are placed)
    return sparse_combine_rf_prepare(mesh, 64, W, {ch: [] for ch in chips}, num_links=1, piece_max=2 * W, axis=0)


def _mesh_param(payload, name):
    return pytest.mark.parametrize(
        "mesh_device, device_params",
        [(_MESH, _params(payload))],
        ids=["x".join(map(str, _MESH)) + "-" + name],
        indirect=True,
    )


@_mesh_param(None, "default_payload")
def test_payload_refused(mesh_device, device_params, expect_error):
    fab = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    assert fab < 2 * W + 32, f"the router default payload {fab} B already fits a {2 * W} B row"
    with expect_error(AssertionError, f"{2 * W + 32} B fabric packets"):
        _prepare(mesh_device)


@_mesh_param(PAYLOAD, "payload_10880")
def test_payload_fits(mesh_device, device_params):
    assert int(ttnn.get_tt_fabric_max_payload_size_bytes()) >= 2 * W + 32
    run = _prepare(mesh_device)
    assert run.cells["accs"], run.cells
