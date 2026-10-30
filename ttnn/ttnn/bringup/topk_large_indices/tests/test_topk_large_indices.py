# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.topk_large_indices against its torch semantics (reference.py), one random-input case per captured model
call (cases.py). Every chip gets its own rows (sharded over the mesh on dim 0) and is checked on them: the values the
ids point at equal the k largest of the valid columns, position by position (ties may take any tied id), ids distinct
and below valid_length, sentinels exactly where a row has fewer finite values than k. Determinism (section 4b of the
skill): 5 runs on A with one on B between, bit-identical."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_topk_large_indices_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    p["require_exact_physical_num_devices"] = True
    return p


def _inputs(mesh_device, c, seed):
    """Host scores [n_dev, 1, R, W] bf16 (device d: block d; columns from neginf_from are -inf) and the device tensor."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    _, _, R, W = c["input"]["shape"]
    x = torch.randn(n_dev, 1, R, W, generator=torch.Generator().manual_seed(seed)).bfloat16()
    if c.get("neginf_from") is not None:
        x[..., c["neginf_from"] :] = float("-inf")
    t = ttnn.from_torch(
        x,
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    return x, t


def _call(c, t):
    kw = {} if c["valid_length"] is None else {"valid_length": c["valid_length"]}
    return ttnn.bringup.topk_large_indices(t, k=c["k"], **kw)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_topk_large_indices(mesh_device, device_params, case):
    c = case
    x, t = _inputs(mesh_device, c, c["seed"])
    out = _call(c, t)
    assert out.dtype == ttnn.uint32, out.dtype
    _, _, R, W = c["input"]["shape"]
    n = W if c["valid_length"] is None else c["valid_length"]
    for d, dt in enumerate(ttnn.get_device_tensors(out)):
        ids = (ttnn.to_torch(dt).to(torch.int64) & 0xFFFFFFFF).reshape(R, c["k"])
        err = ref.check(x[d, 0], ids, c["k"], n)
        assert err is None, f"dev {d}: {err}"
    t_b = _inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(lambda: _call(c, t), lambda: _call(c, t_b), first=out, label=c["id"])
