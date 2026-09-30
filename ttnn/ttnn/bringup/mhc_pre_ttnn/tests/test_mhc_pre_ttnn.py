# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_pre against its torch semantics (reference.py), one random-input case per captured model call
(cases.py). Math: PCC plus a max relative error (rel L2) per output, per device. The inputs differ per device (sharded
over the mesh on dim 0), so each chip's output is checked on its own data; the weights are replicated."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_mhc_pre_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES
_DTYPE = {"BFLOAT16": (ttnn.bfloat16, torch.bfloat16), "FLOAT32": (ttnn.float32, torch.float32)}


@pytest.fixture(scope="module")
def mesh():
    c = CASES[0]
    p = c["device_params"]
    ttnn.set_fabric_config(getattr(ttnn.FabricConfig, p["fabric_config"]))
    m = ttnn.open_mesh_device(ttnn.MeshShape(*c["mesh"]), l1_small_size=p["l1_small_size"])
    yield m
    ttnn.close_mesh_device(m)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _devices(mesh):
    return mesh.get_num_devices()


def _host(spec, g, per_device, n_dev, scale=1.0):
    """per_device: a different random tensor per chip, stacked on dim 0 (sharded); else one (replicated)."""
    shape = list(spec["shape"])
    if per_device:
        shape = [shape[0] * n_dev] + shape[1:]
    return (torch.randn(shape, generator=g) * scale).to(_DTYPE[spec["dtype"]][1])


def _to_device(t, spec, mesh, per_device):
    mapper = ttnn.ShardTensorToMesh(mesh, dim=0) if per_device else ttnn.ReplicateTensorToMesh(mesh)
    return ttnn.from_torch(
        t,
        dtype=_DTYPE[spec["dtype"]][0],
        layout=ttnn.TILE_LAYOUT,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=mapper,
    )


def _from_device(t, mesh):
    return ttnn.to_torch(t, mesh_composer=ttnn.ConcatMeshToTensor(mesh, dim=0))


def _check(name, got, want, pcc, max_rel):
    got, want = got.double().reshape(want.shape), want.double()
    p = torch.corrcoef(torch.stack([got.flatten(), want.flatten()]))[0, 1].item()
    rel = ((got - want).norm() / want.norm()).item()
    assert p >= pcc and rel <= max_rel, f"{name}: pcc {p:.7f} (>= {pcc}), rel L2 {rel:.6f} (<= {max_rel})"
    return p, rel


@pytest.mark.parametrize("case", CASES, ids=[c["id"] for c in CASES])
def test_mhc_pre(mesh, case):
    g = torch.Generator().manual_seed(case["seed"])
    n_dev = _devices(mesh)
    nc = case["input"]["shape"][-1]
    x = _host(case["input"], g, True, n_dev)
    w = _host(case["proj_weight"], g, False, n_dev, scale=nc**-0.5)
    b = _host(case["proj_bias"], g, False, n_dev, scale=0.5)
    out = ttnn.bringup.mhc_pre(
        _to_device(x, case["input"], mesh, True),
        _to_device(w, case["proj_weight"], mesh, False),
        _to_device(b, case["proj_bias"], mesh, False),
        scale=tuple(case["scale"]),
        sinkhorn_iters=case["sinkhorn_iters"],
        eps=case["eps"],
        norm_eps=case["norm_eps"],
    )
    want = ref.mhc_pre(x, w, b, case["scale"], case["sinkhorn_iters"], case["eps"], case["norm_eps"])
    for name, t, e in zip(("y", "post", "comb"), out, want):
        p, r = _check(name, _from_device(t, mesh), e, case["pcc"], case["max_rel"][name])
        print(f"{case['id']} {name}: pcc {p:.7f} rel {r:.6f}")
