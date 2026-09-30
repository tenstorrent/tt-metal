# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.mhc_post against its torch semantics (reference.py), one random-input case per captured model call
(cases.py). Math: PCC plus a max relative error (rel L2) per output, per device. The inputs differ per device (sharded
over the mesh on dim 0), so each chip's output is checked on its own data; the weights are replicated."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_mhc_post_ttnn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES
_DTYPE = {"BFLOAT16": (ttnn.bfloat16, torch.bfloat16), "FLOAT32": (ttnn.float32, torch.float32)}


@pytest.fixture(scope="module")
def mesh():
    # Open the mesh of a case captured on a box of this size (a 2x2 submesh of a 4x2 box fails the 2D fabric
    # handshake); the per-chip math does not depend on the mesh shape.
    n_sys = ttnn.get_num_devices()
    c = next((c for c in CASES if c["mesh"][0] * c["mesh"][1] == n_sys), CASES[0])
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
def test_mhc_post(mesh, case):
    g = torch.Generator().manual_seed(case["seed"])
    n_dev = _devices(mesh)
    f = _host(case["input"], g, True, n_dev)
    x = _host(case["residual"], g, True, n_dev)
    post = (_host(case["post"], g, True, n_dev).float().sigmoid() * 2).to(torch.float32)  # post = 2 sigmoid(.)
    comb = _host(case["comb"], g, True, n_dev).float().abs().to(torch.float32) / 4  # a comb-like positive mix
    kw = {"comb_transposed": case["comb_transposed"]} if "comb_transposed" in case else {}
    out = ttnn.bringup.mhc_post(
        *(
            _to_device(t, case[k], mesh, True)
            for t, k in ((f, "input"), (x, "residual"), (post, "post"), (comb, "comb"))
        ),
        **kw,
    )
    want = ref.mhc_post(f, x, post, comb, **kw)
    p, r = _check("out", _from_device(out, mesh), want, case["pcc"], case["max_rel"])
    if kw.get("comb_transposed") is False:
        # the option must change the math: the default (comb^T) result is far from the comb-as-stored one
        _, r_t = _check("out_t", want, ref.mhc_post(f, x, post, comb), -1.0, float("inf"))
        assert r_t > 50 * max(r, 1e-6), f"comb vs comb^T references too close (rel {r_t})"
    print(f"{case['id']}: pcc {p:.7f} rel {r:.6f}")
