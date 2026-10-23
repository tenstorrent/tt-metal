# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""ttnn.bringup.moe_grouped_topk against its torch semantics (reference.py), one random-input case per captured model
call (cases.py). Every chip gets its own rows (sharded over the mesh on dim 0) and is checked on them.

- Selection (data selection on the fp32 keys): on every CLEAR row the set of selected experts equals the float64 one
  (exact match). A row is clear when its k-th / (k+1)-th key gap exceeds the keys' noise floor (fork_cases.noise_limit:
  calib x the fp32 keys' worst error vs float64 on this case's inputs); the rest are near ties either order may take,
  and the clear rows must be the bulk of the case (>= 99 %). The ids must be valid (k distinct experts < E).
- Weights (math): the dense [rows, E] weights of the clear rows vs float64 (the float64 selection), rel L2 within
  noise_limit at the declared formats (fp32 logits, bf16 out); PCC printed (a diagnostic).
- Determinism (section 4b of the skill): 5 runs on A with one on B between, bit-identical.

The op's own unit suite (tests/unit/, and the source op's carried tests, tests/source.yaml) checks the option against
the stock op chain."""

import importlib.util
from pathlib import Path

import pytest
import torch

import ttnn
from models.demos.common.bringup.testing import determinism
from models.demos.common.bringup.testing.fork_cases import noise_limit

_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"bringup_moe_grouped_topk_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ref = _load("reference")
CASES = _load("cases").CASES
CLEAR_SHARE = 0.99


def _device_params(c):
    p = dict(c["device_params"])
    p["fabric_config"] = getattr(ttnn.FabricConfig, p["fabric_config"])
    p["require_exact_physical_num_devices"] = True
    return p


def _inputs(mesh_device, c, seed):
    """Host logits / bias [n_dev, 1, R, E] (device d: block d) and their device tensors."""
    rows, cols = c["mesh"]
    n_dev = rows * cols
    g = torch.Generator().manual_seed(seed)
    _, _, R, E = c["scores"]["shape"]
    logits = torch.randn(n_dev, 1, R, E, generator=g) * c["logit_scale"] + c["logit_shift"]
    bias = (torch.randn(E, generator=g) * c["bias_scale"]).expand(n_dev, 1, R, E).contiguous()
    put = lambda t: ttnn.from_torch(  # noqa: E731
        t,
        dtype=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=mesh_device,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensorToMesh(mesh_device, dim=0),
    )
    return (logits, bias), (put(logits), put(bias))


def _call(c, dev):
    return ttnn.bringup.moe_grouped_topk(
        *dev,
        n_groups=c["n_groups"],
        summed_experts_per_group=c["summed_experts_per_group"],
        topk_groups=c["topk_groups"],
        n_activated_experts=c["n_activated_experts"],
        route_scale=c["route_scale"],
        epsilon=c["epsilon"],
        stable_sort=c["stable_sort"],
        score_func=c["score_func"],
        fp32_selection=c["fp32_selection"],
        fp32_activation=c["fp32_activation"],
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
    )


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "mesh_device, device_params, case",
    [(tuple(c["mesh"]), _device_params(c), c) for c in CASES],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_moe_grouped_topk(mesh_device, device_params, case):
    c = case
    assert c["n_groups"] == 1 and c["topk_groups"] == 1, "reference.py: the single-group path"
    k, sf = c["n_activated_experts"], c["score_func"]
    (logits, bias), dev = _inputs(mesh_device, c, c["seed"])
    out = _call(c, dev)
    w_d, i_d = out
    assert w_d.dtype == ttnn.bfloat16, w_d.dtype
    _, _, R, E = c["scores"]["shape"]
    for d, (wt, it) in enumerate(zip(ttnn.get_device_tensors(w_d), ttnn.get_device_tensors(i_d))):
        lg, bs = logits[d, 0], bias[d, 0]
        w = ttnn.to_torch(wt).double().reshape(R, -1)[:, :k]
        idx = ttnn.to_torch(it).to(torch.int64).reshape(R, -1)[:, :k]
        assert int(idx.min()) >= 0 and int(idx.max()) < E, "expert ids out of range"
        assert all(len(set(r.tolist())) == k for r in idx), "a row selects an expert twice"
        key64 = ref.keys(lg.double(), bs.double(), sf)
        idx64 = ref.select(key64, k)
        # the keys' noise floor: calib x the fp32 keys' worst error vs float64 (abs, the gap's unit)
        key32 = ref.keys(lg, bs, sf).double()
        lim_k = noise_limit(lambda a, b: ref.keys(a, b, sf), [lg, bs], *c["noise"]["keys"].values())
        tol = lim_k["rel"] / max(lim_k["noise_rel"], 1e-300) * float((key32 - key64).abs().max())
        clear = ref.gap(key64, k) > tol
        n_clear = int(clear.sum())
        got_sets = [frozenset(r.tolist()) for r in idx]
        want_sets = [frozenset(r.tolist()) for r in idx64]
        bad = [r for r in range(R) if clear[r] and got_sets[r] != want_sets[r]]
        near = [r for r in range(R) if not clear[r] and got_sets[r] != want_sets[r]]
        print(
            f"{c['id']} dev {d}: {n_clear}/{R} clear rows (gap > {tol:.2e}); {len(bad)} clear rows differ, "
            f"{len(near)} near-tie rows differ"
        )
        assert n_clear >= CLEAR_SHARE * R, f"dev {d}: only {n_clear} of {R} rows clear of the keys' noise"
        assert not bad, f"dev {d}: {len(bad)} clear rows select other experts than float64 (first {bad[:5]})"
        m = clear.nonzero().flatten()
        dense = torch.zeros(R, E, dtype=torch.float64).scatter_(1, idx, w)[m]
        fw = lambda lgt: torch.zeros(len(m), E, dtype=lgt.dtype).scatter_(  # noqa: E731
            1, idx64[m], ref.weights(lgt, idx64[m], sf, c["route_scale"], c["epsilon"])
        )
        want = fw(lg[m].double())
        lim = noise_limit(fw, [lg[m]], *c["noise"]["weights"].values())
        rel = float((dense - want).norm() / want.norm())
        a, b = (dense - dense.mean()).flatten(), (want - want.mean()).flatten()
        pcc = float(a @ b / (a.norm() * b.norm()))
        print(
            f"{c['id']} dev {d}: weights rel {rel:.3e} (limit {lim['rel']:.3e}, noise {lim['noise_rel']:.3e}) "
            f"pcc {pcc:.8f} (a diagnostic)"
        )
        assert rel <= lim["rel"], f"dev {d}: weights rel L2 {rel:.3e} > limit {lim['rel']:.3e}"
    dev_b = _inputs(mesh_device, c, c["seed"] + 1)[1]
    determinism.assert_deterministic(lambda: _call(c, dev), lambda: _call(c, dev_b), first=out, label=c["id"])
