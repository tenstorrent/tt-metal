# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Model cases of ttnn.bringup.unified_routed_expert_moe (cases.py), random inputs against reference.py.

Per case: a ROW_MAJOR bf16 dispatched buffer per chip, the experts' weights cut to the case's dtype on the device (the
reference reads the same rounded values back), and device-resident counts / region offsets / local->global table
built the way the model's masked_bincount / offset_cumsum lay them out: each mesh column is one dispatch group whose
tokens pick their top-k experts; an expert's count is the number of its group's tokens that picked it; a chip's
regions are its local experts' counts rounded up to a tile, in local order. Some experts are forced hot (count =
group_tokens = max_dispatched_tokens_per_expert, every chunk of the program) and some cold (count 0, skipped).

Checked: the rows [offset, offset + count) of every local expert of every chip. The op allocates its output
uninitialized and writes whole tile rows, so every other row (the region's tail past the count, gaps, unrouted space)
is don't-care, in the reference check and in the determinism check (FORK_TESTS.md 4b) alike.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import ttnn
from bringup_fw.testing import determinism

# torch is imported inside each function: ttnn/ allows no module-level torch import (check-torch-imports-in-ttnn)
_HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(f"unified_routed_expert_ffn_tests_{name}", _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


CASES = _load("cases").CASES
reference = _load("reference")


def _global_id(case, r: int, c: int, l: int) -> int:
    rows, per_chip = case["mesh"][0], case["experts_per_chip"]
    assert case["expert_layout"] == "column_major", case["expert_layout"]
    return (c * rows + r) * per_chip + l


def _routing(case, seed: int):
    """counts [rows, cols, E] and offsets [rows, cols, E] (int64): per column one top-k routing of group_tokens
    tokens over E experts, biased so the loads spread; hot experts picked by every token, cold ones by none."""
    import torch

    rows, cols = case["mesh"]
    E, k, T, per_chip = case["n_experts"], case["top_k"], case["group_tokens"], case["experts_per_chip"]
    g = torch.Generator().manual_seed(seed)
    special = torch.randperm(E, generator=g)
    hot, cold = (
        special[: case["hot_experts"]],
        special[case["hot_experts"] : case["hot_experts"] + case["cold_experts"]],
    )
    counts = torch.zeros(rows, cols, E, dtype=torch.int64)
    offsets = torch.zeros(rows, cols, E, dtype=torch.int64)
    for c in range(cols):
        scores = torch.randn(T, E, generator=g) + torch.randn(E, generator=g) * case["bias_std"]
        scores[:, hot] = float("inf")
        scores[:, cold] = float("-inf")
        picks = scores.topk(k, dim=-1).indices  # k distinct experts per token
        counts[:, c] = torch.bincount(picks.reshape(-1), minlength=E)
        for r in range(rows):
            o = 0
            for l in range(per_chip):
                e = _global_id(case, r, c, l)
                offsets[r, c, e] = o
                o += -(-int(counts[r, c, e]) // ttnn.TILE_SIZE) * ttnn.TILE_SIZE
            assert o <= case["tokens"], (r, c, o)
    return counts, offsets


def _to_mesh(t: torch.Tensor, mesh, dtype, layout, shape):
    """[rows, cols, ...] -> chip (r, c) holds t[r, c], reshaped to ``shape``."""
    rows, cols = tuple(mesh.shape)
    d = ttnn.from_torch(
        t,
        dtype=dtype,
        layout=layout,
        device=mesh,
        memory_config=ttnn.DRAM_MEMORY_CONFIG,
        mesh_mapper=ttnn.ShardTensor2dMesh(mesh, mesh_shape=(rows, cols), dims=(0, 1)),
    )
    return ttnn.reshape(d, shape)


def _per_chip(t) -> list[torch.Tensor]:
    """Device i of the mesh, row-major: chip (i // cols, i % cols)."""
    return [ttnn.to_torch(s) for s in ttnn.get_device_tensors(t)]


def _inputs(case, mesh, seed: int):
    """Device x, offsets, counts plus the host x (bf16, as the device holds it) and the routing."""
    import torch

    rows, cols = case["mesh"]
    counts, offsets = _routing(case, seed)
    g = torch.Generator().manual_seed(seed + 1_000_003)
    x = torch.empty(rows, cols, case["tokens"], case["emb"], dtype=torch.bfloat16)
    for r in range(rows):
        for c in range(cols):
            x[r, c] = torch.randn(case["tokens"], case["emb"], generator=g)
    E = case["n_experts"]
    dev = (
        _to_mesh(x.unsqueeze(2), mesh, ttnn.bfloat16, ttnn.ROW_MAJOR_LAYOUT, (1, 1, case["tokens"], case["emb"])),
        _to_mesh(offsets.to(torch.int32).unsqueeze(2), mesh, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, E)),
        _to_mesh(counts.to(torch.int32).unsqueeze(2), mesh, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, E)),
    )
    return dev, x, counts, offsets


def _weights(case, mesh, g):
    """Per local slot l: chip (r, c) gets its own random expert. Cut to the case's dtype on the device; returns the
    device lists and the rounded values read back, [slot] -> [rows * cols, K, N] bf16 (bfp8 fits bf16 exactly)."""
    import torch

    rows, cols = case["mesh"]
    H, I, s = case["emb"], case["hidden"], case["weight_scale"]
    wdt = getattr(ttnn, case["weight_dtype"])
    dev, host = {"gate": [], "up": [], "down": []}, {"gate": [], "up": [], "down": []}
    for _ in range(case["experts_per_chip"]):
        for name, (K, N) in (("gate", (H, I)), ("up", (H, I)), ("down", (I, H))):
            w = (torch.randn(rows, cols, K, N, generator=g) * s).to(torch.bfloat16)
            t = _to_mesh(w, mesh, ttnn.bfloat16, ttnn.TILE_LAYOUT, (K, N))
            if wdt != ttnn.bfloat16:
                t2 = ttnn.typecast(t, wdt)
                ttnn.deallocate(t)
                t = t2
            dev[name].append(t)
            host[name].append(torch.stack([p.reshape(K, N).to(torch.bfloat16) for p in _per_chip(t)]))
    return dev, host


def _valid_rows(case, counts, offsets, r: int, c: int) -> list[tuple[int, int, int]]:
    """(local slot, start, count) of chip (r, c)'s experts with a nonzero count."""
    out = []
    for l in range(case["experts_per_chip"]):
        e = _global_id(case, r, c, l)
        n = min(int(counts[r, c, e]), case["max_dispatched_tokens_per_expert"])
        if n:
            out.append((l, int(offsets[r, c, e]), n))
    return out


def _gather_valid(out, case, counts, offsets) -> list[torch.Tensor]:
    """Per chip, its valid output rows concatenated (the only rows the op defines)."""
    import torch

    cols = case["mesh"][1]
    res = []
    for i, t in enumerate(_per_chip(out)):
        t = t.reshape(case["tokens"], case["emb"])
        spans = _valid_rows(case, counts, offsets, i // cols, i % cols)
        res.append(torch.cat([t[o : o + n] for _, o, n in spans]) if spans else t[:0])
    return res


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    import torch

    a, b = a.double().flatten(), b.double().flatten()
    a, b = a - a.mean(), b - b.mean()
    return float((a @ b) / (a.norm() * b.norm()))


@pytest.mark.timeout(1800)
@pytest.mark.parametrize(
    "case, mesh_device, device_params",
    [
        (
            c,
            tuple(c["mesh"]),
            # a case's math depends on its mesh (dispatch groups, expert mapping): run it only on a box of that size
            {
                "fabric_config": getattr(ttnn.FabricConfig, c["fabric_config"]),
                "require_exact_physical_num_devices": True,
            },
        )
        for c in CASES
    ],
    ids=[c["id"] for c in CASES],
    indirect=["mesh_device", "device_params"],
)
def test_unified_routed_expert_moe(case, mesh_device, device_params):
    import torch

    rows, cols = case["mesh"]
    assert tuple(mesh_device.shape) == (rows, cols)
    per_chip = case["experts_per_chip"]

    ids = torch.tensor(
        [[[_global_id(case, r, c, l) for l in range(per_chip)] for c in range(cols)] for r in range(rows)],
        dtype=torch.int32,
    )
    idx_table = _to_mesh(ids.unsqueeze(2), mesh_device, ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, (1, per_chip))
    w_dev, w_host = _weights(case, mesh_device, torch.Generator().manual_seed(case["seed"]))
    ckc = case["compute_kernel_config"]
    kwargs = dict(
        compute_kernel_config=ttnn.WormholeComputeKernelConfig(
            math_fidelity=getattr(ttnn.MathFidelity, ckc["math_fidelity"]),
            math_approx_mode=ckc["math_approx_mode"],
            fp32_dest_acc_en=ckc["fp32_dest_acc_en"],
            packer_l1_acc=ckc["packer_l1_acc"],
            dst_full_sync_en=ckc["dst_full_sync_en"],
        ),
        activation=getattr(ttnn.bringup.RoutedExpertActivation, case["activation"]),
        high_precision=case["high_precision"],
    )

    def call(dev):
        x, offs, cnts = dev
        return ttnn.bringup.unified_routed_expert_moe(
            x,
            offs,
            cnts,
            idx_table,
            w_dev["gate"],
            w_dev["up"],
            w_dev["down"],
            case["max_dispatched_tokens_per_expert"],
            **kwargs,
        )

    dev_a, x_a, counts_a, offsets_a = _inputs(case, mesh_device, case["seed"])

    def run_a():
        out = call(dev_a)
        assert out.dtype == ttnn.bfloat16 and out.layout == ttnn.TILE_LAYOUT, (out.dtype, out.layout)
        assert tuple(out.shape) == (1, 1, case["tokens"], case["emb"]), out.shape
        got = _gather_valid(out, case, counts_a, offsets_a)
        ttnn.deallocate(out)
        return got

    got = run_a()

    # reference, chip by chip, on the bf16 x and the device-rounded weights
    worst_pcc, worst_rel, rows_checked, hot_seen, zero_seen = 1.0, 0.0, 0, 0, 0
    for r in range(rows):
        for c in range(cols):
            i = r * cols + c
            spans = _valid_rows(case, counts_a, offsets_a, r, c)
            zero_seen += per_chip - len(spans)
            hot_seen += sum(n == case["max_dispatched_tokens_per_expert"] for _, _, n in spans)
            ref, valid = reference.routed_expert_moe(
                x_a[r, c],
                offsets_a[r, c],
                counts_a[r, c],
                ids[r, c],
                [w[i] for w in w_host["gate"]],
                [w[i] for w in w_host["up"]],
                [w[i] for w in w_host["down"]],
                case["max_dispatched_tokens_per_expert"],
                act=case["activation"],
            )
            assert int(valid.sum()) == got[i].shape[0], (r, c)
            pos = 0
            for l, o, n in spans:
                d, e = got[i][pos : pos + n].float(), ref[o : o + n]
                pos += n
                assert torch.isfinite(d).all(), f"chip ({r},{c}) expert slot {l}: non-finite output"
                rel = float((d - e).norm() / e.norm())
                worst_rel = max(worst_rel, rel)
                if n > 1:
                    worst_pcc = min(worst_pcc, _pcc(d, e))
            rows_checked += pos
    print(
        f"{case['id']}: {rows_checked} rows over {rows * cols} chips ({hot_seen} full experts, {zero_seen} empty): "
        f"min pcc {worst_pcc:.6f}, max rel err {worst_rel:.5f}"
    )
    assert hot_seen > 0 and zero_seen > 0, "the routing must exercise full and empty experts"
    assert worst_pcc >= case["pcc"], f"pcc {worst_pcc:.6f} < {case['pcc']}"
    assert worst_rel <= case["max_rel_err"], f"rel err {worst_rel:.5f} > {case['max_rel_err']}"

    # determinism: A (other routing and x in B, so the counts the kernels read change in between), valid rows only
    del x_a
    dev_b, x_b, counts_b, offsets_b = _inputs(case, mesh_device, case["seed"] + 1)
    del x_b

    def run_b():
        out = call(dev_b)
        got_b = _gather_valid(out, case, counts_b, offsets_b)
        ttnn.deallocate(out)
        return got_b

    determinism.assert_deterministic(run_a, run_b, first=got, label=case["id"])
