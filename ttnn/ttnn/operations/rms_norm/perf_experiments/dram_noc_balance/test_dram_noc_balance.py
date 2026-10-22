# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PERF EXPERIMENT dram_noc_balance — position- and bank-aware NoC choice for the interleaved x reads and output writes.

The whole rms_norm op is dispatched through this dir's copy of the program descriptor + kernels (dnb_program_descriptor.py,
kernels/). Env DNB_POLICY selects the variant (see dnb_program_descriptor.dnb_masks): `base` = the op unchanged
(DM_DEDICATED_NOC, reader NoC0, writer NoC1 — byte-identical loops); every other policy runs both DM kernels in
DM_DYNAMIC_NOC and picks a NoC per (Tensix core, DRAM bank). Every variant of a case runs inside ONE test, in order,
so the profiler CSV rows map 1:1 onto the labels appended to DNB_LOG. Correctness: PCC vs torch at the exact config
AND bit-exact equality with the `base` output of the same case (data movement only — not one bit may change).

    DNB_CASES=focus DNB_VARIANTS=base,near scripts/run_safe_pytest.sh --profile <this file> --import-mode=prepend
    python3 <dir>/dnb_report.py
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")  # ttnn/ forbids a module-level torch import
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dnb_program_descriptor as dpd  # noqa: E402

from eval.sharding import shard_config  # noqa: E402

ML = ttnn.TensorMemoryLayout
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.environ.get("DNB_LOG", os.path.join(HERE, "dnb_order.txt"))

T, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
# id -> (rows, hidden, memory_layout, shard, gamma_layout | None, fp32_dest_acc_en, x layout)
CASES = {
    "focus": (8192, 5120, ML.INTERLEAVED, None, T, False, T),
    "i8192x7168": (8192, 7168, ML.INTERLEAVED, None, T, False, T),
    "i8192x1024": (8192, 1024, ML.INTERLEAVED, None, T, False, T),
    "i7168x2048": (7168, 2048, ML.INTERLEAVED, None, T, False, T),
    "i640x6144": (640, 6144, ML.INTERLEAVED, None, T, False, T),
    "i32x7168": (32, 7168, ML.INTERLEAVED, None, T, False, T),
    "i2048x6144": (2048, 6144, ML.INTERLEAVED, None, T, False, T),
    "i2400x1024": (2400, 1024, ML.INTERLEAVED, None, T, False, T),  # R1 on 75 cores
    "i8192x5000": (8192, 5000, ML.INTERLEAVED, None, T, False, T),  # W % 32 != 0 (pad-column zeroing path)
    "i320x6144": (320, 6144, ML.INTERLEAVED, None, T, False, T),  # R3, strided γ sharing (per-core γ reads)
    "i64x7168": (64, 7168, ML.INTERLEAVED, None, T, False, T),
    "i128x16384": (128, 16384, ML.INTERLEAVED, None, T, False, T),
    "nog8192x5120": (8192, 5120, ML.INTERLEAVED, None, None, False, T),
    "fp32d8192x5120": (8192, 5120, ML.INTERLEAVED, None, T, True, T),
    "rmx8192x5120": (8192, 5120, ML.INTERLEAVED, None, T, False, RM),  # ROW_MAJOR x (stick path, untouched)
    "rmx640x6144": (640, 6144, ML.INTERLEAVED, None, T, False, RM),
    "b8192x1024": (8192, 1024, ML.BLOCK_SHARDED, ([1024, 128], (8, 8)), T, False, T),
    "h8192x1024": (8192, 1024, ML.HEIGHT_SHARDED, ([128, 1024], (8, 8)), T, False, T),
    "w32x7168": (32, 7168, ML.WIDTH_SHARDED, ([32, 256], (7, 4)), T, False, T),
}
VARIANTS = [
    "base",
    "dyn0",
    "swap",
    "swap_east",
    "swap_west",
    "checker",
    "split",
    "split_rd",
    "split_wr",
    "split_core",
    "near",
    "near_rd",
    "near_wr",
    "far",
]


def _select(env, ids):
    want = [w for w in os.environ.get(env, "").split(",") if w]
    return [i for i in ids if not want or i in want]


def _run(tx, tg, cfg):
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(tx.shape)), tx.dtype, tx.layout, tx.device(), tx.memory_config()
    )
    pd = dpd.create_program_descriptor(tx, out, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)
    io = [tx] + ([tg] if tg is not None else []) + [out]
    return ttnn.generic_op(io, pd)


def _pcc(got, exp):
    return torch.corrcoef(torch.stack([got.double().flatten(), exp.double().flatten()]))[0, 1].item()


@pytest.mark.parametrize("case", _select("DNB_CASES", list(CASES)))
def test_dram_noc_balance(device, monkeypatch, case):
    rows, hidden, ml, shard, g_layout, fp32_dest, x_layout = CASES[case]
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    if os.environ.get("DNB_ZONES", "0") == "1":
        monkeypatch.setenv("RMS_NORM_PERF_ZONES", "1")  # stage zones: attribution only, inflates the wall
    torch.manual_seed(0)
    x = torch.randn((1, 1, rows, hidden)).to(torch.bfloat16)
    g = torch.randn(hidden).to(torch.bfloat16)
    mc = ttnn.DRAM_MEMORY_CONFIG
    if shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=x_layout, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=x_layout, device=device, memory_config=mc)
    tg = None
    g_eff = torch.ones(hidden, dtype=torch.float64)
    if g_layout is not None:
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=ttnn.bfloat16, layout=g_layout, device=device)
        g_eff = g.double()
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
    )
    xf = x.double()
    exp = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * g_eff

    base = None
    failures = []
    variants = [v for v in os.environ.get("DNB_VARIANTS", "").split(",") if v] or VARIANTS
    for variant in variants * int(os.environ.get("DNB_REPEAT", "1")):
        ablate = ""
        policy = variant
        if "@" in variant:  # policy@ablation (measurement only, garbage numerics), e.g. near@reads
            policy, ablate = variant.split("@")
            monkeypatch.setenv("RMS_NORM_ABLATE", ablate.replace("+", ","))
        else:
            monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
        monkeypatch.setenv("DNB_POLICY", policy)
        dpd.DNB_LAST_MASKS.clear()
        plan = dpd.select_plan(tx, tg, fp32_dest)
        out = _run(tx, tg, cfg)
        got = ttnn.to_torch(out)
        with open(LOG, "a") as fh:
            swapped = len(dpd.DNB_LAST_MASKS) if policy.startswith("ded:") else "-"
            fh.write(
                f"{case}|{variant}|{plan.regime}|cores={len(plan.work)}|owners={plan.topo.owners}|G={plan.topo.G}"
                f"|block_rows={plan.block_rows}|swapped={swapped}\n"
            )
        if os.environ.get("DNB_PRINT_MASKS") and dpd.DNB_LAST_MASKS:
            for k, v in sorted(dpd.DNB_LAST_MASKS.items(), key=lambda kv: (kv[0][1], kv[0][0])):
                print(f"DNB mask {policy} logical={k} virtual=({v[0]},{v[1]}) rd={v[2]:#04x} wr={v[3]:#04x}")
        if ablate:
            continue
        pcc = _pcc(got, exp)
        print(
            f"DNB {case} {variant}: pcc={pcc:.6f} regime={plan.regime} nan={int(got.isnan().sum())} "
            f"shape={tuple(got.shape)} max={got.float().abs().max().item():.3f}"
        )
        if pcc < 0.9995:
            failures.append(f"{variant}: PCC {pcc}")
        if base is None:
            base = got
        elif not torch.equal(got, base):
            diff = (got.float() - base.float()).abs()
            failures.append(f"{variant}: not bit-exact vs first variant (max |d|={diff.max().item()})")
    assert not failures, failures
