# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""PERF EXPERIMENT gamma_mcast — γ read once per shared-window group and multicast vs per-core DRAM reads.

The whole rms_norm op is dispatched through this dir's copy of the program descriptor + kernels
(gm_program_descriptor.py, kernels/); only the writer's γ supply differs between variants (env GM_GAMMA_MCAST:
off = the op's current per-core direct-prefix read = baseline; all / row / col = multicast topologies; suffix
`+hs` = receiver handshake on). Every variant of a case runs inside ONE test, in order, so the profiler CSV rows
map 1:1 onto the labels appended to GM_LOG. Correctness: PCC vs torch at the exact config AND bit-exact
equality with the baseline output of the same case (γ delivery must not change a single bit).

    GM_CASES=focus GM_VARIANTS=off,all scripts/run_safe_pytest.sh --profile <this file> --import-mode=prepend
    (--import-mode=prepend: pytest.ini importlib mode would import this file as ttnn.ttnn.* and re-run ttnn/__init__)
    python3 <dir>/gm_report.py
"""

import os
import sys

import pytest

torch = pytest.importorskip("torch")  # ttnn/ forbids a module-level torch import
import ttnn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gm_program_descriptor as gpd  # noqa: E402

from eval.sharding import shard_config  # noqa: E402

ML = ttnn.TensorMemoryLayout
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.environ.get("GM_LOG", os.path.join(HERE, "gm_order.txt"))

# id -> (rows, hidden, memory_layout, shard, gamma_layout | None, fp32_dest_acc_en, gamma dtype[, x layout])
T, RM = ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT
CASES = {
    "focus": (8192, 5120, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),
    "i8192x7168": (8192, 7168, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),
    "i8192x1024": (8192, 1024, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),
    "i7168x2048": (7168, 2048, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),
    "i640x6144": (640, 6144, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),
    "i32x7168": (32, 7168, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),
    "rmg8192x5120": (8192, 5120, ML.INTERLEAVED, None, RM, False, ttnn.bfloat16),
    "nog8192x5120": (8192, 5120, ML.INTERLEAVED, None, None, False, ttnn.bfloat16),
    "fp32d8192x5120": (8192, 5120, ML.INTERLEAVED, None, T, True, ttnn.bfloat16),
    "bfp8g8192x5120": (8192, 5120, ML.INTERLEAVED, None, T, False, ttnn.bfloat8_b),
    "b8192x1024": (8192, 1024, ML.BLOCK_SHARDED, ([1024, 128], (8, 8)), T, False, ttnn.bfloat16),
    "rmg640x6144": (640, 6144, ML.INTERLEAVED, None, RM, False, ttnn.bfloat16),
    # domain probes
    "i2400x1024": (2400, 1024, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),  # R1 on 75 cores: 2 rectangles
    "rmx2400x1024": (2400, 1024, ML.INTERLEAVED, None, T, False, ttnn.bfloat16, RM),  # ROW_MAJOR x
    "i320x6144": (320, 6144, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),  # R3, shared slices not groupable
    "i8192x5000": (8192, 5000, ML.INTERLEAVED, None, T, False, ttnn.bfloat16),  # W % 32 != 0
    "h8192x1024": (8192, 1024, ML.HEIGHT_SHARDED, ([128, 1024], (8, 8)), T, False, ttnn.bfloat16),
    "w32x7168": (32, 7168, ML.WIDTH_SHARDED, ([32, 256], (7, 4)), T, False, ttnn.bfloat16),
    "b7168x2048": (7168, 2048, ML.BLOCK_SHARDED, ([736, 192], (11, 10)), T, False, ttnn.bfloat16),
}
VARIANTS = ["off", "all", "row", "col", "all+hs", "tile", "attach", "abl"]


def _select(env, ids):
    want = [w for w in os.environ.get(env, "").split(",") if w]
    return [i for i in ids if not want or i in want]


def _run(tx, tg, cfg):
    out = ttnn.allocate_tensor_on_device(
        ttnn.Shape(list(tx.shape)), tx.dtype, tx.layout, tx.device(), tx.memory_config()
    )
    pd = gpd.create_program_descriptor(tx, out, gamma=tg, epsilon=1e-6, compute_kernel_config=cfg)
    io = [tx] + ([tg] if tg is not None else []) + [out]
    return ttnn.generic_op(io, pd)


def _pcc(got, exp):
    return torch.corrcoef(torch.stack([got.double().flatten(), exp.double().flatten()]))[0, 1].item()


@pytest.mark.parametrize("case", _select("GM_CASES", list(CASES)))
def test_gamma_mcast(device, monkeypatch, case):
    rows, hidden, ml, shard, g_layout, fp32_dest, g_dtype = CASES[case][:7]
    x_layout = CASES[case][7] if len(CASES[case]) > 7 else T
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    if os.environ.get("GM_ZONES", "0") == "1":
        monkeypatch.setenv("RMS_NORM_PERF_ZONES", "1")  # stage zones: attribution only, inflates the wall
    torch.manual_seed(0)
    x = torch.randn((1, 1, rows, hidden)).to(torch.bfloat16)
    g = torch.randn(hidden).to(torch.bfloat16)
    mc = ttnn.DRAM_MEMORY_CONFIG
    if shard is not None:
        mc = shard_config(shard[0], shard[1], ml, layout=x_layout, dtype=ttnn.bfloat16, device=device)
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=x_layout, device=device, memory_config=mc)
    tg = None
    if g_layout is not None:
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=g_dtype, layout=g_layout, device=device)
        g_eff = ttnn.to_torch(tg).flatten()[:hidden].double()  # γ as stored (bfp8 quantized)
    else:
        g_eff = torch.ones(hidden, dtype=torch.float64)
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
    )
    xf = x.double()
    exp = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6) * g_eff

    base = None
    failures = []
    # GM_REPEAT=n: the variant list n times, interleaved (off, all, ..., off, all, ...) to read the noise band.
    for variant in _select("GM_VARIANTS", VARIANTS) * int(os.environ.get("GM_REPEAT", "1")):
        mode, _, hs = variant.partition("+")
        ablate = mode == "abl"  # measurement-only: baseline with the γ DRAM reads ablated (garbage numerics)
        if ablate:
            mode = "off"
            monkeypatch.setenv("RMS_NORM_ABLATE", "gamma")
        else:
            monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
        monkeypatch.setenv("GM_GAMMA_MCAST", mode)
        monkeypatch.setenv("GM_GAMMA_MCAST_HS", "1" if hs == "hs" else "0")
        # Report the topology actually used (None = absent channel = baseline path).
        plan = gpd.select_plan(tx, tg, fp32_dest)
        resident = gpd.choose_gamma_resident(plan.geometry, plan.block_rows, plan.x_depth, gpd.l1_budget(device))
        topo = gpd.gamma_mcast_groups(plan, resident)
        out = _run(tx, tg, cfg)
        got = ttnn.to_torch(out)
        with open(LOG, "a") as fh:
            fh.write(
                f"{case}|{variant}|{plan.regime}|cores={len(plan.work)}|G={plan.topo.G}|resident={int(resident)}"
                f"|direct={gpd.gamma_direct(plan.geometry, resident)}|block_rows={plan.block_rows}|topo={topo}\n"
            )
        if ablate:
            continue
        pcc = _pcc(got, exp)
        print(f"GM {case} {variant}: pcc={pcc:.6f} topo={topo} regime={plan.regime} resident={resident}")
        if pcc < 0.9995:
            failures.append(f"{variant}: PCC {pcc}")
        if base is None:
            base = got
        elif not torch.equal(got, base):
            diff = (got.float() - base.float()).abs()
            failures.append(f"{variant}: not bit-exact vs first variant (max |d|={diff.max().item()})")
    assert not failures, failures


@pytest.mark.parametrize("case", _select("GM_RELAUNCH_CASES", ["focus", "i640x6144", "rmg8192x5120", "i8192x7168"]))
def test_gamma_mcast_relaunch(device, monkeypatch, case):
    """Same cached multicast program, relaunched with a DIFFERENT γ tensor: the second launch must see the new γ
    (guards against a stale Counter semaphore letting receivers consume the previous launch's landed γ)."""
    rows, hidden, ml, shard, g_layout, fp32_dest, g_dtype = CASES[case][:7]
    monkeypatch.delenv("RMS_NORM_PERF_ZONES", raising=False)
    monkeypatch.delenv("RMS_NORM_ABLATE", raising=False)
    monkeypatch.setenv("GM_GAMMA_MCAST", "all")
    monkeypatch.setenv("GM_GAMMA_MCAST_HS", "0")
    torch.manual_seed(1)
    x = torch.randn((1, 1, rows, hidden)).to(torch.bfloat16)
    tx = ttnn.from_torch(
        x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=ttnn.DRAM_MEMORY_CONFIG
    )
    cfg = ttnn.ComputeConfigDescriptor(
        math_fidelity=ttnn.MathFidelity.HiFi2, fp32_dest_acc_en=fp32_dest, math_approx_mode=False
    )
    xf = x.double()
    norm = xf / torch.sqrt((xf * xf).mean(-1, keepdim=True) + 1e-6)
    for k in range(3):
        g = (torch.randn(hidden) * (k + 1) + k).to(torch.bfloat16)
        tg = ttnn.from_torch(g.reshape(1, 1, 1, -1), dtype=g_dtype, layout=g_layout, device=device)
        got = ttnn.to_torch(_run(tx, tg, cfg))
        pcc = _pcc(got, norm * g.double())
        print(f"GM relaunch {case} #{k}: pcc={pcc:.6f}")
        assert pcc >= 0.9995, f"launch {k}: PCC {pcc}"
