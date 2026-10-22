# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""bcast_srcb_reuse compute-only bench (single Tensix core, every operand a resident L1 shard, no NoC traffic).

One rms_norm normalize-stage pass over an [R x WS]-tile block, repeated REPS times inside the kernel. Variants
(BSR_VARIANTS, comma list of names below; default all), shapes (BSR_SHAPES "RxWS" list, default 32x4 = the
focus shard's per-core block), BSR_REPS (default 16), BSR_FP32=1 for fp32_dest_acc_en. Run:

    BSR_VARIANTS=g_chain,g_hold scripts/run_safe_pytest.sh --profile <this file>
    python3 ttnn/ttnn/operations/rms_norm/perf_experiments/bcast_srcb_reuse/report.py

Output-producing variants (g_chain, g_raw, g_hold, i_chain, i_raw, i_hold) are correctness-gated: PCC vs a
torch fp64 product AND bit-exact vs the chain variant on the same input. The "*_u" / "*_m" / "*_p" variants are
thread-isolation probes (real work on one thread, the others replaced by the data-valid handshake): garbage
output, timing only.
"""

import os
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
import ttnn

HERE = Path(__file__).parent
KDIR = HERE / "bench_kernels"
LOG = os.environ.get("BSR_LOG", str(HERE / "bench_order.txt"))

VARIANTS = {
    "g_chain": 1, "g_u": 2, "g_m": 3, "g_p": 4, "g_raw": 5, "g_hold": 6, "g_hold_u": 7, "g_hold_m": 8, "g_hold2": 9,
    "i_chain": 11, "i_u": 12, "i_m": 13, "i_p": 14, "i_raw": 15, "i_hold": 16, "i_hold_u": 17, "i_hold_m": 18, "i_hold_pf": 19,
    "s_chain": 21, "s_u": 22, "s_m": 23, "i_hb1": 31, "i_hsplit": 32, "i_hpt": 33, "i_holdf": 34, "g_holdf": 35, "i_hold_fl": 36,
}  # fmt: skip
CHECKED = {
    "i_hold_fl": "i_chain",
    "i_holdf": "i_chain",
    "g_holdf": "g_chain",
    "i_hpt": "i_chain",
    "i_hsplit": "i_chain",
    "g_raw": "g_chain",
    "g_hold": "g_chain",
    "g_hold2": "g_chain",
    "i_hold_pf": "i_chain",
    "i_raw": "i_chain",
    "i_hold": "i_chain",
}
CB_X, CB_SQ, CB_INV, CB_G, CB_OUT = 1, 3, 4, 6, 16


def _sel(env, default):
    v = [s for s in os.environ.get(env, "").split(",") if s]
    return v or default


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _mc(h, w):
    return ttnn.create_sharded_memory_config(
        shape=(h, w),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _run(device, v, R, WS, reps, fp32, x, g, inv):
    H, W = R * 32, WS * 32
    tx = ttnn.from_torch(x, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mc(H, W))
    tg = ttnn.from_torch(g, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mc(32, W))
    ti = ttnn.from_torch(inv, dtype=ttnn.bfloat16, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_mc(H, 32))
    out = ttnn.allocate_tensor_on_device(ttnn.Shape([H, W]), ttnn.bfloat16, ttnn.TILE_LAYOUT, device, _mc(H, W))
    sb = 4 if fp32 else 8
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, tx),
        ttnn.cb_descriptor_from_sharded_tensor(CB_G, tg),
        ttnn.cb_descriptor_from_sharded_tensor(CB_INV, ti),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, out),
        ttnn.CBDescriptor(
            total_size=2048 * R,
            core_ranges=_core(),
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_SQ, data_format=ttnn.bfloat16, page_size=2048)],
        ),
    ]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KDIR / "bench_compute.cpp"),
        core_ranges=_core(),
        compile_time_args=[R, WS, reps, v, sb],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=getattr(ttnn.MathFidelity, os.environ.get("BSR_FID", "HiFi2")),
            fp32_dest_acc_en=fp32,
            math_approx_mode=False,
        ),
    )
    desc = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
    res = ttnn.generic_op([tx, tg, ti, out], desc)
    return ttnn.to_torch(res).float()


def _pcc(a, b):
    return torch.corrcoef(torch.stack([a.double().flatten(), b.double().flatten()]))[0, 1].item()


@pytest.mark.parametrize("variant", _sel("BSR_VARIANTS", list(VARIANTS)))
@pytest.mark.parametrize("shape", _sel("BSR_SHAPES", ["32x4"]))
def test_bench(device, shape, variant):
    R, WS = (int(s) for s in shape.split("x"))
    reps = int(os.environ.get("BSR_REPS", "16"))
    fp32 = os.environ.get("BSR_FP32", "0") == "1"
    torch.manual_seed(0)
    x = torch.randn(R * 32, WS * 32).to(torch.bfloat16)
    g = torch.randn(32, WS * 32).to(torch.bfloat16)  # row 0 = γ; rows 1..31 must be ignored
    inv = (torch.rand(R * 32, 32) + 0.5).to(torch.bfloat16)  # column 0 = inv_rms; columns 1..31 ignored
    fid = os.environ.get("BSR_FID", "HiFi2")
    tag = f"{shape}{'_fp32' if fp32 else ''}{'' if fid == 'HiFi2' else '_' + fid}"
    got = _run(device, VARIANTS[variant], R, WS, reps, fp32, x, g, inv)
    with open(LOG, "a") as fh:
        fh.write(f"{tag}|{variant}|{reps}\n")
    if variant in CHECKED or variant in ("g_chain", "i_chain"):
        if variant.startswith("g"):
            ref = x.double() * g[0:1, :].double()
        else:
            ref = x.double() * inv[:, 0:1].double()
        p = _pcc(ref, got)
        print(f"BSR_RESULT {tag} {variant} pcc={p:.7f}")
        assert p >= 0.9995, f"PCC {p}"
        if variant in CHECKED:
            base = _run(device, VARIANTS[CHECKED[variant]], R, WS, 1, fp32, x, g, inv)
            with open(LOG, "a") as fh:
                fh.write(f"{tag}|ref_{CHECKED[variant]}|1\n")
            nd = int((base != got).sum())
            print(f"BSR_EXACT {tag} {variant} mismatches={nd}")
            assert nd == 0, f"{nd} elements differ from {CHECKED[variant]}"
