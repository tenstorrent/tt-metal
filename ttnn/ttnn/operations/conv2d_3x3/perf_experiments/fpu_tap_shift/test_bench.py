# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bake-off: conv2d_3x3's shifted N-slice taps (kw = 0 / 2) for a 1-tile-wide channel slice, built by the
untilize -> padded row-major slab -> 1-wide tilize round trip (the op today) vs straight from the TILE slab with
one-row FPU moves (perf experiment `fpu_tap_shift`). Compute only, one core, no data movement.

    FTS_BENCH=1 FTS_VARIANTS=base,shift FTS_D=1 FTS_REPS=24 \
        scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/fpu_tap_shift/test_bench.py

bf8_b slab (32 channels) -> bf8_b taps, fp32 DEST, LoFi (the res4.x prelude's formats). Every non-diag variant is gated
on bit-exactness against the torch shift of the bf8_b input (which the baseline also matches). Perf (DEVICE KERNEL
DURATION, in-process profiler) is printed, never asserted. Appends to results_bench.jsonl.
"""

import os

if os.environ.get("FTS_BENCH") == "1" and os.environ.get("FTS_PROF", "1") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import json
import socket
from pathlib import Path

import pytest

import ttnn

HERE = Path(__file__).parent
KERNEL = HERE / "bench" / "bench_shift.cpp"
RESULTS = HERE / "results_bench.jsonl"
TILE = 32
CB_SRC, CB_ZERO, CB_RM, CB_OUT = 0, 1, 2, 16
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"

VARIANTS = {
    "base": 0,  # untilize + tilize (the op today)
    "base_tilize": 1,
    "base_untilize": 2,
    "shift": 10,  # SrcB only: cur pass + nb pass (nb pass from the zero tile at row ends / zero taps)
    "shift_skip": 13,  # as shift, but zero passes skipped (relies on the DEST section clear)
    "shift_ab": 14,  # SrcA cur (one-row MOVA2D) + SrcB nb: broken (F1/F3 rows unwritten)
    "diag_copy": 11,
    "diag_nomove": 12,
    "diag_a1": 20,  # SrcA 1:1, one-row MOVA2D (expect cur)
    "diag_b1": 21,  # SrcB 1:1, one-row MOVB2D (expect nb)
    "diag_a8": 22,  # SrcA 1:1, 8-row MOVA2D (expect cur)
    "diag_b1i": 23,  # SrcB 1:1 via the SrcB path's immediates (expect cur)
    "diag_b1r": 24,  # SrcB 1:1 via replayed counter-stepping moves (expect cur)
    "diag_packonly": 27,  # standard pack of a cleared DEST, 4 per section (wrong output)
    "diag_unpackonly": 28,  # one full-tile SrcB load per tile, released, packs cleared DEST (wrong output)
    "shift_rep": 25,  # SrcB, cur rows by replayed runs, nb rows explicit, 2 loads per tile
}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _shard_cfg(h, w):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_core(), [h, w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _read_kernel_ns(device):
    ttnn.ReadDeviceProfiler(device)
    total, found = 0.0, False
    for programs in (ttnn.get_latest_programs_perf_data() or {}).values():
        for program in programs:
            entry = (getattr(program, "program_analyses_results", None) or {}).get(_DURATION_KEY)
            if entry is not None:
                total += float(entry.duration)
                found = True
    return total if found else None


def _descriptor(tensors, code, reps, d, R, t0, bm, W):
    tt_src, tt_zero, tt_rm, tt_out = tensors
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[code, reps, d, R, t0, bm, W],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(i, t) for i, t in ((CB_SRC, tt_src), (CB_ZERO, tt_zero))]
    cb_rm = ttnn.cb_descriptor_from_sharded_tensor(CB_RM, tt_rm)
    cb_rm.format_descriptors = [
        ttnn.CBFormatDescriptor(buffer_index=CB_RM, data_format=ttnn.bfloat16, page_size=2 * TILE * TILE)
    ]
    cbs += [cb_rm, ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tt_out)]
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


def _reference(img, d, R, t0, bm, W, which="shift"):
    """Taps (tile-row, kh, kw in 0 / 2) of the [R, W, 32] image, zero outside, stacked [6 bm * 32, 32]."""
    import torch

    tpr = W // TILE
    out = []
    for t in range(bm):
        ti = t0 + t
        ho, wo0 = ti // tpr, (ti % tpr) * TILE
        for kh in range(3):
            h = ho + kh * d - d
            for kw in (0, 2):
                tile = torch.zeros(TILE, TILE)
                if 0 <= h < R and which != "shift":
                    cn = (ti % tpr) + (0 if which == "cur" else (-1 if kw == 0 else 1))
                    if 0 <= cn < tpr:
                        tile = img[h, cn * TILE : (cn + 1) * TILE].clone()
                elif 0 <= h < R:
                    for r in range(TILE):
                        w = wo0 + r + (kw - 1) * d
                        if 0 <= w < W:
                            tile[r] = img[h, w]
                out.append(tile)
    return torch.cat(out)


@pytest.mark.skipif(os.environ.get("FTS_BENCH") != "1", reason="bench: run with FTS_BENCH=1")
def test_shift_bench(device):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    d = int(os.environ.get("FTS_D", "1"))
    W = int(os.environ.get("FTS_W", "64"))
    R = int(os.environ.get("FTS_R", "6"))
    bm = int(os.environ.get("FTS_BM", "8"))
    t0 = int(os.environ.get("FTS_T0", str(W // TILE)))
    reps = int(os.environ.get("FTS_REPS", "24"))
    variants = os.environ.get("FTS_VARIANTS", "base,shift").split(",")
    tag = os.environ.get("FTS_TAG", "")
    S = R * (W // TILE)
    n_out = 6 * bm
    lines = [f"tap shift bench box={box} arch={arch} 1 core, fp32 DEST, d={d} W={W} R={R} bm={bm} t0={t0} reps={reps}"]
    failures = []
    torch.manual_seed(7)
    src = (torch.randn(S * TILE, TILE) * 3).to(torch.bfloat16)
    tt_src = ttnn.from_torch(
        src, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_cfg(S * TILE, TILE)
    )
    img = ttnn.to_torch(tt_src).float().reshape(R, W, TILE)  # the bf8_b values
    ref = _reference(img, d, R, t0, bm, W)
    tt_zero = ttnn.from_torch(
        torch.zeros(TILE, TILE),
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_shard_cfg(TILE, TILE),
    )
    rm_px = R * (W + d) + d + 64
    rm_px = (rm_px + TILE - 1) // TILE * TILE
    tt_rm = ttnn.from_torch(
        torch.zeros(rm_px, TILE, dtype=torch.bfloat16),
        dtype=ttnn.bfloat16,
        layout=ttnn.ROW_MAJOR_LAYOUT,
        device=device,
        memory_config=_shard_cfg(rm_px, TILE),
    )
    base_ns = None
    for name in variants:
        code = VARIANTS[name]
        tt_out = ttnn.allocate_tensor_on_device(
            ttnn.Shape([n_out * TILE, TILE]), ttnn.bfloat8_b, ttnn.TILE_LAYOUT, device, _shard_cfg(n_out * TILE, TILE)
        )
        desc = _descriptor((tt_src, tt_zero, tt_rm, tt_out), code, reps, d, R, t0, bm, W)
        tensors = [tt_src, tt_zero, tt_rm, tt_out]
        ttnn.generic_op(tensors, desc)  # compile + correctness run
        got = ttnn.to_torch(tt_out).float()
        if os.environ.get("FTS_PROF", "1") == "1":
            _read_kernel_ns(device)  # flush
        ttnn.generic_op(tensors, desc)
        ns = _read_kernel_ns(device) if os.environ.get("FTS_PROF", "1") == "1" else 0.0
        exact = None
        if code not in (2,):
            want = ref
            if code in (20, 22, 21, 23, 24):
                want = _reference(img, d, R, t0, bm, W, "nb" if code == 21 else "cur")
            exact = bool(torch.equal(got, want))
            if code in (20, 21, 22, 23, 24) and not exact:
                bad = (got != want).reshape(-1, TILE, TILE)
                print(
                    name,
                    "bad rows L",
                    bad[..., :16].any(-1).sum().item(),
                    "R",
                    bad[..., 16:].any(-1).sum().item(),
                    "of",
                    bad.shape[0] * TILE,
                    "first bad (tile,row)",
                    bad.any(-1).nonzero()[:8].tolist(),
                )
        if name == "base":
            base_ns = ns
        if os.environ.get("FTS_DEBUG") == "1" and exact is False:
            flat = img.reshape(-1, TILE)
            for r in (1, 2, 17):
                print("got ", got[r].tolist())
                print("want", ref[r].tolist())
            for tt in range(0):
                print(f"tile {tt}:")
                for r in range(TILE):
                    row = got[tt * TILE + r]
                    hits = [k for k in range(flat.shape[0]) if torch.equal(flat[k], row)]
                    rhit = [k for k in range(flat.shape[0]) if torch.equal(flat[k], ref[tt * TILE + r])]
                    lh = [k for k in range(flat.shape[0]) if torch.equal(flat[k, :16], row[:16])]
                    rh = [k for k in range(flat.shape[0]) if torch.equal(flat[k, 16:], row[16:])]
                    rh += [f"L{k}" for k in range(flat.shape[0]) if torch.equal(flat[k, :16], row[16:])]
                    print(
                        f"  r{r:2d} got_px {hits[:3]} want_px {rhit[:3]} L {lh[:3]} R {rh[:3]} zero {bool((row == 0).all())}"
                    )
        if not name.startswith("diag") and exact is False:
            bad = (got != ref).reshape(-1, TILE, TILE).any(dim=2).nonzero()
            failures.append(f"{name}: differs from the torch shift ({bad.shape[0]} bad rows, first {bad[:6].tolist()})")
        speed = f"{base_ns / ns:5.3f}x" if (base_ns and ns) else "  -  "
        per_tile = ns / (n_out * reps) if ns else 0
        lines.append(
            f"  {name:<14} {ns:>10.0f} ns  {speed}  {per_tile:6.1f} ns/out-tile  "
            f"{'-' if exact is None else ('bit-exact vs torch shift' if exact else 'DIFFERS')}"
        )
        with RESULTS.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        variant=name,
                        ns=ns,
                        d=d,
                        W=W,
                        R=R,
                        bm=bm,
                        t0=t0,
                        reps=reps,
                        n_out=n_out,
                        ns_per_out_tile=per_tile,
                        exact=exact,
                        box=box,
                        arch=arch,
                        tag=tag,
                    )
                )
                + "\n"
            )
        tt_out.deallocate()
    print("\n" + "\n".join(lines))
    assert not failures, failures
