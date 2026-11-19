# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bench (Perf 12 `stride2_dest_taps`): stride-2 im2col taps built from the input's TILE slab in DEST
(op_copy/kernels/conv2d_3x3_s2_taps.hpp) on one core, compute only, the slab resident.

    S2T_BENCH=1 S2T_VARIANTS=s2t,packonly,nopack S2T_CT=4 S2T_W=256 S2T_H=128 S2T_ROWS=3 S2T_T0=90 S2T_REPS=16 \
        bench_run.sh <tag>

bf8_b slab -> bf8_b taps, fp32 DEST, LoFi (the op's formats). `s2t` is gated on bit-exactness against the torch tap
gather of the bf8_b input. Perf (DEVICE KERNEL DURATION) is joined from the profiler CSV by bench_report.py, never
asserted. Appends to results_bench.jsonl.
"""

import json
import os
import socket
from pathlib import Path

import pytest

import ttnn

HERE = Path(__file__).parent
KERNEL = HERE / "bench" / "bench_s2t.cpp"
READER = HERE / "bench" / "bench_reader.cpp"
RESULTS = HERE / "results_bench.jsonl"
TILE = 32
CB_IN, CB_OUT = 0, 16
VARIANTS = {
    "s2t": 0,
    "packonly": 1,
    "nopack": 2,
    "oplane": 10,
    "oplane_packonly": 11,
    "oplane_nopack": 12,
    "eoplane": 20,
    "eoplane_packonly": 21,
    "eoplane_nopack": 22,
    "op": 30,
    "op_packonly": 31,
    "op_nopack": 32,
    "empty": 99,
}
DIAG = {k for k in VARIANTS if k.endswith("packonly") or k.endswith("nopack") or k == "empty"}


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _shard_cfg(h, w):
    return ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(_core(), [h, w], ttnn.ShardOrientation.ROW_MAJOR),
    )


def _plan(o_first, o_end, W, num_in_px):
    """s2_taps::slab_plan (tile_slab_plan at stride 2, p 1, d 1): [(first input tile-row, count)]."""
    Wo = W // 2
    ho_first, wo_first = divmod(o_first, Wo)
    ho_last, wo_last = divmod(o_end - 1, Wo)
    f = (2 * ho_first - 1) * W + 2 * wo_first - 1
    band_len = (ho_last - ho_first) * 2 * W + (wo_last - wo_first) * 2 + 3
    segs, have_prev, prev_t_hi = [], False, 0
    for kh in range(3):
        v_lo = f + kh * W
        lo, hi = max(v_lo, 0), min(v_lo + band_len, num_in_px)
        if lo >= hi:
            continue
        t_lo, t_hi = lo // TILE, (hi + TILE - 1) // TILE
        first_new = prev_t_hi if (have_prev and t_lo < prev_t_hi) else t_lo
        if t_hi > first_new:
            segs.append((first_new, t_hi - first_new))
        prev_t_hi, have_prev = t_hi, True
    return segs


def _reference(X, ct, rows, W, H, o_first):
    """[rows * 32, 9 ct * 32]: unit (t, kh) tap kw, channel tile ch at column kh * 3 ct + kw * ct + ch."""
    import torch

    Wo = W // 2
    ref = torch.zeros(rows * TILE, 9 * ct * TILE)
    for t in range(rows):
        for r in range(TILE):
            o = o_first + TILE * t + r
            if o >= ((H - 1) // 2 + 1) * Wo:
                continue
            ho, wo = divmod(o, Wo)
            for kh in range(3):
                h = 2 * ho + kh - 1
                if not 0 <= h < H:
                    continue
                for kw in range(3):
                    w = 2 * wo + kw - 1
                    if not 0 <= w < W:
                        continue
                    col = (kh * 3 + kw) * ct
                    ref[t * TILE + r, col * TILE : (col + ct) * TILE] = X[h * W + w, :]
    return ref


def _descriptor(tt_in, tt_out, code, reps, ct, rows, W, H, o_first, pages):
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=[code, reps, ct, rows, W, H, o_first],
        defines=[],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    reader = ttnn.KernelDescriptor(
        kernel_source=str(READER),
        core_ranges=_core(),
        compile_time_args=[CB_IN, pages],
        defines=[],
        config=ttnn.ReaderConfigDescriptor(),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN, tt_in),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, tt_out),
    ]
    return ttnn.ProgramDescriptor(kernels=[reader, compute], semaphores=[], cbs=cbs)


def _cfgs():
    """S2T_CFGS: ';'-separated ct:W:H:rows:T0[:bf16] (default: the S2T_* single config)."""
    e = os.environ
    if e.get("S2T_CFGS"):
        out = []
        for c in e["S2T_CFGS"].split(";"):
            f = c.split(":")
            out.append((int(f[0]), int(f[1]), int(f[2]), int(f[3]), int(f[4]), len(f) > 5 and f[5] == "bf16"))
        return out
    return [
        (
            int(e.get("S2T_CT", "4")),
            int(e.get("S2T_W", "256")),
            int(e.get("S2T_H", "128")),
            int(e.get("S2T_ROWS", "3")),
            int(e.get("S2T_T0", "90")),
            e.get("S2T_BF16") == "1",
        )
    ]


CFGS = _cfgs()


@pytest.mark.skipif(os.environ.get("S2T_BENCH") != "1", reason="bench: run with S2T_BENCH=1")
@pytest.mark.parametrize(
    "cfg", CFGS, ids=[f"ct{c[0]}_W{c[1]}_H{c[2]}_r{c[3]}_t{c[4]}{'_bf16' if c[5] else ''}" for c in CFGS]
)
def test_s2t_bench(device, cfg):
    import torch

    box, arch = socket.gethostname(), str(device.arch())
    e = os.environ
    ct, W, H, rows, t0, bf16 = cfg
    o_first = t0 * TILE
    reps = int(e.get("S2T_REPS", "16"))
    dtype = ttnn.bfloat16 if bf16 else ttnn.bfloat8_b
    variants = e.get("S2T_VARIANTS", "s2t").split(",")
    tag = e.get("S2T_TAG", "")
    Wo = W // 2
    Ho = (H - 1) // 2 + 1
    o_end = min(o_first + rows * TILE, Ho * Wo)
    slab_tiles = [i for f, c in _plan(o_first, o_end, W, H * W) for i in range(f, f + c)]
    torch.manual_seed(7)
    img = (torch.randn(H * W, ct * TILE) * 3).to(torch.bfloat16)
    X = ttnn.to_torch(ttnn.from_torch(img, dtype=dtype, layout=ttnn.TILE_LAYOUT)).float()  # the device values
    slab = torch.cat([X[i * TILE : (i + 1) * TILE] for i in slab_tiles]).to(torch.bfloat16)
    S = len(slab_tiles)
    tt_in = ttnn.from_torch(
        slab, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard_cfg(S * TILE, ct * TILE)
    )
    ref = _reference(X, ct, rows, W, H, o_first)
    n_taps = rows * 9 * ct
    lines = [
        f"s2 taps bench box={box} arch={arch} 1 core fp32 DEST LoFi {dtype}: ct={ct} W={W} H={H} rows={rows} "
        f"o_first={o_first} slab={S} tile-rows reps={reps} taps/rep={n_taps}"
    ]
    failures = []
    for name in variants:
        tt_out = ttnn.from_torch(
            torch.full((rows * TILE, 9 * ct * TILE), 77.0),
            dtype=dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_shard_cfg(rows * TILE, 9 * ct * TILE),
        )
        desc = _descriptor(tt_in, tt_out, VARIANTS[name], reps, ct, rows, W, H, o_first, S * ct)
        ttnn.generic_op([tt_in, tt_out], desc)  # compile + correctness run
        got = ttnn.to_torch(tt_out).float()
        ttnn.generic_op([tt_in, tt_out], desc)  # timed run (2nd GenericOp row)
        exact, nbad = None, None
        if name not in DIAG:
            bad = got != ref
            nbad = int(bad.reshape(rows, TILE, 9 * ct, TILE).any(-1).sum())  # bad 32-wide rows
            exact = nbad == 0
            if not exact:
                idx = bad.reshape(rows, TILE, 9 * ct, TILE).any(-1).nonzero()[:12].tolist()
                failures.append(f"{name}: {nbad} bad tile rows, first (t, r, col) {idx}")
                if e.get("S2T_DEBUG") == "1":
                    keys = {}
                    for q in range(X.shape[0]):
                        for hf in range(2 * ct):
                            keys.setdefault(tuple(X[q, hf * 16 : hf * 16 + 16].tolist()), f"{q}.{hf}")
                    for t, col in [(0, 0), (0, ct), (0, 2 * ct)]:
                        line = []
                        for hf in range(2):
                            for r in range(TILE):
                                v = got[t * TILE + r, col * TILE + hf * 16 : col * TILE + hf * 16 + 16]
                                w_ = ref[t * TILE + r, col * TILE + hf * 16 : col * TILE + hf * 16 + 16]
                                k = "Z" if bool((v == 0).all()) else keys.get(tuple(v.tolist()), "?")
                                line.append(("" if torch.equal(v, w_) else "*") + k)
                        print(f"DBG {name} t={t} col={col}: " + " ".join(line))
        lines.append(
            f"  {name:<10} {'-' if exact is None else ('bit-exact vs torch' if exact else f'DIFFERS ({nbad} rows)')}"
        )
        with RESULTS.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        variant=name,
                        ct=ct,
                        W=W,
                        H=H,
                        rows=rows,
                        o_first=o_first,
                        reps=reps,
                        taps=n_taps,
                        dtype=str(dtype),
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
