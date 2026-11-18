# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bench: conv2d_3x3's HEIGHT K-blocked walk vs an N-slice-outer whole-K walk (per-core compute only).

    HNS_BENCH=1 HNS_VARIANTS=kb,ns1_p,ns2_f scripts/run_safe_pytest.sh --run-all --import-mode=prepend \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/height_nslice_outer/test_hns_bench.py::test_bench

Single core, every operand resident in L1 (bench/hns_walk_compute.cpp). HNS_SHAPE (default sem) = one core's walk:
sem = semantic res2.fuse_conv.1 @ LoFi (bm 5, 9 K-blocks of bk 8, one tap each; kw = 1 K-blocks from a copied center
CB; N 8; 2 M-blocks); see SHAPES for the others. Bias + ReLU, LoFi, fp32 DEST, half-sync DEST. Variants:
    kb             the op's K-blocked walk (helper matmul_block, 1 x 4 subblocks, fp32 L1-acc) + its bias pass
    ns<s>_<b>[_w]  N-slice-outer, slice width s, bias b: p = fp32 partials + the bias pass per M-block, f = bias in
                   DEST per subblock, fr = f with the add at tf32, sp = per-slice bias pass. Subblocks: min(bm, 4) x 1
                   columns on the first rows + 1 x min(s, 4) on the tail rows, or _w: 1 x min(s, 4) rows; _n: odd
                   slices do not reverse the subblock order.
Every variant is checked against a float64 reference (PCC, mean / max abs error) and against the baseline's output
(bit-exact or the count of differing elements). Perf is printed, never asserted.
"""

import os

# `import ttnn` imports every module under ttnn/operations; only touch the environment when the bench was asked for.
if os.environ.get("HNS_BENCH") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import json
import socket
from pathlib import Path

import pytest
from loguru import logger

import ttnn

TILE = 32
HERE = Path(__file__).parent
KERNEL = HERE / "bench" / "hns_walk_compute.cpp"
RESULTS = HERE / "results_bench.jsonl"
CB_IN0, CB_W, CB_RM, CB_CTR, CB_BIAS, CB_PART, CB_OUT, CB_BIASF = range(8)
AICLK_MHZ = 1350
# Per-core walks: (bm, bk, N, K-blocks, M-blocks per core, center layout, tilize width, K-blocked subblock (h, w)).
SHAPES = {
    # semantic res2.fuse_conv.1 @ LoFi: one tap per K-block, kw = 1 K-blocks from the center CB.
    "sem": (5, 8, 8, 9, 2, True, 8, (1, 4)),
    # the same at bm 2 (5 M-blocks per core), the bm the N-slice walk's L1 would allow at slice 4.
    "sem_bm2": (2, 8, 8, 9, 5, True, 8, (1, 4)),
    # one M-block per core, one kernel row per K-block (every K-block tilized here).
    "r3fc0": (3, 30, 8, 3, 1, False, 10, (1, 4)),
    "r3fc1": (3, 24, 8, 3, 1, False, 8, (1, 4)),
    "res3x": (3, 12, 4, 3, 1, False, 4, (1, 4)),
    "res2": (10, 6, 2, 3, 1, False, 2, (2, 2)),
}
SHAPE = os.environ.get("HNS_SHAPE", "sem")
BM, BK, N, NKB, M_ITERS, CENTER, TW, KB_SB = SHAPES[SHAPE]
K = BK * NKB
MACS = BM * K * N * M_ITERS
BIAS_MODES = {"p": 0, "f": 1, "sp": 2, "fr": 3}
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


def _variant(name):
    if name == "kb":
        return dict(mode=0, s=N, bias=0, wide=False)
    parts = name.split("_")
    s = int(parts[0][2:])
    mode = 2 if "n" in parts[2:] else 1
    return dict(mode=mode, s=s, bias=BIAS_MODES[parts[1]], wide="w" in parts[2:])


def _core():
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))])


def _shard(h_rows, w_cols):
    return ttnn.create_sharded_memory_config(
        shape=(h_rows, w_cols),
        core_grid=_core(),
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )


def _cb(index, fmt, page, pages):
    return ttnn.CBDescriptor(
        total_size=page * pages,
        core_ranges=_core(),
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
    )


def _descriptor(v, t):
    ctas = [BM, BK, N, NKB, M_ITERS, v["mode"], v["s"], v["bias"], int(v["wide"]), int(CENTER), TW, *KB_SB]
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=ctas,
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    rm = ttnn.cb_descriptor_from_sharded_tensor(CB_RM, t["rm"])
    rm.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=CB_RM, data_format=ttnn.bfloat16, page_size=2048)]
    in0_tiles = BM * BK * ((6 if CENTER else NKB) if v["mode"] != 0 else 1)
    part_tiles = BM * v["s"] if v["bias"] == 2 else BM * N
    cbs = [
        rm,
        ttnn.cb_descriptor_from_sharded_tensor(CB_W, t["w"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_BIAS, t["bias"]),
        ttnn.cb_descriptor_from_sharded_tensor(CB_OUT, t["out"]),
        _cb(CB_IN0, ttnn.bfloat8_b, 1088, in0_tiles),
        _cb(CB_PART, ttnn.float32, 4096, part_tiles),
        _cb(CB_BIASF, ttnn.bfloat8_b, 1088, N),
    ]
    if CENTER:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_CTR, t["ctr"]))
    return ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)


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


def _bf8(x):
    return ttnn.to_torch(ttnn.from_torch(x, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()


def _data():
    import torch

    torch.manual_seed(1234)
    a = _bf8(torch.randn(BM * TILE, BK * TILE))  # one tilized K-block's im2col rows (every tilized K-block)
    ctr = _bf8(torch.randn(3 * BM * TILE, BK * TILE))  # the 3 center K-blocks (input tiles)
    w = _bf8(torch.randn(K * TILE, N * TILE) * 0.2)
    b = torch.zeros(TILE, N * TILE)
    b[0] = torch.randn(N * TILE)
    b = _bf8(b)
    # The tilize reads each tile-row of the row-major shard as bk / tw consecutive [32, tw * 32] tap blocks (the op's
    # tap-major im2col layout): the logical in0 rows are those blocks side by side.
    flat = a.flatten()
    a_log = torch.zeros_like(a)
    blk = TILE * TW * TILE
    for r in range(BM):
        for t in range(BK // TW):
            off = (r * BK + t * TW) * TILE * TILE
            a_log[r * TILE : (r + 1) * TILE, t * TW * TILE : (t + 1) * TW * TILE] = flat[off : off + blk].reshape(
                TILE, TW * TILE
            )
    cols = []
    for kb in range(NKB):
        cols.append(ctr[(kb // 3) * BM * TILE : (kb // 3 + 1) * BM * TILE] if CENTER and kb % 3 == 1 else a_log)
    full = torch.cat(cols, dim=1).double()
    ref = torch.relu(full @ w.double() + b[0].double())
    return a, ctr, w, b, ref


def _tensors(device, v, a, ctr, w, b):
    import torch

    s = v["s"]
    w_layout = w if v["mode"] == 0 else torch.cat([w[:, y * s * TILE : (y + 1) * s * TILE] for y in range(N // s)], 0)
    t = dict(
        rm=ttnn.from_torch(
            a.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=_shard(BM * TILE, BK * TILE),
        ),
        w=ttnn.from_torch(
            w_layout,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_shard(*w_layout.shape),
        ),
        bias=ttnn.from_torch(
            b, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard(*b.shape)
        ),
        out=ttnn.allocate_tensor_on_device(
            ttnn.Shape([M_ITERS * BM * TILE, N * TILE]),
            ttnn.bfloat8_b,
            ttnn.TILE_LAYOUT,
            device,
            _shard(M_ITERS * BM * TILE, N * TILE),
        ),
    )
    if CENTER:
        t["ctr"] = ttnn.from_torch(
            ctr, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard(*ctr.shape)
        )
    return t


@pytest.mark.skipif(os.environ.get("HNS_BENCH") != "1", reason="bench: run with HNS_BENCH=1")
def test_bench(device):
    import torch

    tag = os.environ.get("HNS_TAG", "")
    names = os.environ.get("HNS_VARIANTS", "kb,ns1_p,ns2_p,ns4_p").split(",")
    box, arch = socket.gethostname(), str(device.arch())
    a, ctr, w, b, ref = _data()
    ref2 = torch.cat([ref] * M_ITERS, 0)
    lines = [
        f"height_nslice_outer bench box={box} arch={arch} 1 core LoFi fp32_dest half-sync, shape={SHAPE} "
        f"bm={BM} bk={BK} N={N} nkb={NKB} m_iters={M_ITERS} macs={MACS}"
    ]
    base_out, failures = None, []
    for name in names:
        v = _variant(name)
        t = _tensors(device, v, a, ctr, w, b)
        try:
            desc = _descriptor(v, t)
        except ValueError as e:
            lines.append(f"  {name:<12} n/a ({e})")
            continue
        _read_kernel_ns(device)
        ttnn.generic_op(list(t.values()), desc)
        ns = _read_kernel_ns(device)
        out = ttnn.to_torch(t["out"]).double()
        for x in t.values():
            x.deallocate()
        pcc = float(torch.corrcoef(torch.stack([ref2.flatten(), out.flatten()]))[0, 1])
        err = (out - ref2).abs()
        if base_out is None:
            base_out = out
            exact = "ref"
        else:
            diff = int((out != base_out).sum())
            exact = "bit-exact" if diff == 0 else f"{diff} differ (max {float((out - base_out).abs().max()):.3e})"
        if pcc < 0.999:
            failures.append(f"{name} pcc {pcc:.6f}")
        cyc = ns * AICLK_MHZ / 1000.0 / MACS if ns else float("nan")
        lines.append(
            f"  {name:<12} {ns:>10.0f} ns  {cyc:6.2f} cyc/tile-MAC  pcc={pcc:.6f} mean_abs={float(err.mean()):.5f} "
            f"max_abs={float(err.max()):.4f}  {exact}"
        )
        with RESULTS.open("a") as f:
            f.write(
                json.dumps(
                    dict(
                        tag=tag,
                        shape=SHAPE,
                        variant=name,
                        ns=ns,
                        cyc_per_mac=cyc,
                        pcc=pcc,
                        mean_abs=float(err.mean()),
                        max_abs=float(err.max()),
                        exact=exact,
                        box=box,
                    )
                )
                + "\n"
            )
    logger.info("\n" + "\n".join(lines))
    print("\n".join(lines))
    assert not failures, failures
