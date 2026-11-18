# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bench: conv2d_3x3's N-slice-outer matmul walk, one slice per call (the op) vs two slices per call.

    PB_BENCH=1 PB_WALKS=res5_1 PB_VARIANTS=base,pair_il,pair_2call,pair_1cb,base_raw \
        scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/nslice_paired_slices/test_pair_bench.py::test_bench \
        --import-mode=prepend

Single core, compute only (kernels/pair_walk_compute.cpp): the [bm, K] bf8 in0 block and every slice's bf8 weights
are resident L1 shards; compute pushes each weight round itself right before it consumes it; the two fp32 slice CBs
(reader-sent / writer-sent) hold all their slices. Precision config = the op's res5 focus config: LoFi,
fp32_dest_acc_en, approx off, half-sync DEST (4 fp32 tiles). Every variant must be bit-identical to `base` (the op's
call); `base` must match torch. Perf is printed and logged, never asserted.
"""

import os

# `import ttnn` imports every module under ttnn/operations (pkgutil.walk_packages): only touch the environment when
# the bench was asked for.
if os.environ.get("PB_BENCH") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import json
import socket
from dataclasses import dataclass
from pathlib import Path

import pytest
from loguru import logger

import ttnn

TILE = 32
HERE = Path(__file__).parent
KERNEL = HERE / "kernels" / "pair_walk_compute.cpp"
RESULTS = HERE / "results_bench.jsonl"
CB_IN0, CB_W, CB_OUT_W, CB_OUT_PAIR, CB_OUT_R = 0, 1, 4, 5, 10
AICLK_MHZ = 1350
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


@dataclass(frozen=True)
class Walk:
    name: str
    bm: int
    K: int
    out_n: int
    nslices: int
    sb: tuple  # the op's slice subblock (_out_subblock(bm, out_n))
    pair_h: int  # pair subblock rows (the pair width 2 * out_n per row)

    @property
    def macs(self):
        return self.bm * self.K * self.out_n * self.nslices


# Per-core N-slice walks (one M-block). res5.1/.2: bm 7, C_in 64 per core (K 18), C_out 512 over 8 columns (out_n 2).
WALKS = {
    w.name: w
    for w in (
        Walk("res5_1", bm=7, K=18, out_n=2, nslices=8, sb=(1, 2), pair_h=1),
        # res5.0 (stride 2): bm 2, slice subblock already 2x2.
        Walk("res5_0", bm=2, K=18, out_n=2, nslices=8, sb=(2, 2), pair_h=1),
        # res4.x / aspp: out_n 1, bm 8 (4x1 slice subblocks); a pair is [K, 2] -> 2x2.
        Walk("res4", bm=8, K=9, out_n=1, nslices=8, sb=(4, 1), pair_h=2),
        Walk("aspp", bm=8, K=54, out_n=1, nslices=8, sb=(4, 1), pair_h=2),
    )
}

# name -> (impl, weights interleaved per pair)
VARIANTS = {
    "base": (0, False),
    "pair_il": (1, True),
    "pair_2call": (2, False),
    "pair_1cb": (3, True),
    "base_raw": (4, False),
}


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


def _to_dev(t, dtype, device):
    return ttnn.from_torch(
        t, dtype=dtype, layout=ttnn.TILE_LAYOUT, device=device, memory_config=_shard(t.shape[0], t.shape[1])
    )


def _alloc(rows, cols, device):
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([rows, cols]), ttnn.float32, ttnn.TILE_LAYOUT, device, _shard(rows, cols)
    )


def _run_variant(device, w: Walk, name, tt_a, host_weights):
    import torch

    impl, inter = VARIANTS[name]
    sw = w.out_n * TILE
    pairs = w.nslices // 2
    if impl in (1, 2, 3):
        sbh, sbw = w.pair_h, 2 * w.out_n
        if w.bm % sbh or sbh * sbw > 4 or (impl == 2 and sbh != 1):
            return None
    else:
        sbh, sbw = w.sb
    ctas = [w.bm, w.K, w.out_n, w.nslices, impl, sbh, sbw]
    # One weights layout resident at a time (L1).
    weights = _to_dev(host_weights[inter], ttnn.bfloat8_b, device)
    tensors = [tt_a, weights]
    cbs = [ttnn.cb_descriptor_from_sharded_tensor(CB_IN0, tt_a), ttnn.cb_descriptor_from_sharded_tensor(CB_W, weights)]
    if impl == 3:
        out_p = _alloc(pairs * w.bm * TILE, 2 * sw, device)
        tensors.append(out_p)
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_OUT_PAIR, out_p))
    else:
        out_r = _alloc(pairs * w.bm * TILE, sw, device)
        out_w = _alloc(pairs * w.bm * TILE, sw, device)
        tensors += [out_r, out_w]
        cbs += [
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT_R, out_r),
            ttnn.cb_descriptor_from_sharded_tensor(CB_OUT_W, out_w),
        ]
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
    desc = ttnn.ProgramDescriptor(kernels=[compute], semaphores=[], cbs=cbs)
    _read_kernel_ns(device)  # flush anything earlier
    ttnn.generic_op(tensors, desc)
    ns = _read_kernel_ns(device)
    # Canonical [bm, nslices * out_n] in walk order (slice j at columns j * out_n).
    out = torch.zeros(w.bm * TILE, w.nslices * sw, dtype=torch.float64)
    if impl == 3:
        o = ttnn.to_torch(out_p).double()
        for p in range(pairs):
            blk = o[p * w.bm * TILE : (p + 1) * w.bm * TILE]
            out[:, 2 * p * sw : (2 * p + 1) * sw] = blk[:, :sw]
            out[:, (2 * p + 1) * sw : (2 * p + 2) * sw] = blk[:, sw:]
        out_p.deallocate()
    else:
        o_r, o_w = ttnn.to_torch(out_r).double(), ttnn.to_torch(out_w).double()
        for j in range(w.nslices):
            src, i = (o_r, j // 2) if (j % 2 == 0 and j != w.nslices - 1) else (o_w, (j - 1) // 2)
            out[:, j * sw : (j + 1) * sw] = src[i * w.bm * TILE : (i + 1) * w.bm * TILE]
        out_r.deallocate()
        out_w.deallocate()
    weights.deallocate()
    return ns, out


def _pcc(ref, out):
    import torch

    return float(torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1])


@pytest.mark.skipif(os.environ.get("PB_BENCH") != "1", reason="bench: run with PB_BENCH=1")
def test_bench(device):
    import torch

    tag = os.environ.get("PB_TAG", "")
    walks = [WALKS[n] for n in os.environ.get("PB_WALKS", "res5_1").split(",")]
    names = [n for n in os.environ.get("PB_VARIANTS", ",".join(VARIANTS)).split(",") if n]
    box, arch = socket.gethostname(), str(device.arch())
    lines = [f"nslice_paired_slices bench  box={box} arch={arch} cores=1 LoFi fp32_dest half-sync"]
    failures = []
    for w in walks:
        torch.manual_seed(1234)
        sw = w.out_n * TILE
        a = torch.randn(w.bm * TILE, w.K * TILE) * 0.5
        b = torch.randn(w.K * TILE, w.nslices * sw) * 0.5
        tt_a = _to_dev(a, ttnn.bfloat8_b, device)
        a_used = ttnn.to_torch(tt_a).double()
        stacked = torch.cat([b[:, j * sw : (j + 1) * sw] for j in range(w.nslices)], dim=0)
        inter = torch.cat([b[:, 2 * p * sw : (2 * p + 2) * sw] for p in range(w.nslices // 2)], dim=0)
        # bf8_b quantizes per tile, and both layouts move whole tiles: the same values either way.
        st = ttnn.to_torch(ttnn.from_torch(stacked, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).double()
        b_used = torch.cat([st[j * w.K * TILE : (j + 1) * w.K * TILE] for j in range(w.nslices)], dim=1)
        ref = a_used @ b_used
        lines.append(f"  {w.name}: bm={w.bm} K={w.K} out_n={w.out_n} slices={w.nslices} macs={w.macs} sb={w.sb}")
        base_out = None
        for name in names:
            res = _run_variant(device, w, name, tt_a, {False: stacked, True: inter})
            if res is None:
                lines.append(f"    {name:<12} n/a")
                continue
            ns, out = res
            pcc = _pcc(ref, out)
            if base_out is None:
                base_out, exact = out, "ref"
                if pcc < 0.999:
                    failures.append(f"{w.name}/{name} pcc {pcc:.5f}")
            else:
                same = torch.equal(out, base_out)
                exact = "bit-exact" if same else f"DIFF(max {float((out - base_out).abs().max()):.3e})"
                if not same:
                    failures.append(f"{w.name}/{name} not bit-identical to {names[0]}")
            cyc = ns * AICLK_MHZ / 1000.0 / w.macs if ns else float("nan")
            lines.append(f"    {name:<12} {ns:>10.0f} ns  {cyc:6.2f} cyc/tile-MAC  pcc={pcc:.6f}  {exact}")
            with RESULTS.open("a") as f:
                rec = dict(tag=tag, walk=w.name, variant=name, ns=ns, cyc_per_mac=cyc, pcc=pcc, exact=exact, box=box)
                f.write(json.dumps(rec) + "\n")
        tt_a.deallocate()
    logger.info("\n" + "\n".join(lines))
    assert not failures, failures
