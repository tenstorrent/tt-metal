# SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
# SPDX-License-Identifier: Apache-2.0

"""Isolated bench: conv2d_3x3's K-blocked fp32 packer-L1-accumulation matmul walk, by output-subblock schedule.

    SB_BENCH=1 SB_SHAPES=sem,sem_bm4 SB_VARIANTS=h1x4,h4x1 scripts/run_safe_pytest.sh --run-all \
        ttnn/ttnn/operations/conv2d_3x3/perf_experiments/matmul_subblock_shape/test_sb_bench.py::test_bench \
        --import-mode=prepend

Single core, compute only: in0 (bf8 tiles, or bf16 row-major im2col rows with SB_TILIZE=1) and in1 (bf8) are
resident in L1 shards; the PreKBlockFn pushes each K-block (and tilizes it when SB_TILIZE=1). Partials are an fp32
shard of exactly bm x bn tiles (the op's contract). Precision config = the op's on the focus case: LoFi,
fp32_dest_acc_en, approx off, half-sync DEST (4 fp32 tiles). Every variant must be bit-identical to the first
variant listed (the helper baseline); the baseline must match torch. Perf is printed, never asserted.
"""

import os

# `import ttnn` imports every module under ttnn/operations (pkgutil.walk_packages); only touch the environment when
# the bench was asked for.
if os.environ.get("SB_BENCH") == "1":
    os.environ.setdefault("TT_METAL_DEVICE_PROFILER", "1")
    os.environ.setdefault("TT_METAL_PROFILER_MID_RUN_DUMP", "1")
    os.environ.setdefault("TT_METAL_PROFILER_CPP_POST_PROCESS", "1")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "error")

import json
import socket
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
from loguru import logger

import ttnn

TILE = 32
HERE = Path(__file__).parent
KERNEL = HERE / "kernels" / "sb_walk_compute.cpp"
RESULTS = HERE / "results_bench.jsonl"
CB_IN0, CB_IN1, CB_RM, CB_PARTIALS = 0, 1, 2, 5
AICLK_MHZ = 1350
_DURATION_KEY = "DEVICE KERNEL DURATION [ns]"


@dataclass(frozen=True)
class Walk:
    name: str
    bm: int
    bk: int
    bn: int
    nkb: int
    m_iters: int = 2
    kb_store: int = 0  # distinct K-blocks resident (0 = nkb)

    @property
    def store(self):
        return self.kb_store or self.nkb

    @property
    def macs(self):
        return self.bm * self.bk * self.bn * self.nkb * self.m_iters


# Per-core walks. sem = semantic res2.fuse_conv.1 @ LoFi (bm 5, bk 8 = one tap, 9 K-blocks, bn 8, 2 M-blocks/core).
WALKS = {
    w.name: w
    for w in (
        Walk("sem", bm=5, bk=8, bn=8, nkb=9),
        # The same walk with 3 distinct K-blocks resident (cycled): fits L1 with bf16 row-major in0 (SB_TILIZE=1).
        Walk("sem_s3", bm=5, bk=8, bn=8, nkb=9, kb_store=3),
        Walk("sem_bm4", bm=4, bk=8, bn=8, nkb=9),
        Walk("sem_bm8", bm=8, bk=8, bn=8, nkb=9, m_iters=1, kb_store=3),
        # Whole K in one K-block: DEST accumulation, one fp32 pack per subblock (pure matmul rate by shape).
        Walk("wholek_bm4", bm=4, bk=72, bn=8, nkb=1, m_iters=2),
        Walk("wholek_bm5", bm=5, bk=72, bn=8, nkb=1, m_iters=2),
        Walk("sem_bm4_s3", bm=4, bk=8, bn=8, nkb=9, kb_store=3),
        # res5.1's N-slice walk per core: bm 7, whole K (18 tiles) in one block per owner slice, out_n 2, 8 slices.
        Walk("res5_slice", bm=7, bk=18, bn=2, nkb=1, m_iters=8),
    )
}


@dataclass(frozen=True)
class Variant:
    name: str
    impl: int  # 0 helper uniform, 1 raw row groups, 2 helper uniform with caller-owned partials
    sbh: int = 1
    sbw: int = 4
    groups: tuple = ()  # ((rows, h, w), ...); rows = -1 -> the rest of bm
    skip_compute: bool = False
    flags: int = 0  # raw walk: 1 = caller-owned partials (no per-K-block drain), 2 = matmul init once per M-block
    bf16_in0: bool = False  # tilize only: in0 tiles CB in bf16 instead of bf8_b (the tilize packs bf16)


def _v(name):
    """h<H>x<W>: helper uniform; o<H>x<W>: helper uniform, caller-owned partials; r<H>x<W>[+<H>x<W>...]: raw groups
    (each group's rows = its h, the last group takes the remaining rows). Suffixes: _own (raw: caller-owned
    partials), _init1 (raw: matmul init once per M-block), _skip (SKIP_COMPUTE)."""
    tokens = name.split("_")
    base, opts = tokens[0], set(tokens[1:])
    skip = "skip" in opts
    bf16_in0 = "bf16in0" in opts
    if base[0] in "ho":
        h, w = map(int, base[1:].split("x"))
        return Variant(name, 0 if base[0] == "h" else 2, h, w, skip_compute=skip, bf16_in0=bf16_in0)
    assert base.startswith("r"), name
    parts = [tuple(map(int, p.split("x"))) for p in base[1:].split("+")]
    groups = tuple((-1 if i == len(parts) - 1 else h, h, w) for i, (h, w) in enumerate(parts))
    flags = (1 if "own" in opts else 0) | (2 if "init1" in opts else 0)
    return Variant(name, 1, groups=groups, skip_compute=skip, flags=flags, bf16_in0=bf16_in0)


def _resolve_groups(v: Variant, bm):
    out, left = [], bm
    for rows, h, w in v.groups:
        rows = left if rows == -1 else rows
        if rows % h:
            raise ValueError(f"{v.name}: group rows {rows} not a multiple of h {h}")
        out.append((rows, h, w))
        left -= rows
    if left != 0:
        raise ValueError(f"{v.name}: groups do not cover bm {bm}")
    return out


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


def _descriptor(walk: Walk, v: Variant, tilize, in0, in1, partials):
    if v.impl in (0, 2):
        if walk.bm % v.sbh or walk.bn % v.sbw or v.sbh * v.sbw > 4:
            raise ValueError(f"{v.name}: bad uniform subblock for bm {walk.bm} bn {walk.bn}")
        groups = [(walk.bm, v.sbh, v.sbw)]
    else:
        groups = _resolve_groups(v, walk.bm)
        for rows, h, w in groups:
            if walk.bn % w or h * w > 4:
                raise ValueError(f"{v.name}: bad group subblock {h}x{w}")
    padded = groups + [(0, 1, 1)] * (3 - len(groups))
    ctas = [walk.bm, walk.bk, walk.bn, walk.nkb, walk.m_iters, v.impl, v.sbh, v.sbw, len(groups)]
    for g in padded:
        ctas += list(g)
    ctas.append(int(tilize))
    if (v.flags & 2) and (tilize or len(groups) != 1):
        raise ValueError(f"{v.name}: init-once needs one group and no tilize")
    ctas.append(v.flags)
    compute = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL),
        core_ranges=_core(),
        compile_time_args=ctas,
        defines=[("SKIP_COMPUTE", "1")] if v.skip_compute else [],
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=ttnn.MathFidelity.LoFi,
            fp32_dest_acc_en=True,
            math_approx_mode=False,
            dst_full_sync_en=False,
        ),
    )
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_IN1, in1),
        ttnn.cb_descriptor_from_sharded_tensor(CB_PARTIALS, partials),
    ]
    if tilize:
        rm = ttnn.cb_descriptor_from_sharded_tensor(CB_RM, in0)
        rm.format_descriptors = [ttnn.CBFormatDescriptor(buffer_index=CB_RM, data_format=ttnn.bfloat16, page_size=2048)]
        cbs.append(rm)
        fmt, page = (ttnn.bfloat16, 2048) if v.bf16_in0 else (ttnn.bfloat8_b, 1088)
        cbs.append(
            ttnn.CBDescriptor(
                total_size=walk.bm * walk.bk * page,
                core_ranges=_core(),
                format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=CB_IN0, data_format=fmt, page_size=page)],
            )
        )
    else:
        cbs.append(ttnn.cb_descriptor_from_sharded_tensor(CB_IN0, in0))
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


def _inputs(device, w: Walk, tilize):
    import torch

    torch.manual_seed(1234)
    a = torch.randn(w.store * w.bm * TILE, w.bk * TILE) * 0.5
    b = torch.randn(w.store * w.bk * TILE, w.bn * TILE) * 0.5
    if tilize:
        # bf8_b-representable per 16-wide face row (the op's im2col rows are untilized bf8_b input pixels, and a
        # face row of an im2col tile is the same 16 channels of one pixel), so the bf8_b re-pack is lossless.
        a = ttnn.to_torch(ttnn.from_torch(a, dtype=ttnn.bfloat8_b, layout=ttnn.TILE_LAYOUT)).float()
        tt_a = ttnn.from_torch(
            a.to(torch.bfloat16),
            dtype=ttnn.bfloat16,
            layout=ttnn.ROW_MAJOR_LAYOUT,
            device=device,
            memory_config=_shard(w.store * w.bm * TILE, w.bk * TILE),
        )
        a_used = a.to(torch.bfloat16).double()
    else:
        tt_a = ttnn.from_torch(
            a,
            dtype=ttnn.bfloat8_b,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=_shard(w.store * w.bm * TILE, w.bk * TILE),
        )
        a_used = ttnn.to_torch(tt_a).double()
    tt_b = ttnn.from_torch(
        b,
        dtype=ttnn.bfloat8_b,
        layout=ttnn.TILE_LAYOUT,
        device=device,
        memory_config=_shard(w.store * w.bk * TILE, w.bn * TILE),
    )
    return tt_a, tt_b, a_used, ttnn.to_torch(tt_b).double()


def _reference(w: Walk, a, b):
    """Every M-iteration computes the same block: K-block j uses stored block j % store."""
    import torch

    acc = torch.zeros(w.bm * TILE, w.bn * TILE, dtype=torch.float64)
    for j in range(w.nkb):
        i = j % w.store
        acc += a[i * w.bm * TILE : (i + 1) * w.bm * TILE] @ b[i * w.bk * TILE : (i + 1) * w.bk * TILE]
    return acc


def _pcc(ref, out):
    import torch

    return float(torch.corrcoef(torch.stack([ref.flatten(), out.flatten()]))[0, 1])


def _walks():
    return [WALKS[n] for n in os.environ.get("SB_SHAPES", "sem").split(",")]


def _variants():
    return [_v(n) for n in os.environ.get("SB_VARIANTS", "h1x4,r1x4,r4x1+1x4").split(",")]


@pytest.mark.skipif(os.environ.get("SB_BENCH") != "1", reason="bench: run with SB_BENCH=1")
def test_bench(device):
    import torch

    tilize = os.environ.get("SB_TILIZE", "0") == "1"
    tag = os.environ.get("SB_TAG", "")
    box, arch = socket.gethostname(), str(device.arch())
    lines = [
        f"matmul_subblock_shape bench  box={box} arch={arch} cores=1 LoFi fp32_dest half-sync tilize={int(tilize)}"
    ]
    failures = []
    for w in _walks():
        tt_a, tt_b, a, b = _inputs(device, w, tilize)
        ref = _reference(w, a, b)
        lines.append(f"  {w.name}: bm={w.bm} bk={w.bk} bn={w.bn} nkb={w.nkb} m_iters={w.m_iters} macs={w.macs}")
        base_out = None
        for v in _variants():
            try:
                partials = ttnn.allocate_tensor_on_device(
                    ttnn.Shape([w.bm * TILE, w.bn * TILE]),
                    ttnn.float32,
                    ttnn.TILE_LAYOUT,
                    device,
                    _shard(w.bm * TILE, w.bn * TILE),
                )
                desc = _descriptor(w, v, tilize, tt_a, tt_b, partials)
            except ValueError as e:
                lines.append(f"    {v.name:<16} n/a ({e})")
                continue
            _read_kernel_ns(device)  # flush anything earlier
            ttnn.generic_op([tt_a, tt_b, partials], desc)
            ns = _read_kernel_ns(device)
            out = ttnn.to_torch(partials).double()
            partials.deallocate()
            pcc = _pcc(ref, out)
            if v.skip_compute:
                exact = "skip"
            elif base_out is None:
                base_out = out
                exact = "ref"
                if pcc < 0.99:
                    failures.append(f"{w.name}/{v.name} pcc {pcc:.5f}")
            else:
                same = torch.equal(out, base_out)
                exact = "bit-exact" if same else f"DIFF(max {float((out - base_out).abs().max()):.3e})"
                if not same:
                    failures.append(f"{w.name}/{v.name} not bit-identical to {_variants()[0].name}")
            cyc = ns * AICLK_MHZ / 1000.0 / w.macs if ns else float("nan")
            lines.append(f"    {v.name:<16} {ns:>10.0f} ns  {cyc:6.2f} cyc/tile-MAC  pcc={pcc:.6f}  {exact}")
            with RESULTS.open("a") as f:
                f.write(
                    json.dumps(
                        dict(
                            tag=tag,
                            walk=w.__dict__,
                            variant=v.name,
                            tilize=tilize,
                            ns=ns,
                            cyc_per_mac=cyc,
                            pcc=pcc,
                            exact=exact,
                            box=box,
                        )
                    )
                    + "\n"
                )
        tt_a.deallocate()
        tt_b.deallocate()
    logger.info("\n" + "\n".join(lines))
    assert not failures, failures
