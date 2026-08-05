# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: does the rms_norm cross-core COMBINE go faster on TWO NoCs?

NOT the op.  Reconstructs the combine (and nothing else) as a standalone
``ttnn.ProgramDescriptor``, exactly the concept isolation
``perf_experiments/hierarchical_gather_r2`` established:

  * every core of a group starts with ``num_rows`` fp32 partial tiles already resident in
    its own L1 (a HEIGHT-sharded fp32 input tensor -- pass A is deliberately NOT modelled,
    so the measured delta is attributable to the collective alone);
  * every core must end with the group's finalized stat
    ``rsqrt(sum_group(partial) * (1/W) + eps)`` in ``cb_row_final``, which is backed on the
    output shard -- so the result IS the output tensor.

THE IDEA
--------
On the focus shape the input is a native zero-copy L1 shard, so the op's READER has
essentially nothing to do (``reader_read_x`` = 56 ns for the whole kernel): NCRISC / NoC0
is idle for ~31 us while BRISC / NoC1 issues every byte and every synchronization of the
combine.  Put the idle half of the hardware to work.

Two levers, and their composition, behind ONE set of compile-time knobs:

  (a) SPLIT the gather ship across both NoCs.  A member's partial for a block is
      BLOCK_ROWS tiles x GATHER_FACES=2 face writes = 16 writes of 1024 B per round; issue
      some of them from the reader kernel (NoC0) and the rest from the writer (NoC1).
      ``SPLIT_MODE``: 1 = by ROW (reader takes rows [0, rows*NUM/DEN)), 2 = by FACE (reader
      takes face 0, writer face 2 -- the only split a BLOCK_ROWS=1 round has), 3 = the
      reader takes the WHOLE ship.
  (b) MOVE whole sub-stages to the reader: ``ZERO_R`` moves the root's one-time
      ``writer_gather_zero`` boot, ``MCAST_R`` moves the stat multicast (sender AND
      receiver pipes) onto NoC0.

THE HAPPENS-BEFORE EDGE THE SPLIT NEEDS (and why it is sound)
------------------------------------------------------------
``Semaphore::up(value)`` is a NON-ATOMIC local read-modify-write, so the arrival signal
must be raised EXACTLY ONCE per member per round, and only after BOTH halves of that
member's partial have landed.  A ``noc_async_write_barrier()`` on the reader flushes only
NoC0's outstanding writes and a barrier on the writer only NoC1's -- neither sees the
other.  So the split builds an explicit edge out of two single-producer/single-consumer
token CBs on the SAME core:

    writer  cb_push_back(tok_w2r)          "the source is readable, ship your share"
    reader  cb_wait_front(tok_w2r) -> ship its share -> noc_async_write_barrier()
            -> cb_push_back(tok_r2w)       "my NoC0 half has LANDED"
    writer  ships its own share -> noc_async_write_barrier() -> cb_wait_front(tok_r2w)
            -> Semaphore::up(root)         ONE signal, after both barriers

The writer owns the signal in every mode (one definition), so no mode can raise it twice
or raise it early.  ``ZERO_R`` uses the same ``tok_r2w`` CB for its one-time
"the unshipped faces are zeroed" edge, consumed by the root writer before its FIRST
``cb_push_back(cb_partials_gathered)`` -- the push is what releases the ring to compute, so
that is the only place the zeroing has to be visible.  Both producers push in the order the
consumer waits, so the shared FIFO cannot reorder.

``base`` and ``tok`` bracket the cost of that edge: ``tok`` is the baseline WITH the token
ping-pong compiled in and the writer still doing the whole ship, so
(``tok`` - ``base``) is the pure overhead and (``base`` - ``s50``) is the net win.

MODELLING A BUSY NoC0 (the carve-out probe)
-------------------------------------------
The idea's premise is an IDLE NoC0.  ``rd_tiles`` gives the reader a synthetic DRAM read
load of ``rd_tiles`` bf16 tiles per round, issued BEFORE its share of the ship (the op's
order: stage x for this block, then ship this block's partial).  ``rd_tiles = 0`` is the
native-shard focus case; a large ``rd_tiles`` is the reader-fed INTERLEAVED case where NoC0
is emphatically not idle and the idea is expected to REGRESS.

Precision contract (FIXED, never a lever): fp32 partials, HiFi2, fp32_dest_acc_en=False,
math_approx_mode=False -- identical for every variant, baseline included.  Every un-ablated
variant is gated on pcc AND rel-RMS AND BIT-EXACTNESS against `base`: this idea only changes
WHICH RISC issues a write, so anything but bit-identical output is a race, not a rounding
difference.  All 25 variants are bit-identical in every regime measured.

===========================================================================================
MEASURED -- blackhole p150b, 1350 MHz, one fresh-cache profiled run per point
(DEVICE KERNEL DURATION [ns]); the focus points are 3x medians (spread < 0.5%).
Baseline root chain = FOLD_STYLE 1, the op's CURRENT fused pairwise DEST fold (Perf 2 / D22),
which puts the bench's fold payload at 11835 ns against the op's own measured 11297 ns.
===========================================================================================
FOCUS GEOMETRY (1,1,8192,1024) BLOCK_SHARDED, G=8, 8 groups, BLOCK_ROWS=8, 64 cores

                       FULL 30233 ns          FOLD-ABLATED 18398 ns
                       (the op's real         (transport + sync only;
                        critical path)         cf. the op's 16097 ns residual)
  sf   face split          27653  1.093x          15944  1.154x
  sfm  ... members only    28005  1.080x          16259  1.132x
  s50  50/50 row split     ~1.06x  (D16 base)     15902  1.158x
  s25 / s75                --                     1.054x / 1.084x
  s100 whole ship -> NoC0  --                     18231  1.010x   (FLAT)
  tok  the edge alone      --                     18653  0.987x   (the edge costs 1.3%)
  mcs  root SEND -> NoC0   32933  0.918x          13501  1.363x
  mc   whole mcast -> NoC0 33133  0.912x          12704  1.448x
  sfm_mc  both levers      31281  0.966x          12828  1.434x
  z    gather-zero -> NoC0 33116  0.913x          21389  0.860x   (REGRESSION)

THE THREE THINGS THIS MEASURES
  1  The win is USING BOTH NoCs, not offloading to one.  `s100` (the whole ship issued from
     the reader) is FLAT at 1.010x while a 50/50 split is 1.15x; `s25`/`s75` sit between.
  2  The gather split and the multicast move DO NOT COMPOSE, and the reason is an edge, not
     a resource: `s50_mcs` measured 0.977x -- WORSE than either lever alone -- because the
     ROOT's `landed` token makes its next round's gather wait on the multicast send the
     mcast move had just decoupled.  Giving the root's own (purely local) slot write back to
     the writer (`sr=0`, SPLIT_ROOT=0) fixes it exactly: 0.977x -> 1.426x.
  3  The multicast move is the biggest TRANSPORT win (1.45x of the residual) and a NET
     REGRESSION (0.91x) with the op's real root chain in place -- the multicast on NoC1 was
     already hidden behind that chain, and NCRISC issues it more slowly (the same asymmetry
     that makes `z` a 0.86x regression even in the transport-only measurement).  Worth
     re-measuring only if the root chain shrinks.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import NamedTuple

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_PARTIALS_GATHERED = 11
CB_ROW_STAT = 14
CB_STAT_HANDOFF = 15
CB_ROW_FINAL = 16
CB_TOK_W2R = 17
CB_TOK_R2W = 18
CB_LOAD = 19
CB_TOK_ZERO = 20

TILE = 32
FP32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048
GATHER_FACES = 2  # the op's D13 compact gather: faces 0 and 2 only
ROW_STAT_DEPTH = 2  # the op's CB_ROW_STAT_DEPTH
TOK_PAGE = 32  # a pure signal; the page carries nothing
TOK_DEPTH = 4  # >= 2 outstanding tokens per direction is possible; 4 leaves margin
LOAD_PAGES = 4  # scratch ring for the synthetic NoC0 load (no consumer)
LOAD_TILES_TOTAL = 512  # tiles in the DRAM load tensor

# SPLIT_MODE
SPLIT_NONE = 0
SPLIT_ROWS = 1
SPLIT_FACES = 2
SPLIT_ALL = 3


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _cb(index, page_size, num_pages, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


# ---------------------------------------------------------------------------
# geometry -- the two placements the op actually builds (carried from
# hierarchical_gather_r2/bench.py so both bake-offs measure the same geometries)
# ---------------------------------------------------------------------------


class Geometry(NamedTuple):
    name: str
    group_size: int
    num_groups: int
    gx: int
    gy: int
    core_range_set: object
    cores: tuple
    groups: tuple
    inactive: frozenset
    per_row: bool  # Mcast1D PerRow (one group per grid row) vs Mcast2D


def build_geometry(device, *, group_size, num_groups, box_w=None):
    grid = device.compute_with_storage_grid_size()
    if num_groups > 1:
        assert group_size <= grid.x, f"{group_size} cores per group exceeds grid.x={grid.x}"
        assert num_groups <= grid.y
        crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(group_size - 1, num_groups - 1))])
        cores = list(ttnn.corerange_to_cores(crs, None, True))
        groups = tuple(tuple(c for c in cores if c.y == g) for g in range(num_groups))
        return Geometry(
            name=f"g{group_size}_ng{num_groups}",
            group_size=group_size,
            num_groups=num_groups,
            gx=group_size,
            gy=1,
            core_range_set=crs,
            cores=tuple(cores),
            groups=groups,
            inactive=frozenset(),
            per_row=True,
        )
    box_w = box_w or min(grid.x, group_size)
    rows = (group_size + box_w - 1) // box_w
    assert rows <= grid.y
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(box_w - 1, rows - 1))])
    cores = list(ttnn.corerange_to_cores(crs, None, True))
    group = tuple(cores[:group_size])
    inactive = frozenset((c.x, c.y) for c in cores[group_size:])
    return Geometry(
        name=f"g{group_size}_box{box_w}",
        group_size=group_size,
        num_groups=1,
        gx=box_w,
        gy=rows,
        core_range_set=crs,
        cores=tuple(cores),
        groups=(group,),
        inactive=inactive,
        per_row=False,
    )


# ---------------------------------------------------------------------------
# the variant menu
# ---------------------------------------------------------------------------


class Variant(NamedTuple):
    label: str
    zero_r: int  # move the root's gather-zero boot to the reader
    split_mode: int
    num: int  # SPLIT_ROWS: reader ships rows [0, rows*num/den)
    den: int
    mcast_r: int  # 0 all on writer / 1 both pipe faces on reader / 2 the root's SEND only
    force_tok: int  # compile the token ping-pong in even at SPLIT_NONE (overhead control)
    split_root: int  # 1 the root splits its own slot too / 0 only the MEMBERS split

    @property
    def tok_round(self) -> int:
        """The per-round GO / LANDED ping-pong (the split needs it; `tok` compiles it in
        with the writer still shipping everything, as the pure-overhead control)."""
        return 1 if (self.split_mode != SPLIT_NONE or self.force_tok) else 0

    @property
    def tok_pair(self) -> int:
        """Whether the per-round GO/LANDED FIFOs must EXIST.  Beyond the split, MCAST_R == 1
        needs the round gate the moved receive took away from the writer (MCAST_R == 2 does
        not -- the member keeps its own receive, so it keeps its own gate)."""
        return 1 if (self.tok_round or self.mcast_r == 1) else 0

    @property
    def tok_zero(self) -> int:
        """ZERO_R's one-time edge rides its OWN one-page FIFO.  Sharing tok_r2w with the
        per-round LANDED signal would let one wait consume the other's token, and since both
        are pure signals the swap is SILENT: the root would publish a gather ring whose own
        slot the reader had not finished writing."""
        return self.zero_r


def V(label, *, z=0, mode=SPLIT_NONE, num=0, den=1, mc=0, tok=0, sr=1):
    return Variant(label, z, mode, num, den, mc, tok, sr)


# The honest baseline FIRST: the op's current approach -- every byte and every
# synchronization of the combine issued by the writer on NoC1.
BASELINE = V("base")

FULL_MENU = [
    BASELINE,
    V("tok", tok=1),  # baseline + the token edge = the edge's pure cost
    V("z", z=1),  # (b) gather_zero boot -> NoC0
    V("s25", mode=SPLIT_ROWS, num=1, den=4),  # (a) 1/4 of the rows -> NoC0
    V("s50", mode=SPLIT_ROWS, num=1, den=2),  # (a) half the rows -> NoC0
    V("s75", mode=SPLIT_ROWS, num=3, den=4),
    V("sf", mode=SPLIT_FACES),  # (a) by FACE: 1 of the 2 writes per row -> NoC0
    V("s100", mode=SPLIT_ALL),  # (b) the WHOLE ship -> NoC0
    V("mc", mc=1),  # (b) the multicast, BOTH pipe faces -> NoC0 (needs a round gate)
    V("mcs", mc=2),  # (b) only the ROOT's SEND -> NoC0 -- NO new edge anywhere
    V("z_s50", z=1, mode=SPLIT_ROWS, num=1, den=2),
    V("z_sf", z=1, mode=SPLIT_FACES),
    V("s50_mcs", mode=SPLIT_ROWS, num=1, den=2, mc=2),
    V("sf_mcs", mode=SPLIT_FACES, mc=2),
    V("z_s50_mcs", z=1, mode=SPLIT_ROWS, num=1, den=2, mc=2),
    V("z_sf_mcs", z=1, mode=SPLIT_FACES, mc=2),
    V("z_s100_mc", z=1, mode=SPLIT_ALL, mc=1),  # the whole combine on NoC0
    # ---- MEMBERS-ONLY split (sr=0): the root keeps its own local slot write on the writer,
    # so its `landed` token cannot re-couple the next round's gather to the mcast send.
    V("s50m", mode=SPLIT_ROWS, num=1, den=2, sr=0),
    V("sfm", mode=SPLIT_FACES, sr=0),
    V("z_mcs", z=1, mc=2),  # does ZERO_R help or hurt with the mcast already moved?
    V("s50m_mcs", mode=SPLIT_ROWS, num=1, den=2, mc=2, sr=0),
    V("sfm_mcs", mode=SPLIT_FACES, mc=2, sr=0),
    V("s50m_mc", mode=SPLIT_ROWS, num=1, den=2, mc=1, sr=0),
    V("sfm_mc", mode=SPLIT_FACES, mc=1, sr=0),
    V("z_sfm_mc", z=1, mode=SPLIT_FACES, mc=1, sr=0),
]

# The menu the domain sweep runs (device time is finite; these are the load-bearing points).
SWEEP_MENU = [
    BASELINE,
    V("z", z=1),  # the measured regression -- re-checked in every regime
    V("sf", mode=SPLIT_FACES),  # split incl. the root (best split when the mcast stays)
    V("mc", mc=1),  # the whole multicast on NoC0
    V("mcs", mc=2),  # only the root's SEND on NoC0 (no new edge)
    V("sfm_mc", mode=SPLIT_FACES, mc=1, sr=0),  # both levers, members-only split
]


def cb_pages(group_size, block_rows):
    """Per-core L1 page counts of the combine's OWN fp32 CBs (cb_row_final is backed on
    the output shard).  IDENTICAL for every variant -- the idea moves work between RISCs,
    it does not change the data layout."""
    return {
        "partials_gathered": group_size * block_rows,
        "row_stat": ROW_STAT_DEPTH * block_rows,
        "stat_handoff": block_rows,
    }


def l1_bytes(group_size, block_rows, variant):
    n = sum(cb_pages(group_size, block_rows).values()) * FP32_TILE_BYTES
    if variant.tok_pair:
        n += 2 * TOK_DEPTH * TOK_PAGE
    if variant.tok_zero:
        n += TOK_PAGE
    return n


# ---------------------------------------------------------------------------
# program
# ---------------------------------------------------------------------------


def _mcast(device, geo, variant):
    # The pipes must run on the NoC the kernel that owns them uses: NoC0 (reader,
    # RISCV_1) when the multicast is moved, NoC1 (writer, RISCV_0) otherwise.
    # The SENDER's rect must be ordered for the NoC the sending kernel runs on: NoC0 in
    # modes 1 and 2 (reader), NoC1 in mode 0 (writer).  A mode-2 RECEIVER still runs on the
    # writer / NoC1, which is fine -- a receiver never multicasts, so it never touches the
    # rect; it only polls a local flag and acks with an atomic on whatever NoC it is on.
    noc = ttnn.NOC.NOC_0 if variant.mcast_r else ttnn.NOC.NOC_1
    cfg = ttnn.McastConfig(noc=noc, handshake=True, base_sem_id=0)
    if geo.per_row:
        return ttnn.Mcast1D(device, geo.core_range_set, ttnn.Mcast1DShape.PerRow, 0, cfg)
    root = geo.groups[0][0]
    return ttnn.Mcast2D(device, geo.core_range_set, ttnn.CoreCoord(root.x, root.y), cfg, geo.group_size - 1)


def build_program(
    device,
    x,
    out,
    load_t,
    geo,
    *,
    variant,
    block_rows,
    num_rows,
    rd_tiles,
    inv_w,
    eps,
    compute_config,
    ablate=0,
    fold_style=1,
):
    ft = ttnn.tile_size(ttnn.float32)
    all_cores = geo.core_range_set
    mcast = _mcast(device, geo, variant)
    sem1 = mcast.next_base_sem_id()

    G = geo.group_size
    pages = cb_pages(G, block_rows)
    cbs = [
        _cb(CB_PARTIALS_GATHERED, ft, pages["partials_gathered"], ttnn.float32, all_cores),
        _cb(CB_ROW_STAT, ft, pages["row_stat"], ttnn.float32, all_cores),
        _cb(CB_STAT_HANDOFF, ft, pages["stat_handoff"], ttnn.float32, all_cores),
        ttnn.cb_descriptor_from_sharded_tensor(CB_ROW_FINAL, out),
    ]
    if variant.tok_pair:
        cbs.append(_cb(CB_TOK_W2R, TOK_PAGE, TOK_DEPTH, ttnn.uint32, all_cores))
        cbs.append(_cb(CB_TOK_R2W, TOK_PAGE, TOK_DEPTH, ttnn.uint32, all_cores))
    if variant.tok_zero:
        cbs.append(_cb(CB_TOK_ZERO, TOK_PAGE, 1, ttnn.uint32, all_cores))
    if rd_tiles:
        cbs.append(_cb(CB_LOAD, ttnn.tile_size(ttnn.bfloat16), LOAD_PAGES, ttnn.bfloat16, all_cores))

    virt = {}

    def v(core):
        key = (core.x, core.y)
        if key not in virt:
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, core.y))
            virt[key] = (c.x, c.y)
        return virt[key]

    x_addr = x.buffer_address()
    load_addr = load_t.buffer_address()

    rd_args = {}
    wr_args = {}
    cp_args = {}
    for group in geo.groups:
        for slot, core in enumerate(group):
            is_root = 1 if slot == 0 else 0
            mc_rt = list(mcast.runtime_args(core))
            wr_args[(core.x, core.y)] = [x_addr, num_rows, is_root, slot] + mc_rt
            rd_args[(core.x, core.y)] = [x_addr, num_rows, is_root, slot, load_addr] + mc_rt
            cp_args[(core.x, core.y)] = [num_rows, is_root, slot]

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core in geo.cores:
        key = (core.x, core.y)
        if key not in wr_args:
            # INACTIVE: in the mcast box, in no group.  num_rows == 0 makes every kernel
            # return before touching anything (the op's contract).
            mc_rt = list(mcast.runtime_args(core))
            wr_args[key] = [x_addr, 0, 0, 0] + mc_rt
            rd_args[key] = [x_addr, 0, 0, 0, load_addr] + mc_rt
            cp_args[key] = [0, 0, 0]
        reader_rt[core.x][core.y] = rd_args[key]
        writer_rt[core.x][core.y] = wr_args[key]
        compute_rt[core.x][core.y] = cp_args[key]

    common_ct = [
        G,
        block_rows,
        sem1,
        GATHER_FACES,
        variant.zero_r,
        variant.split_mode,
        variant.num,
        variant.den,
        variant.mcast_r,
        variant.tok_round,
        variant.split_root,
    ]
    writer_ct = list(common_ct)
    assert len(writer_ct) == 11, "bench_writer.cpp expects McastArgs<11, 4>()"
    writer_ct.extend(mcast.compile_time_args())

    reader_ct = list(common_ct) + [rd_tiles, LOAD_TILES_TOTAL]
    assert len(reader_ct) == 13, "bench_reader.cpp expects McastArgs<13, 5>()"
    reader_ct.extend(mcast.compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(load_t).get_compile_time_args())

    compute_ct = [G, block_rows, _f32_bits(inv_w), _f32_bits(eps), ablate, fold_style]

    semaphores = list(mcast.owned_semaphores())
    semaphores.append(ttnn.SemaphoreDescriptor(id=sem1, core_ranges=all_cores, initial_value=0))

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "bench_reader.cpp"),
                core_ranges=all_cores,
                compile_time_args=reader_ct,
                runtime_args=reader_rt,
                config=ttnn.ReaderConfigDescriptor(),  # NCRISC / NoC0 -- the idle half
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "bench_writer.cpp"),
                core_ranges=all_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),  # BRISC / NoC1, like the op's combine
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "bench_compute.cpp"),
                core_ranges=all_cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_config,
            ),
        ],
        semaphores=semaphores,
        cbs=cbs,
    )
