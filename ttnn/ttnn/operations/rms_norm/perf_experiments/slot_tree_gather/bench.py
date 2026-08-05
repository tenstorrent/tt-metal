# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off: the SLOT-TREE half of hierarchical gather, at m = 1.

NOT the op.  Reconstructs the rms_norm cross-core width COMBINE and nothing else as a
standalone ``ttnn.ProgramDescriptor``:

  * every core of a group starts with ``num_rows`` fp32 partial tiles already resident in
    its own L1 (a HEIGHT-sharded fp32 input tensor -- pass A is deliberately NOT modelled,
    so the measured delta is attributable to the collective alone);
  * every core must end with the group's finalized stat
    ``rsqrt(sum_group(partial) * (1/W) + eps)`` in ``cb_row_final``, which is backed on the
    output shard -- so the result IS the output tensor.

WHY THIS DIR EXISTS, given ``hierarchical_gather_r2`` next door
---------------------------------------------------------------
r2 measured a (K slot chunks x m row-subset gatherers) grid and its winner was the ROW
SPLIT (m = BLOCK_ROWS).  That half is mutually exclusive with ``compact_partial_transpose``
(compaction collapses a sender's BLOCK_ROWS partials into ONE tile, deleting the row axis
the split parallelises over).  This dir isolates the OTHER half -- the tree over the SLOT
axis, with the row split OFF (m == 1) -- because GROUP_SIZE is an axis compaction does not
touch, so a slot tree COMPOSES with it.

Two things also changed under r2's feet and are re-derived here, not inherited:

  1. THE BASELINE.  r2 measured against the post-Perf-1 root chain (D16 packer-accumulate
     fold + D19 separate finalize).  Perf 2 landed **D22**, which folds a row's partials
     PAIRWISE IN DEST over ``GATHER_SLOTS`` (== GROUP_SIZE rounded up to even), finalizes in
     the same DEST window and packs ONCE -- measured 2.18x cheaper than what r2 called
     "flat".  A tree that trades fold work for a NoC hop has to beat the CHEAP fold, so the
     baseline here is D22 carried verbatim.
  2. THE TREE ITSELF.  r2's K > 1 was a fixed TWO-level shape (fold M = G/K, forward, fold
     K).  Here the tree is a general arity list ``F = (f0, f1, ...)``, so 3- and 4-level
     trees (``log_k(GROUP_SIZE)`` levels, which is what the idea actually names) are
     measurable instead of assumed.

Variants
--------
``flat``           the op's CURRENT approach (honest baseline): one root gathers
                   GATHER_SLOTS partials per tile-row into a row-major window, folds +
                   finalizes them in ONE DEST window per row (D22), multicasts back.
``tree_<f0>x<f1>`` the candidate: L = len(F) levels of contiguous slot chunks.  ``F`` with
                   one element is ``flat`` through the generic path -- the overhead control.

TREE GEOMETRY (one definition, mirrored in both kernels)
--------------------------------------------------------
``stride[0] = 1``, ``stride[l+1] = stride[l] * f_l``.
  * a core is a PARTICIPANT at level ``l`` iff ``slot % stride[l] == 0``;
  * it is the GATHERER of its level-``l`` chunk iff ``slot % stride[l+1] == 0``;
  * the chunk gathered by core ``p`` at level ``l`` is ``{p + j*stride[l] : j < f_l}``, so
    its real member count is ``min(f_l, ceil((GROUP_SIZE - p) / stride[l]))`` and the
    remaining ``f_l``-slots (plus the evenness pad) are boot-zeroed and contribute +0.0.
``prod(F) >= GROUP_SIZE`` is asserted, which is exactly what makes slot 0 -- the multicast
root -- the unique gatherer at the LAST level.  So the mcast is untouched by the tree.

Precision contract (FIXED, never a lever): fp32 partials, HiFi2, fp32_dest_acc_en=False,
math_approx_mode=False -- identical for every variant.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import NamedTuple

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# per-level gather rings (levels 0..3)
CB_G0 = 11
CB_NODE_OUT = 15  # an interior gatherer's folded (NOT finalized) partial
CB_STAT_HANDOFF = 16
CB_ROW_FINAL = 17

MAX_LEVELS = 4
TILE = 32
FP32_TILE_BYTES = 4096
GATHER_FACES = 2  # the op's D13 compact gather: faces 0 and 2 only, at level 0

V_FLAT, V_TREE = 0, 1


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _cb(index, page_size, num_pages, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


# ---------------------------------------------------------------------------
# geometry (the two placements the op actually builds)
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
# the arity list
# ---------------------------------------------------------------------------


def gather_slots(group_size: int) -> int:
    """The op's D22 landing stride: GROUP_SIZE rounded UP TO EVEN, so the root's pairwise
    DEST walk always has an even count to halve."""
    return group_size + group_size % 2


def level_slots(f: int) -> int:
    """Same evenness rule, per tree level."""
    return f + f % 2


def legal(group_size: int, arity) -> bool:
    """Is this arity list expressible?

    * every level must actually fold (``f >= 2``);
    * ``prod(F) >= GROUP_SIZE`` -- otherwise slot 0 is not the unique last-level gatherer
      and the group never reduces to one stat;
    * at most MAX_LEVELS levels (the kernels carry four compile-time slots);
    * no level may be redundant: if ``prod(F[:-1]) >= GROUP_SIZE`` the last level gathers a
      single real member and buys nothing but a hop.
    """
    if not arity or len(arity) > MAX_LEVELS:
        return False
    if any(f < 2 for f in arity):
        return False
    prod = 1
    for i, f in enumerate(arity):
        if i > 0 and prod >= group_size:
            return False  # this level is redundant
        prod *= f
    return prod >= group_size


def cb_pages(group_size: int, block_rows: int, arity, variant: int) -> dict:
    """Per-core L1 page counts of the combine's OWN CBs (fp32 tiles), excluding
    cb_row_final (backed on the output shard in both variants).  Every gather ring is
    declared on EVERY core of the program -- that is what lets a sender compute the landing
    address locally -- so this is the honest per-core figure for both variants."""
    if variant == V_FLAT:
        return {"gather0": gather_slots(group_size) * block_rows, "stat_handoff": block_rows}
    out = {f"gather{l}": level_slots(f) * block_rows for l, f in enumerate(arity)}
    out["stat_handoff"] = block_rows
    if len(arity) > 1:
        out["node_out"] = block_rows
    return out


def l1_bytes(group_size: int, block_rows: int, arity, variant: int) -> int:
    return sum(cb_pages(group_size, block_rows, arity, variant).values()) * FP32_TILE_BYTES


def num_semaphores(arity, variant: int) -> int:
    """Semaphores the combine needs BEYOND the multicast helper's two."""
    return 1 if variant == V_FLAT else len(arity)


# ---------------------------------------------------------------------------
# program
# ---------------------------------------------------------------------------


def _mcast(device, geo):
    cfg = ttnn.McastConfig(noc=ttnn.NOC.NOC_1, handshake=True, base_sem_id=0)
    if geo.per_row:
        return ttnn.Mcast1D(device, geo.core_range_set, ttnn.Mcast1DShape.PerRow, 0, cfg)
    root = geo.groups[0][0]
    return ttnn.Mcast2D(device, geo.core_range_set, ttnn.CoreCoord(root.x, root.y), cfg, geo.group_size - 1)


def build_program(
    device,
    x,
    out,
    geo,
    *,
    variant,
    arity=(1,),
    block_rows,
    num_rows,
    inv_w,
    eps,
    compute_config,
):
    ft = ttnn.tile_size(ttnn.float32)
    all_cores = geo.core_range_set
    mcast = _mcast(device, geo)
    sem_base = mcast.next_base_sem_id()

    G = geo.group_size
    if variant == V_FLAT:
        arity = (G,)
    else:
        assert legal(G, arity), f"arity {arity} illegal for GROUP_SIZE={G}"
    levels = len(arity)

    # ---- CBs -------------------------------------------------------------
    pages = cb_pages(G, block_rows, arity, variant)
    cbs = [
        _cb(CB_STAT_HANDOFF, ft, pages["stat_handoff"], ttnn.float32, all_cores),
        ttnn.cb_descriptor_from_sharded_tensor(CB_ROW_FINAL, out),
    ]
    if variant == V_FLAT:
        cbs.append(_cb(CB_G0, ft, pages["gather0"], ttnn.float32, all_cores))
    else:
        for l in range(levels):
            cbs.append(_cb(CB_G0 + l, ft, pages[f"gather{l}"], ttnn.float32, all_cores))
        if levels > 1:
            cbs.append(_cb(CB_NODE_OUT, ft, pages["node_out"], ttnn.float32, all_cores))

    # ---- per-core role table --------------------------------------------
    virt = {}

    def v(core):
        key = (core.x, core.y)
        if key not in virt:
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, core.y))
            virt[key] = (c.x, c.y)
        return virt[key]

    x_addr = x.buffer_address()
    wr_args = {}
    cp_args = {}

    for group in geo.groups:
        # The WHOLE group's virtual-coord table, slot-indexed: any core resolves its own
        # parent at any level from it, so there is no per-level host table to drift.
        gtab = []
        for core in group:
            gtab.extend(v(core))
        for slot, core in enumerate(group):
            is_root = 1 if slot == 0 else 0
            wr_args[(core.x, core.y)] = [x_addr, num_rows, is_root, slot] + gtab + list(mcast.runtime_args(core))
            cp_args[(core.x, core.y)] = [num_rows, is_root, slot]

    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core in geo.cores:
        key = (core.x, core.y)
        if key not in wr_args:
            # INACTIVE: in the mcast box, in no group.  num_rows == 0 makes both kernels
            # return before touching anything (the op's contract).
            wr_args[key] = [x_addr, 0, 0, 0] + [0] * (2 * G) + list(mcast.runtime_args(core))
            cp_args[key] = [0, 0, 0]
        writer_rt[core.x][core.y] = wr_args[key]
        compute_rt[core.x][core.y] = cp_args[key]

    fa = list(arity) + [1] * (MAX_LEVELS - levels)
    writer_ct = [variant, G, block_rows, levels] + fa + [sem_base, GATHER_FACES]
    assert len(writer_ct) == 10, "bench_writer.cpp expects McastArgs<10, 4 + 2*GROUP_SIZE>()"
    writer_ct.extend(mcast.compile_time_args())

    compute_ct = [variant, G, block_rows, levels] + fa + [_f32_bits(inv_w), _f32_bits(eps)]

    semaphores = list(mcast.owned_semaphores())
    for l in range(max(levels, 1)):
        semaphores.append(ttnn.SemaphoreDescriptor(id=sem_base + l, core_ranges=all_cores, initial_value=0))

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "bench_writer.cpp"),
                core_ranges=all_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),  # NoC1, like the op's combine
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
