# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ISOLATED bake-off for the rms_norm cross-core width COMBINE topology.

NOT the op.  Reconstructs the combine (and nothing else) as a standalone
``ttnn.ProgramDescriptor``:

  * every core of a group starts with ``num_rows`` fp32 partial tiles already
    resident in its own L1 (a HEIGHT-sharded fp32 input tensor -- pass A is
    deliberately not modelled, so the measured delta is attributable to the
    collective alone);
  * every core must end with the group's finalized stat
    ``rsqrt(sum_group(partial) * (1/W) + eps)`` in ``cb_row_final``, which is
    backed on the output shard -- so the result IS the output tensor.

Variants
--------
``flat``          the op's current approach (honest baseline): every member ships
                 its partial into its own slot of the ROOT's gather ring and
                 remote-incs the root's arrival semaphore; the root folds
                 GROUP_SIZE partials per row, finalizes, multicasts back.
``tree_k<K>``     two-stage hierarchy over K contiguous slot chunks.  Slots are
                 assigned row-major over the group's core rectangle, so a
                 contiguous chunk is a grid-row prefix: K == the group's grid-row
                 count IS ``two_stage_grid_reduce``.
``rowsplit_w<W>`` the same flat fan-in, but the block's ROW axis is split over W
                 workers, each of which folds AND finalizes its own rows and
                 writes them straight into the root's mcast buffer.

Precision contract (FIXED, never a lever): fp32 partials, HiFi2,
fp32_dest_acc_en=False, math_approx_mode=False -- identical for every variant.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import NamedTuple

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_PARTIALS_GATHERED = 11
CB_STAGE2 = 12
CB_SUBROOT_OUT = 13
CB_ROW_STAT = 14
CB_STAT_HANDOFF = 15
CB_ROW_FINAL = 16

TILE = 32
GATHER_FACES = 2  # the op's D13 compact gather: faces 0 and 2 only
RSQRT_COL = 1  # the op's D15 column-scoped rsqrt
ROW_STAT_DEPTH = 2  # the op's CB_ROW_STAT_DEPTH

V_FLAT, V_TREE, V_ROWSPLIT = 0, 1, 2


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _cb(index, page_size, num_pages, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


class Geometry(NamedTuple):
    """One group placement.

    ``groups`` is a list of per-group core lists (logical CoreCoord), slot order.
    ``cores`` is every core of the program in row-major order (the shard order).
    ``inactive`` are cores inside the mcast bounding box that are NOT in any group.
    """

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
    """Two placements, mirroring the two the op actually builds.

    ``num_groups > 1``  -> one group per GRID ROW (group_size <= grid.x), i.e. the
                           BLOCK-sharded focus shape's topology.  Mcast1D PerRow.
    ``num_groups == 1``  -> one group packed row-major into a ``box_w``-wide box.
                           A rectangle when box_w divides group_size, otherwise a
                           genuinely NON-RECTANGULAR group whose trailing in-box
                           cores join INACTIVE -- the op's row-major-packed
                           WIDTH-shard grid.  Mcast2D.
    """
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
# program
# ---------------------------------------------------------------------------


def _mcast(device, geo, *, num_active_override=None):
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
    k=1,
    w_max=0,
    block_rows,
    num_rows,
    inv_w,
    eps,
    compute_config,
):
    ft = ttnn.tile_size(ttnn.float32)
    all_cores = geo.core_range_set
    mcast = _mcast(device, geo)
    sem1 = mcast.next_base_sem_id()
    sem2 = sem1 + 1

    if variant == V_TREE:
        assert geo.group_size % k == 0, "TREE: K must divide GROUP_SIZE"
        m = geo.group_size // k
        fanin = m
    else:
        m = geo.group_size
        fanin = geo.group_size

    # ---- CBs -------------------------------------------------------------
    cbs = [
        _cb(CB_PARTIALS_GATHERED, ft, fanin * block_rows, ttnn.float32, all_cores),
        _cb(CB_ROW_STAT, ft, ROW_STAT_DEPTH * block_rows, ttnn.float32, all_cores),
        _cb(CB_STAT_HANDOFF, ft, block_rows, ttnn.float32, all_cores),
        ttnn.cb_descriptor_from_sharded_tensor(CB_ROW_FINAL, out),
    ]
    if variant == V_TREE:
        cbs.append(_cb(CB_STAGE2, ft, k * block_rows, ttnn.float32, all_cores))
        cbs.append(_cb(CB_SUBROOT_OUT, ft, block_rows, ttnn.float32, all_cores))

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
        root = group[0]
        # ROWSPLIT worker coord table (slots 0..w_max-1), padded to w_max entries.
        worker_coords = []
        for i in range(w_max):
            wc = group[i] if i < len(group) else group[0]
            worker_coords.extend(v(wc))
        for slot, core in enumerate(group):
            if variant == V_TREE:
                j = slot // m
                pos = slot % m
                subroot = group[j * m]
                is_subroot = 1 if pos == 0 else 0
            else:
                j, pos, is_subroot = 0, slot, 0
                subroot = root
            is_root = 1 if slot == 0 else 0
            wr_args[(core.x, core.y)] = (
                [
                    x_addr,
                    num_rows,
                    is_root,
                    slot,
                    pos,
                    j,
                    is_subroot,
                    v(subroot)[0],
                    v(subroot)[1],
                    0,
                    0,
                    0,
                ]
                + worker_coords
                + list(mcast.runtime_args(core))
            )
            cp_args[(core.x, core.y)] = [num_rows, is_root, is_subroot, slot]

    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core in geo.cores:
        key = (core.x, core.y)
        if key not in wr_args:
            # INACTIVE: in the mcast box, in no group.  num_rows == 0 makes both
            # kernels return before touching anything (the op's contract).
            wr_args[key] = [x_addr, 0] + [0] * 10 + [0] * (2 * w_max) + list(mcast.runtime_args(core))
            cp_args[key] = [0, 0, 0, 0]
        writer_rt[core.x][core.y] = wr_args[key]
        compute_rt[core.x][core.y] = cp_args[key]

    writer_ct = [
        variant,
        geo.group_size,
        block_rows,
        max(1, k),
        sem1,
        sem2,
        GATHER_FACES,
        w_max,
    ]
    assert len(writer_ct) == 8, "bench_writer.cpp expects McastArgs<8, 12 + 2*W_MAX>()"
    writer_ct.extend(mcast.compile_time_args())

    compute_ct = [
        variant,
        geo.group_size,
        block_rows,
        max(1, k),
        w_max,
        _f32_bits(inv_w),
        _f32_bits(eps),
        RSQRT_COL,
    ]

    semaphores = list(mcast.owned_semaphores())
    semaphores.append(ttnn.SemaphoreDescriptor(id=sem1, core_ranges=all_cores, initial_value=0))
    semaphores.append(ttnn.SemaphoreDescriptor(id=sem2, core_ranges=all_cores, initial_value=0))

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


# ---------------------------------------------------------------------------
# variants
# ---------------------------------------------------------------------------


def variant_spec(name, group_size, block_rows):
    """(variant, k, w_max) for a variant name, or None when inexpressible here."""
    if name == "flat":
        return (V_FLAT, 1, 0)
    if name == "tree_ksqrt":
        k = _best_divisor_near(group_size, math.sqrt(group_size))
        if k is None:
            return None
        return (V_TREE, k, 0)
    if name == "tree_grid_axis":
        # K == the group's grid-row count is `two_stage_grid_reduce`: stage 1 reduces
        # along grid-x inside each grid row, stage 2 along grid-y.  Needs the
        # geometry, so the caller resolves it via grid_axis_k().
        return None
    if name.startswith("tree_k"):
        k = int(name[len("tree_k") :])
        if k < 2 or k >= group_size or group_size % k != 0:
            return None
        return (V_TREE, k, 0)
    if name.startswith("rowsplit_w"):
        w = int(name[len("rowsplit_w") :])
        w = min(w, block_rows, group_size)
        if w < 2:
            return None
        return (V_ROWSPLIT, 1, w)
    raise ValueError(name)


def _best_divisor_near(n, target):
    divs = [d for d in range(2, n) if n % d == 0]
    if not divs:
        return None
    return min(divs, key=lambda d: (abs(d - target), d))


def grid_axis_k(geo):
    """K for `two_stage_grid_reduce`, or None when the group is 1-D (collapses)."""
    if geo.gy <= 1:
        return None
    if geo.group_size % geo.gy != 0:
        return None
    # slots are row-major over a gx-wide box, so a contiguous chunk of gx slots IS a
    # grid row -> K = gy sub-roots, each gathering gx.
    if geo.group_size // geo.gy != geo.gx:
        return None
    return geo.gy
