# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ROUND 2 of the ISOLATED bake-off for the rms_norm cross-core width COMBINE topology.

Round 1 is ``perf_experiments/hierarchical_gather/``.  This dir re-measures the same idea
against the POST-PERF-1 root chain (D16 + D17 + D19), because round 1's own note said the
absolute win scales with the per-fold cost and Perf 1 cut that cost.

NOT the op.  Reconstructs the combine (and nothing else) as a standalone
``ttnn.ProgramDescriptor``:

  * every core of a group starts with ``num_rows`` fp32 partial tiles already resident in
    its own L1 (a HEIGHT-sharded fp32 input tensor -- pass A is deliberately not modelled,
    so the measured delta is attributable to the collective alone);
  * every core must end with the group's finalized stat
    ``rsqrt(sum_group(partial) * (1/W) + eps)`` in ``cb_row_final``, which is backed on the
    output shard -- so the result IS the output tensor.

Variants
--------
``flat``            the op's CURRENT approach (honest baseline): one root gathers
                    GROUP_SIZE partials per row, folds them all, finalizes them all, and
                    multicasts back.
``grid_k<K>_m<m>``  ONE unified topology whose corners are the whole policy space:
                    K slot chunks (tree arity) x m row-subset gatherers.
                      (1,1) == flat, through the generic path (the overhead control)
                      (K,1) == round 1's slot ``tree_kK``
                      (1,m) == round 1's ``rowsplit_wm``
                      (K,m) == the combined point round 1 never measured.

Precision contract (FIXED, never a lever): fp32 partials, HiFi2, fp32_dest_acc_en=False,
math_approx_mode=False -- identical for every variant.
"""

from __future__ import annotations

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
FP32_TILE_BYTES = 4096
GATHER_FACES = 2  # the op's D13 compact gather: faces 0 and 2 only
ROW_STAT_DEPTH = 2  # the op's CB_ROW_STAT_DEPTH

V_FLAT, V_GRID = 0, 1


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _cb(index, page_size, num_pages, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


# ---------------------------------------------------------------------------
# geometry (unchanged from round 1 -- the two placements the op actually builds)
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
# the (K, m) policy space
# ---------------------------------------------------------------------------


def legal(group_size, block_rows, k, m):
    """Is (K, m) expressible on this geometry?

    K must divide GROUP_SIZE (contiguous slot chunks), a chunk must contain its m
    gatherers (m <= M = GROUP_SIZE / K, which is also what fixes every gatherer's
    expected arrival count at M - 1 and forbids a self-signal), and there is no point
    splitting more row subsets than there are rows in a block.
    """
    if k < 1 or m < 1:
        return False
    if group_size % k != 0:
        return False
    if m > group_size // k:
        return False
    if m > block_rows:
        return False
    return True


def cb_pages(group_size, block_rows, k, m, variant):
    """Per-core L1 page counts of the combine's OWN CBs (fp32 tiles), excluding
    cb_row_final (which is backed on the output shard in both variants)."""
    if variant == V_FLAT:
        return {
            "partials_gathered": group_size * block_rows,
            "row_stat": ROW_STAT_DEPTH * block_rows,
            "stat_handoff": block_rows,
        }
    mm = group_size // k
    rpw = -(-block_rows // m)
    out = {
        "partials_gathered": mm * rpw,
        "row_stat": ROW_STAT_DEPTH * rpw,
        "stat_handoff": rpw,
    }
    if k > 1:
        out["stage2"] = k * rpw
        out["subroot_out"] = rpw
    return out


def l1_bytes(group_size, block_rows, k, m, variant):
    return sum(cb_pages(group_size, block_rows, k, m, variant).values()) * FP32_TILE_BYTES


def num_semaphores(k, m):
    """Semaphores the combine needs BEYOND the multicast helper's two."""
    n = 1  # sem1: stage-1 gather arrivals (the op already has this one)
    if k > 1:
        n += 1  # sem2: stage-2 forward arrivals
    if m > 1:
        n += 1  # sem3: finished-rows-in-the-mcast-buffer arrivals
    return n


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
    k=1,
    m=1,
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
    sem3 = sem1 + 2

    G = geo.group_size
    if variant == V_FLAT:
        k, m = 1, 1
    else:
        assert legal(G, block_rows, k, m), f"(K={k}, m={m}) illegal for G={G}, BLOCK_ROWS={block_rows}"
    mm = G // k
    gmax = k * m

    # ---- CBs -------------------------------------------------------------
    pages = cb_pages(G, block_rows, k, m, variant)
    cbs = [
        _cb(CB_PARTIALS_GATHERED, ft, pages["partials_gathered"], ttnn.float32, all_cores),
        _cb(CB_ROW_STAT, ft, pages["row_stat"], ttnn.float32, all_cores),
        _cb(CB_STAT_HANDOFF, ft, pages["stat_handoff"], ttnn.float32, all_cores),
        ttnn.cb_descriptor_from_sharded_tensor(CB_ROW_FINAL, out),
    ]
    if variant == V_GRID and k > 1:
        cbs.append(_cb(CB_STAGE2, ft, pages["stage2"], ttnn.float32, all_cores))
        cbs.append(_cb(CB_SUBROOT_OUT, ft, pages["subroot_out"], ttnn.float32, all_cores))

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
        # gatherer coord table, gidx = j * m + w  ->  slot j * mm + w
        gtab = []
        for j in range(k):
            for w in range(m):
                gtab.extend(v(group[j * mm + w]))
        for slot, core in enumerate(group):
            is_root = 1 if slot == 0 else 0
            wr_args[(core.x, core.y)] = (
                [x_addr, num_rows, is_root, slot] + [0] * 8 + gtab + list(mcast.runtime_args(core))
            )
            cp_args[(core.x, core.y)] = [num_rows, is_root, slot]

    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core in geo.cores:
        key = (core.x, core.y)
        if key not in wr_args:
            # INACTIVE: in the mcast box, in no group.  num_rows == 0 makes both kernels
            # return before touching anything (the op's contract).
            wr_args[key] = [x_addr, 0] + [0] * 10 + [0] * (2 * gmax) + list(mcast.runtime_args(core))
            cp_args[key] = [0, 0, 0]
        writer_rt[core.x][core.y] = wr_args[key]
        compute_rt[core.x][core.y] = cp_args[key]

    writer_ct = [variant, G, block_rows, k, m, sem1, sem2, sem3, GATHER_FACES]
    assert len(writer_ct) == 9, "bench_writer.cpp expects McastArgs<9, 12 + 2*K*m>()"
    writer_ct.extend(mcast.compile_time_args())

    compute_ct = [variant, G, block_rows, k, m, _f32_bits(inv_w), _f32_bits(eps)]

    semaphores = list(mcast.owned_semaphores())
    for sid in (sem1, sem2, sem3):
        semaphores.append(ttnn.SemaphoreDescriptor(id=sid, core_ranges=all_cores, initial_value=0))

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
