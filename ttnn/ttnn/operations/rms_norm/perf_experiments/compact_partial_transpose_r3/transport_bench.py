# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""BENCH B (Perf 3): rms_norm's WHOLE cross-core combine -- FLAT vs COMPACT partials.

Bench A (combine_bench.py) isolates the root's COMPUTE.  This bench exists because the Perf-3
cumulative peel says the combine's TRANSPORT + SYNC residue (16097 ns, 46.7% of the wall) is BIGGER
than the root chain's payload (11297 ns, 32.8%), and the compact layout collapses both:

  gather  a member ships BLOCK_ROWS x GATHER_FACES NoC writes per round (16 writes / 16 kB at the
          focus geometry) to carry 1 kB of information; COMPACT ships ONE whole tile (1 write,
          4 kB).
  ring    the root's landing ring goes GATHER_SLOTS * BLOCK_ROWS pages -> GATHER_SLOTS pages.
  boot    `writer_gather_zero` exists only because GATHER_FACES < 4 leaves landing bytes undefined.
          A compact page is written WHOLE by exactly one member, so the stage disappears (bar the
          odd-GROUP_SIZE pad, now ONE page).
  mcast   the root broadcasts ONE tile instead of BLOCK_ROWS.
  fold    ONE DEST window per round instead of BLOCK_ROWS.
  and it ADDS, in parallel on every core, one pack (sender) and one un-pack (receiver).

Every core starts with its `num_rows` fp32 partials already resident in its own L1 shard and must
end with the group's finalized stat rsqrt(sum_group(partial)/W + eps) in column 0 of each of its
`num_rows` output tiles -- and cb_row_final is backed on the OUTPUT shard, so the result IS the
output tensor.  Pass A, pass B and the write-back are NOT modelled, so the program's whole device
duration IS the combine's EXPOSED cost.  That is deliberately pessimistic for both variants
relative to the op, which hides part of it behind D25's pipeline; the ratio is what transfers.

Precision contract (FIXED, never a lever, identical in both variants): fp32 partial/stat CBs,
fp32_dest_acc_en = False, math_fidelity = HiFi2, math_approx_mode = False.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import NamedTuple

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

CB_X = 0
CB_BANK = 1
CB_SUM_HANDOFF = 2
CB_PARTIALS_GATHERED = 3
CB_STAT_HANDOFF = 4
CB_MCAST_IN = 5
CB_ROW_FINAL = 6

TILE = 32
FP32_TILE_BYTES = 4096
FLAT_GATHER_FACES = 2  # the op's D13 compact gather: faces 0 and 2 only

V_FLAT, V_COMPACT = 0, 1
FIN_SKIP, FIN_C, FIN_RC = 0, 1, 2


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def gather_slots(group_size):
    return group_size + group_size % 2


def fin_for(variant, block_rows):
    """The NARROWEST finalize scope that is CORRECT for the variant's stat-tile layout.

    FLAT's stat is a column vector, so D17's shipped even-parity <2,4> VectorMode::C covers it.
    A COMPACT tile spreads BLOCK_ROWS stats across columns 0..BLOCK_ROWS-1, so the scope must
    widen -- <1,8> C reaches columns 0..15 (faces 0/2), <1,8> RC reaches all 32.  Measured, not
    argued: D17's scope on a compact tile at BLOCK_ROWS = 2 gives pcc 0.99730 / rel-RMS 1036.
    """
    if variant == V_FLAT or block_rows == 1:
        return FIN_SKIP
    return FIN_C if block_rows <= 16 else FIN_RC


def _cb(index, page_size, num_pages, data_format, core_ranges):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


# ---------------------------------------------------------------------------
# geometry -- the two placements the op actually builds (same as the op's combine)
# ---------------------------------------------------------------------------


class Geometry(NamedTuple):
    name: str
    group_size: int
    num_groups: int
    core_range_set: object
    cores: tuple
    groups: tuple
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
            core_range_set=crs,
            cores=tuple(cores),
            groups=groups,
            per_row=True,
        )
    box_w = box_w or min(grid.x, group_size)
    rows = (group_size + box_w - 1) // box_w
    assert rows <= grid.y
    crs = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(box_w - 1, rows - 1))])
    cores = list(ttnn.corerange_to_cores(crs, None, True))
    group = tuple(cores[:group_size])
    return Geometry(
        name=f"g{group_size}_box{box_w}",
        group_size=group_size,
        num_groups=1,
        core_range_set=crs,
        cores=tuple(cores),
        groups=(group,),
        per_row=False,
    )


# ---------------------------------------------------------------------------
# L1 accounting -- the combine's OWN CBs, excluding the shard-backed x / out / bank
# ---------------------------------------------------------------------------


def cb_pages(group_size, block_rows, variant):
    slots = gather_slots(group_size)
    if variant == V_FLAT:
        return {"partials_gathered": slots * block_rows, "stat_handoff": block_rows}
    return {"partials_gathered": slots, "stat_handoff": 2, "sum_handoff": 2, "mcast_in": 2}


def l1_bytes(group_size, block_rows, variant):
    """Combine CBs + the one-hot bank (COMPACT only).  The bank is fp32 here; bench A measured a
    bf16 bank as perf-flat, which halves it, so this is the pessimistic figure."""
    n = sum(cb_pages(group_size, block_rows, variant).values())
    if variant == V_COMPACT:
        n += block_rows  # the one-hot bank
    return n * FP32_TILE_BYTES


def gather_transfers(group_size, block_rows, variant):
    """(NoC writes per member per round, bytes per member per round) for the gather."""
    if variant == V_FLAT:
        n = block_rows * (2 if FLAT_GATHER_FACES == 2 else 1)
        b = block_rows * (FLAT_GATHER_FACES * FP32_TILE_BYTES // 4)
        return n, b
    return 1, FP32_TILE_BYTES


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
    bank,
    out,
    geo,
    *,
    variant,
    block_rows,
    num_rows,
    inv_w,
    eps,
    fin=None,
    dest_batch=4,
    compute_config,
):
    ft = ttnn.tile_size(ttnn.float32)
    all_cores = geo.core_range_set
    mcast = _mcast(device, geo)
    sem1 = mcast.next_base_sem_id()
    G = geo.group_size
    if fin is None:
        fin = fin_for(variant, block_rows)

    pages = cb_pages(G, block_rows, variant)
    cbs = [
        ttnn.cb_descriptor_from_sharded_tensor(CB_X, x),
        ttnn.cb_descriptor_from_sharded_tensor(CB_BANK, bank),
        ttnn.cb_descriptor_from_sharded_tensor(CB_ROW_FINAL, out),
        _cb(CB_PARTIALS_GATHERED, ft, pages["partials_gathered"], ttnn.float32, all_cores),
        _cb(CB_STAT_HANDOFF, ft, pages["stat_handoff"], ttnn.float32, all_cores),
        _cb(CB_SUM_HANDOFF, ft, pages.get("sum_handoff", 1), ttnn.float32, all_cores),
        _cb(CB_MCAST_IN, ft, pages.get("mcast_in", 1), ttnn.float32, all_cores),
    ]

    virt = {}

    def v(core):
        key = (core.x, core.y)
        if key not in virt:
            c = device.worker_core_from_logical_core(ttnn.CoreCoord(core.x, core.y))
            virt[key] = (c.x, c.y)
        return virt[key]

    x_addr = x.buffer_address()
    wr_args, cp_args = {}, {}
    for group in geo.groups:
        for slot, core in enumerate(group):
            is_root = 1 if slot == 0 else 0
            wr_args[(core.x, core.y)] = [x_addr, num_rows, is_root, slot] + [0] * 4 + list(mcast.runtime_args(core))
            cp_args[(core.x, core.y)] = [num_rows, is_root]

    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    for core in geo.cores:
        key = (core.x, core.y)
        if key not in wr_args:
            # INACTIVE: in the mcast box, in no group.  num_rows == 0 makes both kernels return
            # before touching anything (the op's contract).
            wr_args[key] = [x_addr, 0] + [0] * 6 + list(mcast.runtime_args(core))
            cp_args[key] = [0, 0]
        writer_rt[core.x][core.y] = wr_args[key]
        compute_rt[core.x][core.y] = cp_args[key]

    writer_ct = [variant, G, block_rows, sem1, FLAT_GATHER_FACES]
    assert len(writer_ct) == 5, "transport_writer.cpp expects McastArgs<5, 8>()"
    writer_ct.extend(mcast.compile_time_args())
    compute_ct = [variant, G, block_rows, _f32_bits(inv_w), _f32_bits(eps), fin, dest_batch]

    semaphores = list(mcast.owned_semaphores())
    semaphores.append(ttnn.SemaphoreDescriptor(id=sem1, core_ranges=all_cores, initial_value=0))

    return ttnn.ProgramDescriptor(
        kernels=[
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "transport_writer.cpp"),
                core_ranges=all_cores,
                compile_time_args=writer_ct,
                runtime_args=writer_rt,
                config=ttnn.WriterConfigDescriptor(),  # NoC1, like the op's combine
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "transport_compute.cpp"),
                core_ranges=all_cores,
                compile_time_args=compute_ct,
                runtime_args=compute_rt,
                config=compute_config,
            ),
        ],
        semaphores=semaphores,
        cbs=cbs,
    )
