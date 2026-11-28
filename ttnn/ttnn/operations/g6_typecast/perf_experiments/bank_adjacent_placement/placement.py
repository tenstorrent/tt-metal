# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0
"""bank_adjacent_placement: host-side core placement for the TILE path of g6_typecast.

The op's TILE descriptor is used UNCHANGED (same kernels, CBs, CT args, block plan, tile ranges).
Only WHICH core gets which contiguous tile range changes: the descriptor's calls to
`ttnn.split_work_to_cores` / `ttnn.corerange_to_cores` are answered by a proxy that returns the same
group sizes and per-core counts, but an ordered core list chosen by a placement mode.

Modes (string):
  default            the op's own row-major order from (0,0) (pass-through)
  list:x,y/x,y/...   explicit core per unit, in unit order (unit = one core's contiguous range)
  noc0 | noc1        unit whose first tile t lives in DRAM bank b = t % nb goes to the row the device
                     names optimal for bank b on that NoC (get_optimal_dram_bank_to_logical_worker_assignment);
                     inside the row, nearest free x to the assigned x; full row -> nearest row (|dy|)
  rows:r0,r1,..,r7   same, with an explicit bank -> row table
  spread0 | spread1  bank row on that NoC, but the start x of unit i is i % grid.x (units spread over the
                     columns instead of piling into the assigned column)
  rule               RECOMMENDED: DRAM -> spread0 up to 4 units per bank, noc0 above; L1 -> default
  cm                 control: column-major order from (0,0), no bank awareness
"""
from __future__ import annotations

import importlib

import ttnn

_pd = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast_program_descriptor")
_op = importlib.import_module("ttnn.operations.g6_typecast.g6_typecast")  # the module, not the function

_ORIG_TILE = _pd._create_tile_program_descriptor
MODE = {"value": "default"}


def _bank_rows(device, mode):
    if mode.startswith("rows:"):
        rows = [int(v) for v in mode[5:].split(",")]
        return [(None, r) for r in rows]
    noc = ttnn.NOC.NOC_0 if mode == "noc0" else ttnn.NOC.NOC_1
    return [(c.x, c.y) for c in device.get_optimal_dram_bank_to_logical_worker_assignment(noc)]


def ordered_cores(device, grid, mode, unit_first_tiles, num_banks=8, is_dram=True):
    """-> ordered list of CoreCoord, one per unit."""
    n = len(unit_first_tiles)
    if mode == "rule":  # RECOMMENDED: see placement_rule()
        if not is_dram or n >= grid.x * grid.y:  # L1, or full grid (measured +3..10% on Q2188/Q2219/Q2187/Q2209)
            mode = "default"
        else:
            mode = "spread0" if n <= SPREAD_MAX_UNITS_PER_BANK * num_banks else "noc0"
    if mode == "default":
        return [ttnn.CoreCoord(k % grid.x, k // grid.x) for k in range(n)]
    if mode.startswith("list:"):
        pts = [tuple(int(v) for v in p.split(",")) for p in mode[5:].split("/")]
        assert len(pts) >= n, (mode, n)
        return [ttnn.CoreCoord(x, y) for (x, y) in pts[:n]]
    if mode == "cm":  # control: column-major order, no bank awareness
        return [ttnn.CoreCoord(k // grid.y, k % grid.y) for k in range(n)]
    spread = mode.startswith("spread")
    table = _bank_rows(device, "noc1" if mode == "spread1" else ("noc0" if spread else mode))
    used = set()
    out = []
    for i, t in enumerate(unit_first_tiles):
        x0, y0 = table[t % num_banks]
        if spread:  # bank row, but walk the units across the columns (x = unit index mod grid.x)
            x0 = i % grid.x
        if x0 is None:
            x0 = 0 if (t % num_banks) < num_banks // 2 else grid.x - 1
        placed = False
        for y in sorted(range(grid.y), key=lambda y: (abs(y - y0), y)):
            for x in sorted(range(grid.x), key=lambda x: (abs(x - x0), x)):
                if (x, y) not in used:
                    used.add((x, y))
                    out.append(ttnn.CoreCoord(x, y))
                    placed = True
                    break
            if placed:
                break
        assert placed
    return out


# Units per DRAM bank up to which the units of one bank are spread across the columns (spread0); above it
# they sit side by side from the bank's assigned column (noc0). Measured: 4/bank (32 tiles) spread0 = noc0;
# 6-8/bank (F320 49, Q2203 64 tiles) noc0 is 6-8% faster than spread0; 1-2/bank spread0 is 5-10% faster.
SPREAD_MAX_UNITS_PER_BANK = 4


def placement_rule(device, grid, unit_first_tiles, is_dram):
    """The host rule proposed for the op: ordered core per unit (unit = one core's contiguous tile range,
    unchanged from split_work_to_cores). DRAM: unit whose first tile t sits in bank b = t % nb goes to the
    row the device names NoC0-optimal for bank b (any column; full row -> nearest row). Up to 4 units per
    bank the units walk the columns (x = unit index % grid.x); above, a bank's units sit side by side from
    the bank's assigned column. L1 buffers, and any split that uses the whole grid (tiles >= cores): the
    default row-major order (measured: placement only moves the NoC0 DRAM-read latency, ~75 ns/core)."""
    return ordered_cores(device, grid, "rule", unit_first_tiles, is_dram=is_dram)


class _TtnnProxy:
    """Delegates everything to ttnn except the two work-split calls of the TILE descriptor."""

    def __init__(self, device, mode, is_dram=True):
        self._is_dram = is_dram
        self._device = device
        self._mode = mode
        self._groups = {}

    def __getattr__(self, name):
        return getattr(ttnn, name)

    def split_work_to_cores(self, grid, units, row_wise=True):
        num_cores, all_cores, g1, g2, n1, n2 = ttnn.split_work_to_cores(grid, units, row_wise=row_wise)
        c1, c2 = g1.num_cores(), (g2.num_cores() if n2 else 0)
        firsts = [k * n1 for k in range(c1)] + [c1 * n1 + k * n2 for k in range(c2)]
        cores = ordered_cores(self._device, grid, self._mode, firsts, is_dram=self._is_dram)
        self.placed = list(zip(cores, firsts))

        def crs(cs):
            return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cs]) if cs else ttnn.CoreRangeSet([])

        new_all, new_g1, new_g2 = crs(cores), crs(cores[:c1]), crs(cores[c1:])
        self._groups = {id(new_g1): cores[:c1], id(new_g2): cores[c1:]}
        return num_cores, new_all, new_g1, new_g2, n1, n2

    def corerange_to_cores(self, group, row_wise=True):
        if id(group) in self._groups:
            return self._groups[id(group)]
        return ttnn.corerange_to_cores(group, row_wise=row_wise)


LAST = {}


def _tile_descriptor(input_tensor, output_tensor):
    mode = MODE["value"]
    if mode == "default":
        return _ORIG_TILE(input_tensor, output_tensor)
    is_dram = input_tensor.memory_config().buffer_type == ttnn.BufferType.DRAM
    proxy = _TtnnProxy(input_tensor.device(), mode, is_dram)
    saved = _pd.ttnn
    _pd.ttnn = proxy
    try:
        desc, plan = _ORIG_TILE(input_tensor, output_tensor)
    finally:
        _pd.ttnn = saved
    plan = dict(plan, placement=mode)
    LAST["placed"] = [((c.x, c.y), t) for c, t in proxy.placed]
    return desc, plan


def set_mode(mode):
    MODE["value"] = mode
    _op._PROGRAM_CACHE.clear()


def install():
    _pd._create_tile_program_descriptor = _tile_descriptor
    _op._PROGRAM_CACHE.clear()
