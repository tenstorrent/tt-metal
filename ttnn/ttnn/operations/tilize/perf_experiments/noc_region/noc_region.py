"""noc_region: static per-region NoC assignment for the tilize reader / writer (experiment only).

The real program descriptor (tilize_program_descriptor.py) is NOT copied or edited. `install()`
swaps the module's `ttnn` global for a proxy that
  * records every KernelDescriptor / RuntimeArgs the descriptor builds, and
  * at ProgramDescriptor time, splits the reader and writer KernelDescriptors into one
    descriptor per region (same source, same CT args, the region's cores' RT args), each with
    its own static (DM_DEDICATED_NOC) `DataMovementConfigDescriptor(processor, noc)`.
Compute, CBs and semaphores are untouched. Processors never move (reader = NCRISC / RISCV_1,
writer = BRISC / RISCV_0); only the NoC each one issues on changes. The kernels use `noc_index`
everywhere (TensorAccessor::get_noc_addr, loopback get_noc_addr(0), barriers), so the same sources
run on either NoC.

A region rule maps a Tensix core's logical CoreCoord -> "swap" (reader NoC1 / writer NoC0) or
"default" (reader NoC0 / writer NoC1). Both RISC-Vs on one NoC is not expressible in
DM_DEDICATED_NOC (each NoC's command buffers / counters are owned by exactly one RISC-V).

Rules are written in NoC0 physical coordinates of the Tensix core (WH B0 n150 8x8 worker grid:
logical x 0..7 -> NoC0 x {1,2,3,4,6,7,8,9}; logical y 0..7 -> NoC0 y {1,2,3,4,7,8,9,10}; DRAM
endpoints sit in NoC0 columns x = 0 and x = 5). `phys()` resolves the mapping from the device.
"""

from __future__ import annotations

import types
from collections import defaultdict

import ttnn

# Logical -> NoC0 physical Tensix coordinates of this box (WH B0 n150, 8x8 worker grid), read off
# the device profiler's core_x / core_y (which are NoC0 physical). NOTE: the device's
# worker_core_from_logical_core returns *virtual* (translated, 18+) coordinates on WH, which are
# useless for a geometry rule.
_NOC0_X = (1, 2, 3, 4, 6, 7, 8, 9)
_NOC0_Y = (1, 2, 3, 4, 7, 8, 9, 10)


def phys(core):
    return _NOC0_X[core.x], _NOC0_Y[core.y]


def _load_phys(device):
    g = device.compute_with_storage_grid_size()
    assert (g.x, g.y) == (8, 8), "the NoC0 table above is for the 8x8 WH n150 grid"


# ---- region rules (core -> True = swap: reader on NoC1, writer on NoC0) ----
# Geometry of the WH worker grid in NoC0 coordinates: x in 1..9 (5 = DRAM column), y in 1..10
# (5, 6 = DRAM / eth rows between the two Tensix bands). "bottom" = the lower Tensix band
# (NoC0 y >= 7), "right" = east of the DRAM column 5 (NoC0 x >= 6).
def _bottom(c):
    return phys(c)[1] >= 7


def _right(c):
    return phys(c)[0] >= 6


RULES = {
    "default": lambda c: False,
    "swap_all": lambda c: True,
    "swap_bottom": _bottom,
    "swap_top": lambda c: not _bottom(c),
    "swap_right": _right,
    "swap_left": lambda c: not _right(c),
    "swap_br": lambda c: _bottom(c) and _right(c),
    "swap_tl": lambda c: not _bottom(c) and not _right(c),
    "swap_diag": lambda c: _bottom(c) == _right(c),  # bottom-right + top-left quadrants
    "swap_anti": lambda c: _bottom(c) != _right(c),  # bottom-left + top-right quadrants
    "swap_tail9": lambda c: phys(c)[0] in (6, 7, 8) and phys(c)[1] >= 8,  # the measured BRISC tail
    "swap_col6": lambda c: phys(c)[0] == 6,  # the Tensix column just east of DRAM column 5
    "swap_row10": lambda c: phys(c)[1] == 10,  # bottom Tensix row
    # small swapped sets in the upper-left band (swap_tl made the lower-left NoC1 writers fast)
    "swap_tl4": lambda c: phys(c)[0] <= 2 and phys(c)[1] <= 2,
    "swap_row1_left": lambda c: phys(c)[0] <= 4 and phys(c)[1] == 1,
    "swap_col1": lambda c: phys(c)[0] == 1,
    "swap_col1_top": lambda c: phys(c)[0] == 1 and phys(c)[1] <= 4,
    "swap_one": lambda c: phys(c) == (1, 1),
    "swap_one_2_1": lambda c: phys(c) == (2, 1),
    "swap_one_9_1": lambda c: phys(c) == (9, 1),
    "swap_one_4_4": lambda c: phys(c) == (4, 4),
    "swap_one_1_10": lambda c: phys(c) == (1, 10),
    "swap_one_7_9": lambda c: phys(c) == (7, 9),
    "swap_one_9_10": lambda c: phys(c) == (9, 10),
    "swap_two_1_1_2_1": lambda c: phys(c) in ((1, 1), (2, 1)),
    "swap_tr": lambda c: not _bottom(c) and _right(c),
    "swap_bl": lambda c: _bottom(c) and not _right(c),
    # controls: two kernel groups, nothing moved
    "split_one": lambda c: 2 if phys(c) == (1, 1) else 0,
    "split_half": lambda c: 2 if _bottom(c) else 0,
    "checker": lambda c: (c.x + c.y) % 2 == 1,
    "checker_row": lambda c: c.y % 2 == 1,  # every other Tensix row
    "checker_col": lambda c: c.x % 2 == 1,  # every other Tensix column
}


class _RecRT:
    """Stand-in for ttnn.RuntimeArgs: rt[x][y] = list (and .extend on it)."""

    def __init__(self):
        self.d = defaultdict(dict)

    def __getitem__(self, x):
        return self.d[x]

    def real(self, keep=lambda x, y: True):
        rt = ttnn.RuntimeArgs()
        for x, col in self.d.items():
            for y, args in col.items():
                if keep(x, y):
                    rt[x][y] = list(args)
        return rt


class _RecKD:
    def __init__(self, **kw):
        self.kw = kw


def _cores_of(crs):
    return ttnn.corerange_to_cores(crs, None, True)


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _build(rec, rule):
    kw = dict(rec.kw)
    rt = kw.pop("runtime_args")
    rt = rt if isinstance(rt, _RecRT) else None
    src = kw["kernel_source"]
    role = "reader" if src.endswith("tilize_reader.cpp") else "writer" if src.endswith("tilize_writer.cpp") else None
    if role is None or rule is None:
        return [ttnn.KernelDescriptor(runtime_args=rt.real() if rt else ttnn.RuntimeArgs(), **kw)]
    cfg = kw.pop("config")
    cores = _cores_of(kw.pop("core_ranges"))
    out = []
    # rule value: False/0 = default NoCs, True/1 = swapped NoCs, 2 = default NoCs in a SEPARATE
    # kernel group (control: splits the descriptors without moving any traffic)
    for group in (0, 1, 2):
        sel = [c for c in cores if int(rule(c)) == group]
        swapped = group == 1
        if not sel:
            continue
        keys = {(c.x, c.y) for c in sel}
        if role == "reader":
            noc = ttnn.NOC.NOC_1 if swapped else ttnn.NOC.NOC_0
            proc = ttnn.DataMovementProcessor.RISCV_1
        else:
            noc = ttnn.NOC.NOC_0 if swapped else ttnn.NOC.NOC_1
            proc = ttnn.DataMovementProcessor.RISCV_0
        out.append(
            ttnn.KernelDescriptor(
                core_ranges=_crs(sel),
                runtime_args=rt.real(lambda x, y: (x, y) in keys),
                config=ttnn.DataMovementConfigDescriptor(proc, noc, ttnn.NOC_MODE.DM_DEDICATED_NOC),
                **kw,
            )
        )
    return out


def _rebalance_rows(kernels, rows_fn):
    """Secondary lever: rewrite each core's (row_start, core_row_tiles) RT args (reader [1],[2],
    writer [1],[2], compute [0]) so core c owns rows_fn(cores, R)[c] tile-rows, in the original
    row order. Only for the row split (every core owns all C tile-columns); CT args untouched."""
    by_role = {}
    for k in kernels:
        src = k.kw["kernel_source"]
        role = next((r for r in ("reader", "writer", "compute") if src.endswith(f"tilize_{r}.cpp")), None)
        by_role[role] = k.kw["runtime_args"].d
    rd, wr, cp = by_role["reader"], by_role["writer"], by_role["compute"]
    cores = sorted(((x, y) for x in rd for y in rd[x]), key=lambda c: rd[c[0]][c[1]][1])
    R = sum(rd[x][y][2] for x, y in cores)
    if not all(rd[x][y][3] == 0 for x, y in cores):
        return  # grid_2d_split (column groups): not a row split, left as the op built it
    new = rows_fn([ttnn.CoreCoord(x, y) for x, y in cores], R)
    assert sum(new.values()) == R and min(new.values()) >= 1, (sum(new.values()), R)
    start = 0
    for x, y in cores:
        n = new[(x, y)]
        rd[x][y][1], rd[x][y][2] = start, n
        wr[x][y][1], wr[x][y][2] = start, n
        cp[x][y][0] = n
        start += n


def make_proxy(rule, rows_fn=None):
    proxy = types.ModuleType("ttnn_noc_region_proxy")
    proxy.__dict__.update({k: getattr(ttnn, k) for k in dir(ttnn) if not k.startswith("__")})
    proxy.RuntimeArgs = _RecRT
    proxy.KernelDescriptor = _RecKD

    def program_descriptor(kernels, **kw):
        if rows_fn is not None:
            _rebalance_rows(kernels, rows_fn)
        real = []
        for k in kernels:
            real.extend(_build(k, rule))
        return ttnn.ProgramDescriptor(kernels=real, **kw)

    proxy.ProgramDescriptor = program_descriptor
    return proxy


def install(monkeypatch, device, rule_name, rows_fn=None):
    """Route tilize's program descriptor through the region splitter for `rule_name`
    (None = keep the op's own single reader / writer descriptors) and the optional row rebalance."""
    import ttnn.operations.tilize.tilize_program_descriptor as pd

    _load_phys(device)
    rule = None if rule_name in (None, "unpatched") else RULES[rule_name]
    monkeypatch.setattr(pd, "ttnn", make_proxy(rule, rows_fn))
