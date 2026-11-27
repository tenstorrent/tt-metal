# SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""Bank-local weights for ttnn.bringup.grouped_matmul (option ``w_layout``; deepseek_v41_flash_d_p WA.2).

Today's weight is DRAM interleaved: tile (g, k, n) of w [1, G, K, N] sits in bank (page % banks), so every core reads
its columns one 1088 B tile at a time from banks across the chip. Here the weight is one WIDTH_SHARDED DRAM tensor, one
32-wide column of a [rows, banks * 32] host matrix a bank (the routed experts' layout: flat_routed_expert_weights
``bank_host_tensor`` / ``bank_sharded``, reused as is), and every core reads only its own bank:

    units     the G x NT (group, column tile) units in order, an equal run a bank (G == banks: bank b = group b); a
              bank's units of one group are split over a rectangle of its cores (column_split)
    region    a core's PN columns K-major in its bank: tile (k, j) at tile0 + k * PN + j, the cores' regions back to
              back, so a K block of any BW is ONE contiguous read of BW x PN tiles at tile0 + kb * BW * PN (in the CB
              slot order [k][j] of today's path)
    cores     each bank's band: a W x 2 rectangle (W = cores_per_bank / 2) on its anchor's side of the grid (the bank's
              optimal NoC-0 worker), in the 2-row band nearest the anchor; its x sender is the rectangle's first core
    same      tiles, K order, fidelity, fp32 partials and output writes as today's path: bit-identical output

No padding anywhere (OV-3): a shape whose units do not split evenly over the banks, or whose bands do not fit the
grid, has no layout (``plan`` returns None) and the caller keeps today's path."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from functools import cached_property

import torch

import ttnn
from ttnn.bringup.moe_block_ttnn.flat_routed_expert.flat_routed_expert_weights import (
    _bank_memory_config,
    bank_host_tensor,
)

from .grouped_matmul import column_split


SHAPES = ("band", "row", "rect")


@dataclass(frozen=True)
class Slot:
    """One core of the bank path: its bank, its logical core, its group and columns, its region's first tile, its
    cluster (the rectangle its x block is multicast to) and whether it is that cluster's x sender."""

    bank: int
    core: tuple  # (x, y) logical
    g: int
    n0: int
    pn: int
    tile0: int
    cluster: int
    sender: bool


@dataclass(frozen=True)
class BankLayout:
    """A weight's bank layout for one device grid (``plan``). ``rect[c]`` = the logical (x0, y0, x1, y1) of cluster c,
    ``sender[c]`` its sender's logical core."""

    G: int
    KT: int
    NT: int
    banks: int
    cores_per_bank: int
    slots: tuple
    rect: tuple
    sender: tuple

    @property
    def tiles_per_bank(self) -> int:
        return self.G * self.NT * self.KT // self.banks

    @property
    def pn_max(self) -> int:
        return max(s.pn for s in self.slots)

    @cached_property
    def _key(self) -> tuple:
        return (self.G, self.KT, self.NT, self.banks, self.cores_per_bank, hash((self.slots, self.rect, self.sender)))

    def key(self) -> tuple:
        """The program-cache key of the layout: computed once (hashing the slot table on every call cost host time)."""
        return self._key

    def memory_config(self):
        return _bank_memory_config(self.tiles_per_bank * 32, self.banks)


def _bands(anchor_ys: list, n_bands: int, height: int, contiguous: bool = False) -> tuple | None:
    """Band indices (rows [i * height, (i + 1) * height)) for anchors sorted by y, increasing, distinct, nearest in
    sum of |band centre - anchor y|; ``contiguous``: consecutive bands only (no gap between a side's bands)."""
    if len(anchor_ys) > n_bands:
        return None
    best = None
    n = len(anchor_ys)
    picks = (
        (tuple(range(i, i + n)) for i in range(n_bands - n + 1))
        if contiguous
        else itertools.combinations(range(n_bands), n)
    )
    for pick in picks:
        cost = sum(abs(i * height + (height - 1) / 2 - y) for i, y in zip(pick, anchor_ys))
        if best is None or cost < best[0]:
            best = (cost, pick)
    return best[1]


def plan(
    device,
    G: int,
    K: int,
    N: int,
    max_cores: int,
    cores_per_bank: int = 12,
    shape: str = "band",
    sender: str = "start",
    order: str = "row",
    contiguous: bool = False,
    grid=None,
) -> BankLayout | None:
    """The bank layout of w [1, G, K, N] on ``device``'s grid, or None (no even split over the banks, odd
    cores_per_bank, a band that does not fit, a bank's group share not a whole number of rectangle columns, more cores
    than ``max_cores``: the matmul core cap, agent rule 18). ``shape``: "band" (a cores_per_bank / 2 x 2 rectangle a
    bank on its anchor's side), "row" (one grid row a bank, cores_per_bank wide, on its anchor's side, the rows
    nearest the anchors) or "rect" (today's interleaved path's core set: grid row b = bank b, columns 0 ..
    cores_per_bank - 1, one rectangle from (0, 0); WA.3, OV-79). ``sender``: the cluster's x sender, "start" (default: the rectangle's first core, (x0, y0):
    every multicast leaves it eastward / southward on NoC 0, as today's rows) or "anchor" (the core nearest the bank's
    anchor: a NoC-0 multicast from inside the rectangle wraps around the chip, 2x slower x stream on the 8x4, WA.2).
    ``order``: the order a rectangle's cores take its columns (the wider cores first), "row" (default: row by row, the
    order before WA.3) or "col" (column by column: the wider cores of every band then share the rectangles' first
    grid column, so a kernel group's cores merge into a few core ranges: fewer dispatch writes a launch, WA.3 / OV-78).
    ``contiguous`` (band / row): a side's bands are consecutive rows (the window of bands nearest its anchors) instead
    of each band nearest its own anchor, so a side's cores form one rectangle (with order "col": one core range a side
    and kernel group). ``grid`` (CoreCoord or (x, y), default the device's compute grid): the cores the layout may use,
    from (0, 0) (a caller whose ops run inside a sub-device, e.g. a model's matmul-capped 11 x 10 compute sub-device
    next to a gather column, passes its grid: WA.3, DS41-382)."""
    if K % 32 or N % 32 or cores_per_bank < 2 or shape not in SHAPES or sender not in ("anchor", "start"):
        return None
    if order not in ("row", "col"):
        return None
    if shape == "band" and cores_per_bank % 2:
        return None
    KT, NT = K // 32, N // 32
    banks = device.dram_grid_size().x
    units = G * NT
    if units % banks or banks * cores_per_bank > max_cores:
        return None
    per_bank = units // banks
    full = device.compute_with_storage_grid_size()
    gx, gy = full.x, full.y
    if grid is not None:
        mx, my = (grid.x, grid.y) if hasattr(grid, "x") else tuple(grid)
        gx, gy = min(gx, int(mx)), min(gy, int(my))
    h, w = (2, cores_per_bank // 2) if shape == "band" else (1, cores_per_bank)
    if (2 * w if shape == "band" else w) > gx or (shape == "rect" and banks > gy):
        return None
    anchors = [(c.x, c.y) for c in device.get_optimal_dram_bank_to_logical_worker_assignment(ttnn.NOC.NOC_0)]
    if len(anchors) != banks:
        return None
    sides = {0: [], 1: []}  # 0 west (columns 0 .. w-1), 1 east (gx-w .. gx-1)
    for b, (ax, _) in enumerate(anchors):
        sides[0 if ax < gx / 2 else 1].append(b)
    band_of = {b: b for b in range(banks)} if shape == "rect" else {}  # (rect: grid row b = bank b, from column 0)
    # (band: each side its own bands; row: one grid row a bank, the rows shared by both sides)
    for bs in () if shape == "rect" else sides.values() if shape == "band" else [sides[0] + sides[1]]:
        bs.sort(key=lambda b: anchors[b][1])
        pick = _bands([anchors[b][1] for b in bs], gy // h, h, contiguous)
        if pick is None:
            return None
        band_of.update(zip(bs, pick))
    slots, rect, senders = [], [], []
    for b in range(banks):
        x0 = 0 if shape == "rect" or b in sides[0] else gx - w
        y0 = band_of[b] * h
        ax, ay = anchors[b]
        by_g: dict = {}
        for u in range(b * per_bank, (b + 1) * per_bank):
            by_g.setdefault(u // NT, []).append(u % NT)
        t0, cx = 0, x0
        for g in sorted(by_g):
            ns = by_g[g]
            wg = w * len(ns) // per_bank
            if wg * per_bank != w * len(ns) or wg == 0:
                return None
            if order == "col":
                cores = [(x, y) for x in range(cx, cx + wg) for y in range(y0, y0 + h)]
            else:
                cores = [(x, y) for y in range(y0, y0 + h) for x in range(cx, cx + wg)]
            if sender == "anchor":
                snd = min(cores, key=lambda c: (abs(c[0] - ax) + abs(c[1] - ay), c))
            else:
                snd = cores[0]
            ci = len(rect)
            rect.append((cx, y0, cx + wg - 1, y0 + h - 1))
            senders.append(snd)
            n0 = ns[0]
            for core, pn in zip(cores, column_split(len(ns), len(cores))):
                if pn == 0:
                    continue
                slots.append(Slot(b, core, g, n0, pn, t0, ci, core == snd))
                n0, t0 = n0 + pn, t0 + KT * pn
            cx += wg
    if not all(any(s.sender and s.cluster == c for s in slots) for c in range(len(rect))):
        return None  # (a sender core with no columns: never on the 8x4's shapes)
    return BankLayout(G, KT, NT, banks, cores_per_bank, tuple(slots), tuple(rect), tuple(senders))


def regions(w: torch.Tensor, layout: BankLayout) -> list:
    """w [1, G, K, N] (x @ w orientation) -> one [tiles, 32, 32] region a bank: whole tiles in their own orientation
    (a bfp8 tile's bytes, its shared exponents included, are the same wherever it lies)."""
    G, KT, NT = layout.G, layout.KT, layout.NT
    t = w.reshape(G, KT, 32, NT, 32).permute(0, 1, 3, 2, 4)  # [G, KT, NT, 32, 32]
    per = [[] for _ in range(layout.banks)]
    for s in sorted(layout.slots, key=lambda s: (s.bank, s.tile0)):
        per[s.bank].append(t[s.g, :, s.n0 : s.n0 + s.pn].reshape(-1, 32, 32))
    return [torch.cat(p) for p in per]


def upload(w: torch.Tensor, layout: BankLayout, mesh_device, dtype=ttnn.bfloat8_b) -> ttnn.Tensor:
    """w [1, G, K, N] torch -> the replicated WIDTH_SHARDED DRAM weight of ``layout`` (bank_host_tensor's matrix).

    Converted to the block format as a plain TILE host tensor and only then placed with the sharded memory config:
    ``from_torch(..., memory_config=<WIDTH_SHARDED>)`` (flat_routed_expert_weights.bank_sharded's route) quantizes
    bfloat8_b differently (max |d| 0.0625 on randn tiles, WA.2), so its tiles would not be today's bytes."""
    assert tuple(w.shape) == (1, layout.G, layout.KT * 32, layout.NT * 32), (tuple(w.shape), layout.G)
    host = bank_host_tensor(regions(w, layout), layout.banks)
    h = ttnn.from_torch(host.contiguous(), dtype=dtype, layout=ttnn.TILE_LAYOUT)
    return h.to(mesh_device, layout.memory_config())
