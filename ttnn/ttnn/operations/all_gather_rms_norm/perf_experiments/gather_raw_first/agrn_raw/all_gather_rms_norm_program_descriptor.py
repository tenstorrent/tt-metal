# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm — the MeshProgramDescriptor (one generic_op per call).

Scheme (op_design.md): stats-first, gather-normalized (SF-GN).
  * Port cores: 2L per chip, one per (direction, link), next to the link's Ethernet core. The link-0 ports are the
    stats HUBS (they hold the stats table and exchange it over the fabric, one fused increment per slot).
  * Workers (grid - ports): row-groups x k W members (select_split). Each reduces its block to per-row stats records,
    publishes them into both hubs' tables, waits for the stats round, combines the P = G*k partitions of its rows,
    normalizes its block into cb_out_block (resident) and writes it to its own output pages.
  * The ports pull the own normalized slice from the workers' L1 and send it into the peers' FINAL output pages
    (relay along the ring / line with arrival counters, the far slot of an even ring split between both directions).
  * W_local % 32 != 0 (R_U): the slices go to a slot-contiguous staging scratch instead, then every worker assembles
    its output tiles by element-shifted copies.

Every block knob is a single host value here (block_row_tiles = core_row_tiles, core_w_tiles, k, batch_chunks,
inc_every, scatter runs, balance threshold); CB page counts and loop trips derive from them.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
import os
from pathlib import Path

import ttnn


class _LazyFag:
    """fabric_all_gather's host planning helpers (groups, schedules, port placement), imported on first use: the
    example module touches ttnn.Topology at import time, which is not bound while ttnn.operations imports."""

    def __getattr__(self, name):
        from ttnn.operations.examples.fabric_all_gather import program_descriptor_with_inline_kernels as mod

        return getattr(mod, name)


_fag = _LazyFag()

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- CB indices (kernels/agrn_common.hpp) ----
CB_X = 0
CB_GAMMA = 1
CB_BETA = 2
CB_SCALER = 3
CB_MASK = 4
CB_SUMSQ = 5
CB_LN_M = 6
CB_LN_SHIFT = 7
CB_LN_SQ = 8
CB_PARTIAL = 9
CB_RECORD_STAGING = 10
CB_GATHERED = 12
CB_STAT_CONSTS = 13
CB_INV = 14
CB_MEAN_COL = 15
CB_MEAN_B = 16
CB_RSTD_B = 17
CB_LN_TMP = 18
CB_OUT_BLOCK = 19
CB_ASSEMBLE = 20
CB_PORT_CHUNKS = 21
CB_X32 = 22
CB_GAMMA_ROWS = 23
CB_GATHER_ROWS = 24
CB_STATS_TABLE = 25  # hubs: per-call stats table (program-scoped, see agrn_common.hpp)
# WIDTH_SHARDED row-owner pre-combine (pre_combine, Refinement 1b)
CB_X_STATS = 26  # second zero-copy view of the input shard: the local-stats pass pops it per row chunk
CB_PRE_TABLE = 27  # row owner: the k members' records of its rows
CB_FINAL_TABLE = 28  # every member: the final per-row stats records (inv | mean, rstd) of all R rows
CB_FINAL_INV = 29  # reader -> compute: RMS inv / LN rstd (col 0) of one row chunk
CB_FINAL_MEAN = 30  # reader -> compute: LN mean (col 0) of one row chunk
CB_STAT_CONSTS_X = 31  # LN: n_q of the cross-chip partitions (one per chip)
CB_PORT_DESC = 32  # port reader -> sender: one packet descriptor (runs) per port-CB chunk slot
CB_XG = 33  # RMS xg_pre: x * gamma of the block (T_acc), computed in the stats shadow
CB_STAGE = 34  # own-entry staging: run_pages-page chunks of the ports' own walks, on the active workers
CB_LANDED = 35  # gather-raw-first: per (slot, stream) 16 B landing flags
CB_XR = 36  # gather-raw-first: raw tiles of one remote slot (reader -> compute)
CB_OUT_R = 37  # gather-raw-first: normalized tiles of one remote slot (compute -> writer)
LANDED_FLAG_BYTES = 16
PORT_DESC_BYTES = 64  # agrn_port_common.hpp DESC_BYTES

# ---- program-local semaphores (declared before the fabric connections take theirs) ----
SEM_RECORDS_LOCAL = 0
SEM_STATS_READY = 1
SEM_OWN_READY = 2
SEM_LANDED = 3
SEM_STATS_READY_BWD = 4
SEM_STATS_DONE = 5
SEM_PRE = 6  # row owner: one inc per member (its records of my rows landed in my pre table)
SEM_FINAL = 7  # member: one inc per row owner (its final records landed in my final table)

RECORD_BYTES = 128
TILE = 32

# ---- knobs (single source; op_design.md Blocking Model) ----
P_CAP = 128  # stats partitions per row (G * k), <= 4 gather tiles
COMB_TILE_COST_FUSED = 1  # select_split cost model: one gather-tile reduce per row-tile
STATS_TABLE_CAP = 256 * 1024  # bytes of the stats table on each hub
ROW_CHUNK_L1_BUDGET = 320 * 1024  # pre_combine: bytes of the row-chunked per-row CBs (sets row_chunk)


def _knob(name, default):
    """A block knob: its single host value, overridable by env AGRN_<NAME> for perf sweeps."""
    return int(os.environ.get(f"AGRN_{name}", default))


BATCH_CHUNKS = 4  # port CB batch (TRANSPORT; catalog fabric_all_gather: 4 per batch best or within 5-16%)
INC_EVERY = 8  # arrival increment every N packets (+ each entry's last)
BALANCE_MIN_TILES = 64  # balance an even ring's far slot from this many tiles per slot (catalog)
SCATTER_RUNS = 4  # destination runs per bulk packet (fabric scatter write, <= NOC_SCATTER_WRITE_MAX_CHUNKS)
READ_DEPTH = 4  # port reader: packets with reads in flight (one NoC transaction id each), <= BATCH_CHUNKS (the
#                sender pops whole groups: deeper would hold the CB space its pop waits for)
SPLIT_RUNS = 0  # bulk packets: 1 = split a run that does not fit the open packet (fewer, fuller packets with more
#                scatter chunks: measured +1.2 us on the focus case), 0 = a run that does not fit closes the packet
OWN_SRC_DST = 1  # port reader own entry: 1 = read back the own final pages (one read per bank run, measured faster),
#                 0 = pull page by page from the workers' L1 blocks

# OVERLAP (Refinement 3): own-ready granularity. The row-groups are cut into OWN_BANDS bands (consecutive groups); each
# worker announces its own block on its band's counter and the port's own entry waits only for the bands a packet's
# rows fall in, so the first packets go out before the slowest worker finished. 1 = one counter for all workers.
OWN_BANDS = 1
# port reader: push each packet as soon as its own reads landed (non-blocking check after every issue), not only once
# READ_DEPTH packets are in flight — the first packet of an entry no longer waits for 3 more to be issued
EAGER_PUSH = 1
# RMS with gamma: compute x * gamma into a T_acc block during the stats round, so only x*g * inv (one FPU multiply per
# tile instead of two) follows the stats; on while the block fits XG_L1_BUDGET bytes. Measured (focus): normalize
# 1.03 -> 0.65 us in the zone profile, the harness call time unchanged (29.21 vs 29.18 us) — parked at 0 (live knob)
XG_PRE = 0
XG_L1_BUDGET = 128 * 1024
OWN_BANDS_MAX = 16  # persistent band counters (global semaphores) per port core
# Own-entry staging (Perf 1, OVERLAP): each port's own-entry walk is cut into chunks of run_pages pages, chunk
# g = l * nch + c lives in cb_stage of active worker g * n_active // (L * nch) (round-robin, slot g // n_active, once
# chunks outnumber workers). Every worker writes each normalized page to its chunk home, then increments one readiness
# counter on every port; the port reads each own packet with <= 2 L1 reads instead of DRAM bank runs after the own
# write. cb_stage is capped by STAGE_L1_BUDGET per core: past it, a lane's walk tail goes the DRAM read-back way.
STAGE_L1_BUDGET = 128 * 1024
STAGE_MAX_LANES = 4
STAGE_MAX_NB = 16
STAGE_CT = 7 + STAGE_MAX_LANES + STAGE_MAX_NB + 5 + 3  # agrn_worker_args.hpp stage block (+ raw-first)
# EXPERIMENT gather_raw_first: the bulk gathers RAW x right after the fence; every chip normalizes each landed slot.
RAW_FIRST = 1
NOC_READER = ttnn.NOC.RISCV_0_default  # NoC0 (fabric_all_gather's convention)
NOC_WRITER = ttnn.NOC.RISCV_1_default  # NoC1


# ---------------------------------------------------------------------------------------------------------------------
# Geometry and work split
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    W_local: int
    M: int
    num_images: int
    R: int  # tile-rows of the slice (leading dims folded)
    Wlt: int  # tile-columns of the local slice
    G: int
    W: int
    Wt: int  # tile-columns of the gathered output
    w_valid_last: int  # valid columns of the last local tile (32 if aligned)

    @property
    def aligned(self) -> bool:
        return self.W_local % TILE == 0


def derive_geometry(shape, G) -> Geometry:
    shape = [int(s) for s in shape]
    W_local, M = shape[-1], shape[-2]
    num_images = math.prod(shape[:-2]) if len(shape) > 2 else 1
    R = num_images * (-(-M // TILE))
    Wlt = -(-W_local // TILE)
    W = G * W_local
    return Geometry(W_local, M, num_images, R, Wlt, G, W, -(-W // TILE), W_local - TILE * (Wlt - 1))


@dataclass(frozen=True)
class Split:
    k: int
    n_groups: int
    rows_hi: int
    big: int  # groups with rows_hi tile-rows; the rest rows_hi - 1
    cw_nom: int
    last_crt: int = 0  # tile-rows of the LAST group (sharded: the ragged last shard); 0 = by rows_hi / big

    @property
    def last_rows(self):
        if self.last_crt:
            return self.last_crt
        return self.rows_hi if self.big == self.n_groups else self.rows_hi - 1

    @property
    def n_active(self):
        return self.n_groups * self.k

    def member_width(self, m, Wlt):
        return max(0, min(self.cw_nom, Wlt - m * self.cw_nom))


def stats_k_cap(R, G, nstats):
    """Largest k whose stats table (G * R * k * nstats records) fits STATS_TABLE_CAP (Mechanism caps)."""
    return STATS_TABLE_CAP // (G * R * nstats * RECORD_BYTES)


def select_split(R, Wlt, Nw, G, k_cap) -> Split:
    """op_design.md Work Distribution: cost = rows * (slice + COMB_TILE_COST_FUSED * ceil(G*k/32)), ties -> smaller k.
    k_cap already folds in P_CAP / G, the LN one-gather-tile rule and the stats-table cap (stats_k_cap)."""
    best = None
    for k in range(1, max(1, min(Wlt, k_cap)) + 1):
        groups = min(Nw // k, R)
        if groups == 0:
            break
        cost = (-(-R // groups)) * ((-(-Wlt // k)) + COMB_TILE_COST_FUSED * (-(-(G * k) // 32)))
        if best is None or cost < best[0]:
            best = (cost, k)
    k = best[1]
    k = -(-Wlt // (-(-Wlt // k)))  # no empty member
    n_groups = min(Nw // k, R)
    rows_hi = -(-R // n_groups)
    big = R - n_groups * (rows_hi - 1)
    return Split(k, n_groups, rows_hi, big, -(-Wlt // k))


def _ordered_shard_cores(grid, row_major: bool):
    """Cores of a shard grid in shard-assignment order (row-major or column-major) — base op mechanism."""
    cores = set()
    for r in grid.ranges():
        for y in range(r.start.y, r.end.y + 1):
            for x in range(r.start.x, r.end.x + 1):
                cores.add((x, y))
    return sorted(cores, key=(lambda c: (c[1], c[0])) if row_major else (lambda c: (c[0], c[1])))


def shard_cores(tensor):
    """All cores of a sharded tensor's shard grid (empty for an interleaved tensor)."""
    mc = tensor.memory_config()
    if not mc.is_sharded():
        return set()
    return set(_ordered_shard_cores(mc.shard_spec.grid, True))


def select_sharded_split(geo: Geometry, input_tensor):
    """HEIGHT / WIDTH_SHARDED input: the shard spec IS the work split, block = the resident shard (zero-copy cb_x).

    HEIGHT: k = 1, one row-group per shard (ragged last shard -> last_crt). WIDTH: one group (all R tile-rows),
    k = number of W shards — the k members' partials are pre-combined on the chip by row owners (pre_combine), so
    the stats table holds one record per (chip, row, stat) whatever k is. Returns (Split, worker cores in
    worker-index order g * k + m)."""
    mc = input_tensor.memory_config()
    ss = mc.shard_spec
    sh, sw = int(ss.shape[0]), int(ss.shape[1])
    if sh % TILE or sw % TILE:
        raise RuntimeError(f"all_gather_rms_norm: TILE shard {sh}x{sw} is not a whole number of tiles")
    unit_h, unit_w = sh // TILE, sw // TILE
    order = _ordered_shard_cores(ss.grid, ss.orientation == ttnn.ShardOrientation.ROW_MAJOR)
    if mc.memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED:
        if unit_w != geo.Wlt:
            raise RuntimeError(f"all_gather_rms_norm: HEIGHT shard width {sw} != the padded slice width")
        n_h = -(-geo.R // unit_h)
        if len(order) < n_h:
            raise RuntimeError(f"all_gather_rms_norm: {n_h} row shards but only {len(order)} shard-grid cores")
        split = Split(1, n_h, unit_h, n_h, geo.Wlt, geo.R - (n_h - 1) * unit_h)
        return split, order[:n_h]
    if mc.memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED:
        if unit_h < geo.R:
            raise RuntimeError(f"all_gather_rms_norm: WIDTH shard height {sh} does not cover the {geo.R} tile-rows")
        k = -(-geo.Wlt // unit_w)
        # a ragged last shard keeps the nominal row pitch (unit_w): the kernels address x with that pitch (strided)
        if len(order) < k:
            raise RuntimeError(f"all_gather_rms_norm: {k} W shards but only {len(order)} shard-grid cores")
        return Split(k, 1, geo.R, 1, unit_w, geo.R), order[:k]
    raise RuntimeError(f"all_gather_rms_norm: memory layout {mc.memory_layout} is not built")


# ---------------------------------------------------------------------------------------------------------------------
# Mesh plan: groups, schedules, ports (cached per mesh / axis / topology / links)
# ---------------------------------------------------------------------------------------------------------------------

_PLAN_CACHE = {}


def _node(mesh_device, coord):
    return mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def usable_links(mesh_device, cluster_axis, ring):
    """min over the group hops of the forwarding links (None if a needed hop has none)."""
    mesh_shape = tuple(mesh_device.shape)
    best = None
    for grp in _fag.build_groups(mesh_shape, cluster_axis):
        G = len(grp)
        hops = [(grp[i], grp[i + 1]) for i in range(G - 1)]
        if ring:
            hops.append((grp[G - 1], grp[0]))
        for a, b in hops:
            for x, y in ((a, b), (b, a)):
                n = len(ttnn.get_forwarding_link_indices(_node(mesh_device, x), _node(mesh_device, y)))
                best = n if best is None else min(best, n)
    return best


def _eth_physical(mesh_device, coord, channel):
    """Physical NoC (x, y) of the Ethernet core of a fabric channel (cluster descriptor + arch SoC descriptor)."""
    eth_list, _, _ = _fag._chip_maps(mesh_device, coord)
    if eth_list is None or channel >= len(eth_list):
        return None
    return eth_list[channel]


def build_plan(mesh_device, cluster_axis, topology, num_links, excluded=frozenset()):
    """excluded: logical cores the ports may not take (the input shard grid: those cores are the workers)."""
    excluded = frozenset(excluded)
    key = (id(mesh_device), cluster_axis, int(topology == ttnn.Topology.Ring), num_links, excluded)
    if key in _PLAN_CACHE:
        return _PLAN_CACHE[key][1]
    mesh_shape = tuple(mesh_device.shape)
    groups = _fag.build_groups(mesh_shape, cluster_axis)
    G = len(groups[0])
    ring = topology == ttnn.Topology.Ring and G > 2  # a 2-chip ring is the line
    grid = mesh_device.compute_with_storage_grid_size()
    allowed = {(x, y) for x in range(grid.x) for y in range(grid.y)} - excluded
    if len(allowed) < 2 * num_links:
        raise RuntimeError(
            f"all_gather_rms_norm: {len(allowed)} cores outside the input shard grid, {2 * num_links} port cores needed"
        )
    chips = {}
    for grp in groups:
        for p, coord in enumerate(grp):
            chips[coord] = dict(
                p=p,
                G=G,
                next=grp[(p + 1) % G] if (ring or p < G - 1) else None,
                prev=grp[(p - 1) % G] if (ring or p > 0) else None,
            )
    for coord, ch in chips.items():
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            if peer is None:
                ch[f"{d}_links"] = []
                continue
            links = ttnn.get_forwarding_link_indices(_node(mesh_device, coord), _node(mesh_device, peer))
            if len(links) < num_links:
                raise ValueError(
                    f"all_gather_rms_norm: {coord} -> {peer} has {len(links)} usable link(s), {num_links} requested"
                )
            ch[f"{d}_links"] = list(links[:num_links])
    # Port placement: the free core with the fewest NoC1 hops to the link's Ethernet core (fabric_all_gather.plan);
    # the channel comes from the connection's own runtime args (no probe program).
    for coord, ch in chips.items():
        taken = set()
        ports = {}
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            for l in range(num_links):
                core = None
                if peer is not None:
                    dummy = ttnn.ProgramDescriptor()
                    args = ttnn.setup_fabric_connection(
                        _node(mesh_device, coord),
                        _node(mesh_device, peer),
                        ch[f"{d}_links"][l],
                        dummy,
                        ttnn.CoreCoord(0, 0),
                    )
                    exy = _eth_physical(mesh_device, coord, int(args[0]))
                    if exy is not None:
                        core = _fag._worker_nearest_noc1(mesh_device, coord, exy, taken, allowed)
                ports[(d, l)] = core
                if core is not None:
                    taken.add((core.x, core.y))
        for key2, core in ports.items():
            if core is None:
                core = _fag._worker_below(mesh_device, coord, 0, taken, allowed)
                ports[key2] = core
                taken.add((core.x, core.y))
        ch["ports"] = ports
    plan = dict(chips=chips, G=G, ring=ring, num_links=num_links, grid=(grid.x, grid.y))
    _PLAN_CACHE[key] = (mesh_device, plan)
    return plan


# ---------------------------------------------------------------------------------------------------------------------
# Chunk walk (host replica of agrn_port_common.hpp for_each_chunk: counts only)
# ---------------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Walk:
    nb: int
    rows_eff: int
    cols_eff: int
    row_stride: int
    slot_stride: int
    run_pages: int
    scatter_runs: int
    shard_w: int = 0  # > 0: sharded output — lanes are tile-rows, runs split at shard_w-tile column boundaries
    split_runs: int = 1  # agrn_port_common.hpp split_runs

    def _add(self, st, n):
        """Append a run of n pages to the open packet st = [count, runs, pages] (agrn_port_common.hpp add_run)."""
        if not self.split_runs:
            if st[1] == self.scatter_runs or st[2] + n > self.run_pages:
                st[0] += 1
                st[1] = st[2] = 0
            st[1] += 1
            st[2] += n
            return
        while n:
            if st[1] == self.scatter_runs or st[2] == self.run_pages:
                st[0] += 1
                st[1] = st[2] = 0
            m = min(n, self.run_pages - st[2])
            st[1] += 1
            st[2] += m
            n -= m

    def chunks(self, q, part, first_bank, stride):
        """Packets of slot q for one lane / part (host replica of agrn_port_common.hpp for_each_packet)."""
        if self.shard_w:
            return self._chunks_sharded(q, part, first_bank, stride)
        if first_bank >= self.nb:
            return 0
        nb = self.nb
        nl = (nb - first_bank + stride - 1) // stride
        lo = nl // 2 if part == 2 else 0
        hi = nl // 2 if part == 1 else nl
        st = [0, 0, 0]
        base = q * self.slot_stride
        for i in range(lo, hi):
            if i == nl // 2 and st[1]:
                st = [st[0] + 1, 0, 0]
            b = first_bank + i * stride
            for rr in range(self.rows_eff):
                rb = base + rr * self.row_stride
                jj = (b - rb) % nb
                while jj < self.cols_eff:
                    n = min((self.cols_eff - 1 - jj) // nb + 1, self.run_pages)
                    self._add(st, n)
                    jj += n * nb
        return st[0] + (1 if st[1] else 0)

    def _chunks_sharded(self, q, part, first_row, stride):
        if first_row >= self.rows_eff:
            return 0
        nl = (self.rows_eff - first_row + stride - 1) // stride
        lo = nl // 2 if part == 2 else 0
        hi = nl // 2 if part == 1 else nl
        st = [0, 0, 0]
        base = q * self.slot_stride
        for i in range(lo, hi):
            if i == nl // 2 and st[1]:
                st = [st[0] + 1, 0, 0]
            jj = 0
            while jj < self.cols_eff:
                gc = base + jj
                seg_end = min(self.cols_eff, (gc // self.shard_w + 1) * self.shard_w - base)
                n = min(seg_end - jj, self.run_pages)
                self._add(st, n)
                jj += n
        return st[0] + (1 if st[1] else 0)


def _inc_sim(counts, inc_every):
    sent = incs = 0
    S0, I0 = [], []
    for n in counts:
        S0.append(sent)
        I0.append(incs)
        for i in range(n):
            sent += 1
            if sent % inc_every == 0 or i == n - 1:
                incs += 1
    return S0, I0, incs


# ---------------------------------------------------------------------------------------------------------------------
# Persistent resources (created once per mesh / port set, outside any trace)
# ---------------------------------------------------------------------------------------------------------------------

_RES_CACHE = {}


def _single_cores(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(x, y), ttnn.CoreCoord(x, y)) for x, y in sorted(cores)])


def global_semaphores(mesh_device, plan):
    port_cores = {(c.x, c.y) for ch in plan["chips"].values() for c in ch["ports"].values()}
    key = ("sems", id(mesh_device), tuple(sorted(port_cores)))
    if key not in _RES_CACHE:
        cores = _single_cores(port_cores)
        sems = tuple(ttnn.create_global_semaphore(mesh_device, cores, 0) for _ in range(3 + OWN_BANDS_MAX + 1))
        ttnn.synchronize_device(mesh_device)  # every chip's counters are 0 before any chip's first call
        _RES_CACHE[key] = (mesh_device, sems)
    # ready, bulk, stats, then OWN_BANDS_MAX own-ready band counters, then the own-entry staging readiness counter
    return [int(ttnn.get_global_semaphore_address(s)) for s in _RES_CACHE[key][1]]


def raw_landing_buffer(mesh_device, output_tensor):
    """gather-raw-first: DRAM interleaved scratch with the output's shape / page layout (remote raw slots land here)."""
    shape = tuple(int(s) for s in output_tensor.shape)
    key = ("raw", id(mesh_device), shape, output_tensor.dtype)
    if key not in _RES_CACHE or not _RES_CACHE[key][1].is_allocated():
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape(list(shape)), output_tensor.dtype, ttnn.TILE_LAYOUT, mesh_device, ttnn.DRAM_MEMORY_CONFIG
        )
        _RES_CACHE[key] = (mesh_device, t)
    return _RES_CACHE[key][1]


def staging_buffer(mesh_device, geo: Geometry, dtype):
    key = ("staging", id(mesh_device), geo.G, geo.R, geo.Wlt, dtype)
    # Re-create a cached buffer that is no longer allocated: the golden conftest deallocates the device tensors a
    # failed call's frames hold, which includes this cached one.
    if key not in _RES_CACHE or not _RES_CACHE[key][1].is_allocated():
        t = ttnn.allocate_tensor_on_device(
            ttnn.Shape([1, 1, geo.G * geo.R * TILE, geo.Wlt * TILE]),
            dtype,
            ttnn.TILE_LAYOUT,
            mesh_device,
            ttnn.DRAM_MEMORY_CONFIG,
        )
        _RES_CACHE[key] = (mesh_device, t)
    return _RES_CACHE[key][1]


# ---------------------------------------------------------------------------------------------------------------------
# Program descriptor
# ---------------------------------------------------------------------------------------------------------------------


def _f32_bits(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _datum_bytes(dtype):
    return {ttnn.float32: 4, ttnn.bfloat16: 2, ttnn.bfloat8_b: 1}[dtype]


def _cb(index, num_pages, page_size, dtype, core_ranges):
    return ttnn.CBDescriptor(
        total_size=int(num_pages * page_size),
        core_ranges=core_ranges,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=int(page_size))],
    )


def _dm(processor, noc):
    return ttnn.DataMovementConfigDescriptor(processor=processor, noc=noc)


def _schedule(p, G, ring, balance):
    return _fag._schedule(p, G, ring, balance)


def create_mesh_program_descriptor(
    mesh_device,
    plan,
    geo: Geometry,
    input_tensor,
    output_tensor,
    gamma,
    beta,
    *,
    is_ln,
    epsilon,
    compute_kernel_config,
):
    chips, G, ring, L = plan["chips"], plan["G"], plan["ring"], plan["num_links"]
    gx, gy = plan["grid"]
    n_ports = 2 * L
    Nw = gx * gy - n_ports
    k_cap = P_CAP // G
    nstats = 3 if is_ln else 1
    if is_ln:
        k_cap = min(k_cap, 32 // G)  # LN combine: one gather tile per stat (P <= 32)
    k_cap = max(1, min(k_cap, stats_k_cap(geo.R, G, nstats)))
    x_sharded = input_tensor.memory_config().is_sharded()
    if x_sharded:
        # k is fixed by the shard count (WIDTH: pre-combined on the chip, so no stats-table cap on k)
        split, shard_order = select_sharded_split(geo, input_tensor)
    else:
        split = select_split(geo.R, geo.Wlt, Nw, G, k_cap)
        shard_order = None
    k = split.k
    # WIDTH_SHARDED: row-owner pre-combine (op_design.md Refinement 1b). Member m owns tile-rows [m*rpo, ...): it sums
    # the k members' partials of its rows into ONE record per (row, stat) before the fabric round (table: kt = 1
    # member per row), combines the G chips' records after it, and returns final per-row stats to all k members.
    pre_combine = x_sharded and input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.WIDTH_SHARDED
    kt = 1 if pre_combine else k  # members per (slot, row) in the stats table
    P = G * kt
    gather_tiles = -(-P // 32)  # gather tiles of the cross-chip combine
    gt_pc = -(-k // 32) if pre_combine else 0  # gather tiles of the on-chip pre-combine (P = k)
    gt_max = max(gather_tiles, gt_pc)
    rpo = -(-geo.R // k) if pre_combine else 0  # tile-rows per row owner
    n_owners = -(-geo.R // rpo) if pre_combine else 0
    nfinal = 2 if is_ln else 1  # final per-row stats returned to the members (inv | mean, rstd)
    crt_max = split.rows_hi
    cw_max = split.cw_nom
    slot_bytes = geo.R * kt * nstats * RECORD_BYTES
    table_bytes = G * slot_bytes
    if table_bytes > STATS_TABLE_CAP:
        raise RuntimeError(
            f"all_gather_rms_norm: stats table {table_bytes} B exceeds STATS_TABLE_CAP {STATS_TABLE_CAP} B "
            f"(R={geo.R}, k={k}, nstats={nstats}); a row-windowed stats round (R_WIN) is deferred"
        )

    x_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    # AGRN_FORCE_STAGED=1: perf probe only — run an aligned slice through the R_U staged regime
    is_ru = (not geo.aligned) or os.environ.get("AGRN_FORCE_STAGED") == "1"
    fp32_dest = bool(compute_kernel_config.fp32_dest_acc_en)
    acc_dtype = ttnn.float32 if fp32_dest else ttnn.bfloat16  # T_acc: intermediate CB formats follow the DEST width
    T_acc = ttnn.tile_size(acc_dtype)
    T_in = ttnn.tile_size(x_dtype)
    out_page = int(output_tensor.buffer_aligned_page_size())
    T32 = ttnn.tile_size(ttnn.float32)
    has_gamma, has_beta = gamma is not None, beta is not None
    g_dtype = gamma.dtype if has_gamma else ttnn.bfloat16
    g_eb = _datum_bytes(g_dtype)
    gamma_page = 2 * 16 * 16 * g_eb  # faces 0/1 of a gamma tile (row 0 of each is read)
    gamma_is_tile = has_gamma and gamma.layout == ttnn.TILE_LAYOUT
    gamma_src_page = (TILE * TILE * g_eb) if gamma_is_tile else TILE * g_eb
    x_is_f32 = x_dtype == ttnn.float32
    # sharded input with a W tail: the padding of the caller's shard is never zeroed — masked in compute instead
    mask_tail = input_tensor.memory_config().is_sharded() and not geo.aligned
    # pre_combine: the per-row Float32 CBs hold one row CHUNK (row_chunk tile-rows), never crt = R rows; the chunk is
    # the largest that keeps them within ROW_CHUNK_L1_BUDGET. Elsewhere the chunk is the whole block (one pass).
    if pre_combine:
        if is_ln:
            per_row = (nstats * (1 + 2 * gt_max) + 6 + gt_max + 2) * T32 + cw_max * T_acc
        else:
            per_row = (1 + 2 * gt_max + 2) * T32 + (cw_max * T_acc if mask_tail else 0)
        cap = max(1, min(geo.R, _knob("ROW_CHUNK_L1_BUDGET", ROW_CHUNK_L1_BUDGET) // per_row))
        # Every block pushed into a per-row CB must stay contiguous in its ring: all chunks have exactly row_chunk
        # rows (a divisor of R) and every owner a multiple of them (rpo rounded up to row_chunk; the last owner's
        # R - (n_owners - 1) * rpo rows is then a multiple too).
        row_chunk = max(d for d in range(1, cap + 1) if geo.R % d == 0)
        rpo = -(-rpo // row_chunk) * row_chunk
        n_owners = -(-geo.R // rpo)
    else:
        row_chunk = crt_max

    # x * gamma in the stats shadow (OVERLAP): RMS with gamma, dense block (the WIDTH pre-combine normalizes per row
    # chunk from a strided shard — not built there)
    xg_pre = bool(
        _knob("XG_PRE", XG_PRE)
        and not is_ln
        and has_gamma
        and not pre_combine
        and crt_max * cw_max * T_acc <= _knob("XG_L1_BUDGET", XG_L1_BUDGET)
    )

    # destination of the slices: final output pages (aligned) or the slot-contiguous staging scratch (R_U)
    staging = staging_buffer(mesh_device, geo, out_dtype) if is_ru else None
    dst_tensor = staging if is_ru else output_tensor
    dram = mesh_device.dram_grid_size()
    nb = dram.x * dram.y
    run_pages = max(1, ttnn.get_tt_fabric_max_payload_size_bytes() // out_page)
    batch_chunks = _knob("BATCH_CHUNKS", BATCH_CHUNKS)
    inc_every = _knob("INC_EVERY", INC_EVERY)
    scatter_runs = _knob("SCATTER_RUNS", SCATTER_RUNS)
    read_depth = max(1, min(_knob("READ_DEPTH", READ_DEPTH), batch_chunks, 15))
    own_from_dst = _knob("OWN_SRC_DST", OWN_SRC_DST)
    split_runs = _knob("SPLIT_RUNS", SPLIT_RUNS)
    payload_bytes = run_pages * out_page
    stats_payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    out_sharded = output_tensor.memory_config().is_sharded()
    # HEIGHT in -> HEIGHT out, one tile-row per shard on the same cores: the own slice of a worker's output shard is
    # Wlt contiguous pages, so cb_out_block IS that slice (normalize packs into its final place — no block, no own
    # write; the ports pull it from there)
    out_in_place = False
    if (
        x_sharded
        and out_sharded
        and not is_ru
        and input_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        and output_tensor.memory_config().memory_layout == ttnn.TensorMemoryLayout.HEIGHT_SHARDED
        and split.rows_hi == 1
    ):
        o_ss = output_tensor.memory_config().shard_spec
        o_order = _ordered_shard_cores(o_ss.grid, o_ss.orientation == ttnn.ShardOrientation.ROW_MAJOR)
        out_in_place = int(o_ss.shape[0]) == TILE and o_order[: len(shard_order)] == list(shard_order)
    if is_ru:
        walk = Walk(nb, 1, geo.R * geo.Wlt, geo.R * geo.Wlt, geo.R * geo.Wlt, run_pages, scatter_runs, 0, split_runs)
    elif out_sharded:
        # final pages in L1 shards: a run is a tile-row's columns inside one shard core (HEIGHT: the whole slot row)
        out_ss = output_tensor.memory_config().shard_spec
        shard_w = int(out_ss.shape[1]) // TILE
        walk = Walk(nb, geo.R, geo.Wlt, geo.Wt, geo.Wlt, run_pages, scatter_runs, shard_w, split_runs)
    else:
        walk = Walk(nb, geo.R, geo.Wlt, geo.Wt, geo.Wlt, run_pages, scatter_runs, 0, split_runs)
    balance = ring and G % 2 == 0 and G >= 4 and geo.R * geo.Wlt >= BALANCE_MIN_TILES

    # R_U assembly: max distinct source tiles per output tile
    src_max = 1
    if is_ru:
        for oc in range(geo.Wt):
            pages = set()
            for c in range(TILE):
                g_ = oc * TILE + c
                if g_ >= geo.W:
                    break
                q = g_ // geo.W_local
                pages.add((q, (g_ - q * geo.W_local) // TILE))
            src_max = max(src_max, len(pages))

    ready_addr, bulk_addr, stats_addr, *band_addrs, stage_ctr_addr = global_semaphores(mesh_device, plan)
    # own-ready bands (OVERLAP): band of row-group g = g * own_bands // n_groups (same formula on the workers); the
    # staged (R_U) slot is flat and the WIDTH pre-combine has one group: one counter (the program semaphore) there
    stage_on = L <= STAGE_MAX_LANES and (walk.shard_w > 0 or nb <= STAGE_MAX_NB)  # CT tables cap (feasibility)
    own_bands = (
        1
        if (is_ru or pre_combine or stage_on)
        else max(1, min(_knob("OWN_BANDS", OWN_BANDS), split.n_groups, OWN_BANDS_MAX))
    )
    band_need = [0] * own_bands  # announcements per band: k per row-group
    band_row_end = [0] * own_bands  # first tile-row after the band
    row = 0
    for g in range(split.n_groups):
        b = g * own_bands // split.n_groups
        row += split.rows_hi if g < split.big else split.rows_hi - 1
        if g == split.n_groups - 1:
            row = geo.R
        band_need[b] += k
        band_row_end[b] = row
    band_addrs = band_addrs[:own_bands]

    # ---- worker compile-time args (agrn_worker_args.hpp) ----
    worker_ct = [
        geo.R,
        geo.Wlt,
        k,
        split.n_groups,
        split.rows_hi,
        split.big,
        split.cw_nom,
        gx,
        G,
        nstats,
        gather_tiles,
        int(is_ln),
        int(has_gamma),
        int(has_beta),
        int(gamma_is_tile),
        T_in,
        out_page,
        g_eb,
        geo.w_valid_last,
        geo.W_local,
        geo.Wt,
        int(is_ru),
        int(x_is_f32),
        _datum_bytes(out_dtype),
        n_ports,
        src_max,
        gamma_page,
        gamma_src_page,
        int(x_sharded),  # worker identity from the shard map (common args), zero-copy cb_x, masked W tail
        split.last_rows,
        int(out_in_place),
        int(pre_combine),
        row_chunk,
        rpo,
        own_bands,
        int(xg_pre),
    ]
    worker_ct_base = list(worker_ct)
    # Sharded: the local shard is consumed in place, never through an accessor -> empty accessor args.
    acc_ct = list(
        (ttnn.TensorAccessorArgs() if x_sharded else ttnn.TensorAccessorArgs(input_tensor)).get_compile_time_args()
    )
    acc_ct += list(ttnn.TensorAccessorArgs(dst_tensor).get_compile_time_args())
    acc_ct += list(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    acc_ct += list((ttnn.TensorAccessorArgs(gamma) if has_gamma else ttnn.TensorAccessorArgs()).get_compile_time_args())
    acc_ct += list((ttnn.TensorAccessorArgs(beta) if has_beta else ttnn.TensorAccessorArgs()).get_compile_time_args())

    # member widths in elements -> 1/n (LN local mean)
    inv_n = []
    for m in range(k):
        w_tiles = split.member_width(m, geo.Wlt)
        n = w_tiles * TILE - ((TILE - geo.w_valid_last) if m * split.cw_nom + w_tiles == geo.Wlt else 0)
        inv_n.append(_f32_bits(1.0 / max(1, n)))
    inv_wl = _f32_bits(1.0 / geo.W_local)  # pre_combine: the row owner's slice mean

    # ---- port compile-time args (agrn_port_common.hpp) ----
    port_ct = [
        CB_PORT_CHUNKS,
        out_page,
        run_pages,
        batch_chunks,
        inc_every,
        nb,
        walk.rows_eff,
        walk.cols_eff,
        walk.row_stride,
        walk.slot_stride,
        int(is_ru),
        geo.Wlt,
        k,
        split.n_groups,
        split.rows_hi,
        split.big,
        split.cw_nom,
        stats_payload,
        scatter_runs,
        walk.shard_w,
        split.last_rows,
        read_depth,
        own_from_dst,
        CB_PORT_DESC,
        split_runs,
        own_bands,
        _knob("EAGER_PUSH", EAGER_PUSH),
        CB_STAGE,
        0,  # stage_on (set per chip below)
    ] + list(ttnn.TensorAccessorArgs(dst_tensor).get_compile_time_args())
    port_ct_base = list(port_ct)

    # ---- own-entry staging plan (per chip: the own slot's walk depends on p) ----
    stage_cp = run_pages  # pages per staging chunk

    def stage_plan(p):
        """Per lane l: unit page counts (walk order), staged unit count (a prefix within the L1 budget), per-bank lane
        offsets; the chunk count and cb_stage slots per core."""
        if not stage_on:
            return None
        base = p * walk.slot_stride
        unit_off = [0] * STAGE_MAX_NB
        lane_pages = []
        for l in range(L):
            if walk.shard_w:
                units = list(range(l, walk.rows_eff, L))
                pages = [walk.cols_eff for _ in units]
            else:
                units = list(range(l, nb, L))
                pages = []
                for b in units:
                    c = 0
                    for rr in range(walk.rows_eff):
                        rb = base + rr * walk.row_stride
                        j0 = (b - rb) % nb
                        c += (walk.cols_eff - 1 - j0) // nb + 1 if j0 < walk.cols_eff else 0
                    pages.append(c)
                off = 0
                for b, c in zip(units, pages):
                    unit_off[b] = off
                    off += c
            lane_pages.append(pages)
        # cap: at most max_slots chunks per core over the active workers, whole lane units (a lane's staged prefix)
        max_slots = max(1, _knob("STAGE_L1_BUDGET", STAGE_L1_BUDGET) // (stage_cp * out_page))
        cap = (max_slots * split.n_active // L) * stage_cp
        lanes = []
        for pages in lane_pages:
            n_st, tot = 0, 0
            while n_st < len(pages) and tot + pages[n_st] <= cap:
                tot += pages[n_st]
                n_st += 1
            lanes.append(dict(n_st=n_st, tot=tot))
        if max(ln["tot"] for ln in lanes) == 0:
            return None
        nch_max = max(-(-ln["tot"] // stage_cp) for ln in lanes)
        slots = -(-(L * nch_max) // split.n_active)
        return dict(lanes=lanes, unit_off=unit_off, max_pages=slots * stage_cp, nch_max=nch_max)

    stage_plans = {coord: stage_plan(ch["p"]) for coord, ch in chips.items()}
    stage_max_pages = max([sp["max_pages"] for sp in stage_plans.values() if sp] + [0])

    # ---- gather-raw-first (EXPERIMENT) ----
    full_staged = all(
        sp is not None and all(ln["n_st"] == len(range(l, nb, L)) for l, ln in enumerate(sp["lanes"]))
        for sp in stage_plans.values()
    )
    raw_on = bool(
        _knob("RAW_FIRST", RAW_FIRST)
        and not is_ru
        and not out_sharded
        and not x_sharded
        and not pre_combine
        and not xg_pre
        and own_bands == 1
        and x_dtype == out_dtype
        and full_staged
    )
    if _knob("RAW_FIRST", RAW_FIRST) and not raw_on and os.environ.get("AGRN_RAW_STRICT") == "1":
        raise RuntimeError("gather_raw_first: path not expressible for this configuration")
    raw_t = raw_landing_buffer(mesh_device, output_tensor) if raw_on else None
    raw_maxs = 2 * L
    port_dst_addr = int(raw_t.buffer_address()) if raw_on else int(dst_tensor.buffer_address())

    # ---- compute config ----
    cfg = ttnn.ComputeConfigDescriptor()
    cfg.math_fidelity = compute_kernel_config.math_fidelity
    cfg.fp32_dest_acc_en = fp32_dest
    cfg.math_approx_mode = compute_kernel_config.math_approx_mode
    modes = [ttnn.UnpackToDestMode.Default] * 64
    # UnpackToDestFp32 needs a 32-bit DEST: under 16-bit DEST (the caller's choice) nothing is tagged, the records stay
    # Float32 pages (wire contract) and every DEST-resident sum is 16-bit (op_design.md Key Risks).
    tagged = []
    if fp32_dest:
        tagged = [CB_GATHER_ROWS]  # exact fp32 copy into DEST for the transpose
        if is_ln:
            tagged += [CB_GATHERED, CB_STAT_CONSTS, CB_LN_TMP, CB_MEAN_COL]
            if pre_combine:
                tagged += [CB_STAT_CONSTS_X, CB_FINAL_MEAN]
            if not mask_tail and (x_is_f32 or not x_sharded):
                # exact x in DEST (the Float32 x block); a sharded W tail's masked sums read x through srcA (DestReuse
                # mask multiply), which a tagged CB cannot feed — fp32 x is truncated to tf32 there
                tagged.append(CB_X if x_is_f32 else CB_X32)
                if raw_on and x_is_f32:
                    tagged.append(CB_XR)  # remote raw fp32 x exact in DEST, as the own block
                if pre_combine and x_is_f32:
                    tagged.append(CB_X_STATS)
        else:
            # RMS: the local collapse and the cross-chip combine sum the partials in fp32 (Accurate SFPU reduce
            # reading these CBs exactly): "per-row partial sums MUST stay fp32 on the wire and when summed" (prompt)
            tagged += [CB_SUMSQ, CB_GATHERED]
    for cb in tagged:
        modes[cb] = ttnn.UnpackToDestMode.UnpackToDestFp32
    cfg.unpack_to_dest_mode = modes

    full = ttnn.CoreRangeSet([ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(gx - 1, gy - 1))])
    # per-stage profiler zones (perf probes only; the harness needs them off)
    defines = [("AGRN_PROFILE_ZONES", "1")] if os.environ.get("AGRN_PROFILE_ZONES") == "1" else []
    # perf ablations (stubbed payloads, output garbage): env AGRN_ABLATE="NO_BULK,NO_OWN_WAIT" -> AGRN_ABL_<name>
    defines += [(f"AGRN_ABL_{n.strip()}", "1") for n in os.environ.get("AGRN_ABLATE", "").split(",") if n.strip()]
    in_addr = int(input_tensor.buffer_address())
    out_addr = int(output_tensor.buffer_address())
    dst_addr = int(dst_tensor.buffer_address())
    g_addr = int(gamma.buffer_address()) if has_gamma else 0
    b_addr = int(beta.buffer_address()) if has_beta else 0

    mesh_desc = ttnn.MeshProgramDescriptor()
    for coord, ch in chips.items():
        p = ch["p"]
        sp = stage_plans[coord]
        st_lanes = [0] * STAGE_MAX_LANES
        if sp:
            for l in range(L):
                st_lanes[l] = sp["lanes"][l]["n_st"]
        worker_ct = (
            worker_ct_base
            + [
                int(sp is not None),
                L,
                nb,
                walk.cols_eff,
                walk.row_stride,
                walk.slot_stride,
                int(walk.shard_w > 0),
            ]
            + st_lanes
            + (sp["unit_off"] if sp else [0] * STAGE_MAX_NB)
            + [
                stage_cp,
                sp["nch_max"] if sp else 1,
                split.n_active,
                0,  # common-arg index of the staging core table (patched below)
                stage_ctr_addr,
                int(raw_on),
                raw_maxs,
                0,  # common-arg index of the raw-first slot list (patched below)
            ]
        )
        ST_TBL_CT = len(worker_ct) - 5
        RAW_CRT_CT = len(worker_ct) - 1
        assert len(worker_ct) == len(worker_ct_base) + STAGE_CT
        port_ct = list(port_ct_base)
        port_ct[28] = int(sp is not None)
        ports = ch["ports"]
        port_list = [ports[(d, l)] for d in ("fwd", "bwd") for l in range(L)]
        port_set = {(c.x, c.y) for c in port_list}
        if x_sharded:
            active = list(shard_order)
            assert not (set(active) & port_set), "port cores must lie outside the input shard grid"
        else:
            workers = [(x, y) for y in range(gy) for x in range(gx) if (x, y) not in port_set]
            active = workers[: split.n_active]
        virt = lambda c: mesh_device.worker_core_from_logical_core(ttnn.CoreCoord(*c))
        pack = lambda v: (v.x & 0xFFFF) | (v.y << 16)
        active_noc = [pack(virt(c)) for c in active]
        hub = {d: virt((ports[(d, 0)].x, ports[(d, 0)].y)) for d in ("fwd", "bwd")}
        mc0, mc1 = virt((0, 0)), virt((gx - 1, gy - 1))  # program grid rectangle (NoC0 order)

        stats_sched = {}
        bulk_sched = {}
        for cc in chips:  # schedules of every chip (for upstream lookups)
            pp = chips[cc]["p"]
            f, b = _schedule(pp, G, ring, False)
            stats_sched[cc] = {"fwd": [q for q, _ in f], "bwd": [q for q, _ in b]}
            f, b = _schedule(pp, G, ring, balance)
            bulk_sched[cc] = {"fwd": f, "bwd": b}
        # which hub holds each slot
        up_of = {"fwd": ch["prev"], "bwd": ch["next"]}
        recv = {d: (stats_sched[up_of[d]][d] if up_of[d] is not None else []) for d in ("fwd", "bwd")}
        hub_mask = 0
        for q in range(G):
            if q == p:
                continue
            in_f, in_b = q in recv["fwd"], q in recv["bwd"]
            assert in_f or in_b, f"slot {q} never reaches chip {p}"
            if not in_f:
                hub_mask |= 1 << q

        # gather-raw-first: every incoming bulk entry of every port is one landing stream of its slot; per port the
        # arrival count at the entry's end and its flag index; workers process remote slots by first landing entry
        raw_port_in = {}
        raw_order = []
        if raw_on:
            n_streams = [0] * G
            first_e = [1 << 30] * G
            for d in ("fwd", "bwd"):
                up = up_of[d]
                up_sends = bulk_sched[up][d] if up is not None else []
                for l in range(L):
                    counts = [walk.chunks(q, part, l, L) for q, part in up_sends]
                    S0, I0, total = _inc_sim(counts, inc_every)
                    lst = []
                    for e, (q, part) in enumerate(up_sends):
                        assert q != p
                        thr = I0[e + 1] if e + 1 < len(up_sends) else total
                        lst += [thr, q * raw_maxs + n_streams[q]]
                        n_streams[q] += 1
                        first_e[q] = min(first_e[q], e)
                    raw_port_in[(d, l)] = lst
            assert max(n_streams) <= raw_maxs, n_streams
            rem = [q for q in range(G) if q != p]
            assert all(n_streams[q] > 0 for q in rem), (p, n_streams)
            rem.sort(key=lambda q: (first_e[q], min((q - p) % G, (p - q) % G), q))
            raw_order = [q | (n_streams[q] << 16) for q in rem]

        program = ttnn.ProgramDescriptor()
        program.semaphores = [
            ttnn.SemaphoreDescriptor(id=sid, core_ranges=full, initial_value=0)
            for sid in (
                SEM_RECORDS_LOCAL,
                SEM_STATS_READY,
                SEM_OWN_READY,
                SEM_LANDED,
                SEM_STATS_READY_BWD,
                SEM_STATS_DONE,
                SEM_PRE,
                SEM_FINAL,
            )
        ]

        # ---- CBs (all on the full program range: identical addresses on every core of the chip) ----
        if x_sharded:
            cb_x_desc = ttnn.cb_descriptor_from_sharded_tensor(CB_X, input_tensor)  # zero-copy: the shard IS cb_x
            cb_x_desc.core_ranges = full
        else:
            cb_x_desc = _cb(CB_X, crt_max * cw_max, T_in, x_dtype, full)
        cbs = [
            cb_x_desc,
            _cb(CB_SCALER, 1, ttnn.tile_size(ttnn.bfloat16), ttnn.bfloat16, full),
            # per-row CBs hold one row chunk (row_chunk = the whole block unless pre_combine)
            _cb(CB_PARTIAL, row_chunk * nstats, T32, ttnn.float32, full),
            _cb(CB_RECORD_STAGING, row_chunk * nstats, RECORD_BYTES, ttnn.float32, full),
            _cb(CB_GATHER_ROWS, row_chunk * nstats * gt_max, T32, ttnn.float32, full),
            _cb(CB_GATHERED, row_chunk * nstats * gt_max, T32, ttnn.float32, full),
            _cb(CB_INV, row_chunk, T_acc, acc_dtype, full),
            (
                ttnn.cb_descriptor_from_sharded_tensor(
                    CB_OUT_BLOCK, output_tensor, p * geo.Wlt * out_page, geo.Wlt * out_page, full
                )
                if out_in_place
                else _cb(CB_OUT_BLOCK, crt_max * cw_max, out_page, out_dtype, full)
            ),
            _cb(CB_PORT_CHUNKS, 2 * batch_chunks * run_pages, out_page, out_dtype, full),
            _cb(CB_PORT_DESC, 2 * batch_chunks, PORT_DESC_BYTES, ttnn.float32, full),
            *([_cb(CB_STAGE, stage_max_pages, out_page, out_dtype, full)] if stage_max_pages else []),
            # program-scoped: same address on every core of every chip, no persistent L1 (see agrn_common.hpp)
            _cb(CB_STATS_TABLE, table_bytes // RECORD_BYTES, RECORD_BYTES, ttnn.float32, full),
        ]
        n_aff = G if raw_on else 1  # raw-first: own + every remote slot's gamma / beta columns
        if has_gamma:
            cbs.append(_cb(CB_GAMMA, n_aff * cw_max, gamma_page, g_dtype, full))
            cbs.append(_cb(CB_GAMMA_ROWS, 1, n_aff * (cw_max + 2) * 128 + 64, ttnn.float32, full))
        if has_beta:
            cbs.append(_cb(CB_BETA, n_aff * cw_max, gamma_page, g_dtype, full))
        if raw_on:
            cbs += [
                _cb(CB_LANDED, G * raw_maxs, LANDED_FLAG_BYTES, ttnn.float32, full),
                # every remote slot's block at its own offset: a reservation never straddles the ring end (the
                # reader writes and the unpacker indexes the block contiguously)
                _cb(CB_XR, (G - 1) * crt_max * cw_max, T_in, x_dtype, full),
                _cb(CB_OUT_R, crt_max * cw_max, out_page, out_dtype, full),
            ]
        if is_ln or mask_tail:
            # T_acc like cb_ln_sq: the reduce helper's masked fold (interleaved LN T) unpacks the mask as srcB without a
            # format reconfig. Sharded LN on a 16-bit-mantissa x: cb_ln_sq ((x - K)^2 scratch, one block of tiles
            # beside the caller's resident in/out shards) is bf16 — the sharded paths never fold it with the mask.
            sq_dtype = ttnn.bfloat16 if (is_ln and x_sharded and not x_is_f32) else acc_dtype
            cbs += [
                _cb(CB_MASK, cw_max if mask_tail else 1, T_acc, acc_dtype, full),  # sharded tail: the mask block
                _cb(CB_LN_SQ, row_chunk * cw_max, ttnn.tile_size(sq_dtype), sq_dtype, full),
            ]
        if is_ln:
            cbs += [
                _cb(CB_LN_M, row_chunk, T32, ttnn.float32, full),
                _cb(CB_LN_SHIFT, row_chunk, T32, ttnn.float32, full),
                _cb(CB_STAT_CONSTS, gt_pc if pre_combine else gather_tiles, T32, ttnn.float32, full),
                _cb(CB_MEAN_COL, row_chunk, T32, ttnn.float32, full),
                _cb(CB_MEAN_B, row_chunk, T32, ttnn.float32, full),
                _cb(CB_RSTD_B, row_chunk, T_acc, acc_dtype, full),
                _cb(CB_LN_TMP, row_chunk * gt_max, T32, ttnn.float32, full),
            ]
            if pre_combine:
                cbs += [
                    _cb(CB_STAT_CONSTS_X, gather_tiles, T32, ttnn.float32, full),
                    _cb(CB_FINAL_MEAN, row_chunk, T32, ttnn.float32, full),
                ]
            # 16-bit DEST / sharded input: the LN paths read cb_x itself (bf16 / bf8b lands in DEST exactly anyway)
            if not x_is_f32 and fp32_dest and not x_sharded:
                cbs.append(_cb(CB_X32, crt_max * cw_max, T32, ttnn.float32, full))
        else:
            cbs.append(_cb(CB_SUMSQ, row_chunk, T_acc, acc_dtype, full))
        if xg_pre:
            cbs.append(_cb(CB_XG, crt_max * cw_max, T_acc, acc_dtype, full))
        if is_ru:
            cbs.append(_cb(CB_ASSEMBLE, src_max + 1, out_page, out_dtype, full))
        if pre_combine:
            x_stats_desc = ttnn.cb_descriptor_from_sharded_tensor(CB_X_STATS, input_tensor)  # zero-copy, 2nd view
            x_stats_desc.core_ranges = full
            cbs += [
                x_stats_desc,
                _cb(CB_PRE_TABLE, rpo * k * nstats, RECORD_BYTES, ttnn.float32, full),
                _cb(CB_FINAL_TABLE, geo.R * nfinal, RECORD_BYTES, ttnn.float32, full),
                _cb(CB_FINAL_INV, row_chunk, T32, ttnn.float32, full),
            ]
        program.cbs = cbs

        # ---- worker kernels (common runtime args only) ----
        worker_set = full.subtract(_single_cores(port_set))
        common = [
            in_addr,
            dst_addr,
            out_addr,
            g_addr,
            b_addr,
            _f32_bits(epsilon),
            _f32_bits(1.0 / geo.W),
            p,
            hub_mask,
            pack(hub["fwd"]),
            pack(hub["bwd"]),
        ]
        common += [c.y * gx + c.x for c in port_list]
        common += [pack(virt((c.x, c.y))) for c in port_list]
        common += inv_n
        common += [inv_wl]
        common += band_addrs  # own-ready band counters (own_bands > 1)
        if pre_combine:
            common += active_noc  # member NoC coords: row owners and the final-stats fan-out
        if x_sharded:
            # worker identity: byte [y * gx + x] = worker index (shard order), 0xFF = not a worker; 4 per word
            wmap = [0xFF] * (gx * gy)
            for i, (x, y) in enumerate(shard_order):
                wmap[y * gx + x] = i
            wmap += [0xFF] * (-len(wmap) % 4)
            common += [
                wmap[i] | wmap[i + 1] << 8 | wmap[i + 2] << 16 | wmap[i + 3] << 24 for i in range(0, len(wmap), 4)
            ]
        if sp:
            worker_ct[ST_TBL_CT] = len(common)
            common += active_noc  # staging cores (worker order)
        if raw_on:
            worker_ct[RAW_CRT_CT] = len(common)
            common += [int(raw_t.buffer_address()), len(raw_order)] + raw_order
        kernels = [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_reader.cpp"),
                core_ranges=worker_set,
                compile_time_args=worker_ct + acc_ct,
                defines=defines,
                common_runtime_args=common,
                config=_dm(ttnn.DataMovementProcessor.RISCV_1, NOC_READER),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_writer.cpp"),
                core_ranges=worker_set,
                compile_time_args=worker_ct + acc_ct,
                defines=defines,
                common_runtime_args=common,
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, NOC_WRITER),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_compute.cpp"),
                core_ranges=worker_set,
                compile_time_args=worker_ct,
                defines=defines,
                common_runtime_args=common,
                config=cfg,
            ),
        ]

        # ---- port kernels (per-core runtime args) ----
        reader_rt, sender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
        sig = []
        for d in ("fwd", "bwd"):
            peer = ch["next"] if d == "fwd" else ch["prev"]
            up = up_of[d]
            for l in range(L):
                core = ports[(d, l)]
                sends = bulk_sched[coord][d] if peer is not None else []
                up_sends = bulk_sched[up][d] if up is not None else []
                up_counts = [walk.chunks(q, part, l, L) for q, part in up_sends]
                S0, I0, expect_in = _inc_sim(up_counts, inc_every)
                entries = []
                for kk, (q, part) in enumerate(sends):
                    if kk == 0:
                        entries += [q | (part << 16), 0, 0]
                        continue
                    uq, upart = up_sends[kk - 1]
                    assert uq == q and (upart == part or upart == 0), (uq, q, upart, part)
                    off = walk.chunks(q, 1, l, L) if (part == 2 and upart == 0) else 0
                    e = kk - 1
                    entries += [q | (part << 16), I0[e] - S0[e] // inc_every + 1, S0[e] + off]
                r_args = [port_dst_addr, l, L, bulk_addr, expect_in, split.n_active, int(is_ru), len(sends)]
                r_args += entries + active_noc
                hub_sem = (SEM_STATS_READY if d == "fwd" else SEM_STATS_READY_BWD) + 1 if l == 0 else 0
                r_args += [hub_sem, pack(mc0), pack(mc1), gx * gy - 1]
                # hub: stats-table multicast — the fwd hub pushes the own slot, each hub the slots landing in it
                recv_slots = recv[d] if l == 0 else []
                r_args += [
                    stats_addr,
                    slot_bytes,
                    p if (l == 0 and d == "fwd") else 0xFFFF,
                    n_owners if pre_combine else split.n_active,
                    len(recv_slots),
                ] + list(recv_slots)
                r_args += band_need + band_row_end + band_addrs  # own_bands > 1: band gating of the own entry
                # own-entry staging: my lane's staged pages, readiness counter, chunk-home geometry
                if sp:
                    r_args += [1, sp["lanes"][l]["tot"], stage_ctr_addr, split.n_active, sp["nch_max"], l, L, stage_cp]
                else:
                    r_args += [0]
                rin = raw_port_in.get((d, l), [])
                r_args += [int(raw_on), len(rin) // 2] + rin  # gather-raw-first landing streams
                reader_rt[core.x][core.y] = r_args

                is_hub = l == 0
                st_sends = stats_sched[coord][d] if (peer is not None and is_hub) else []
                st_expect = len(stats_sched[up][d]) if (up is not None and is_hub) else 0
                send_ready = peer is not None
                rc = (
                    virt((chips[peer]["ports"][(_fag._OPP[d], l)].x, chips[peer]["ports"][(_fag._OPP[d], l)].y))
                    if send_ready
                    else None
                )
                pc = (
                    virt((chips[peer]["ports"][(d, l)].x, chips[peer]["ports"][(d, l)].y)) if peer is not None else None
                )
                pn = _node(mesh_device, peer) if peer is not None else None
                s_args = [
                    port_dst_addr,
                    l,
                    L,
                    bulk_addr,
                    ready_addr,
                    int(send_ready),
                    pack(rc) if rc is not None else 0,
                    pack(pc) if pc is not None else 0,
                    int(pn.mesh_id) if pn is not None else 0,
                    int(pn.chip_id) if pn is not None else 0,
                    int(peer is not None),
                    int(is_hub),
                    slot_bytes,
                    stats_addr,
                    st_expect,
                    n_owners if pre_combine else split.n_active,  # record writers per hub table
                    len(st_sends),
                ]
                s_args += st_sends
                s_args += [len(sends)]
                for q, part in sends:
                    s_args += [q | (part << 16), walk.chunks(q, part, l, L)]
                sig.append((d, l, core.x, core.y, tuple(r_args[1:3] + r_args[4:]), tuple(s_args[1:3] + s_args[5:])))
                if peer is not None:
                    s_args += list(
                        ttnn.setup_fabric_connection(_node(mesh_device, coord), pn, ch[f"{d}_links"][l], program, core)
                    )
                sender_rt[core.x][core.y] = s_args
        port_cores = _single_cores(port_set)
        kernels += [
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_port_reader.cpp"),
                core_ranges=port_cores,
                compile_time_args=port_ct,
                defines=defines,
                runtime_args=reader_rt,
                config=_dm(ttnn.DataMovementProcessor.RISCV_1, NOC_READER),
            ),
            ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_port_sender.cpp"),
                core_ranges=port_cores,
                compile_time_args=port_ct,
                defines=defines,
                runtime_args=sender_rt,
                config=_dm(ttnn.DataMovementProcessor.RISCV_0, NOC_WRITER),
            ),
        ]
        program.kernels = kernels
        # Program cache key: everything that shapes the program except the per-call tensor addresses.
        h = hashlib.sha256(
            repr(
                (
                    coord,
                    worker_ct,
                    acc_ct,
                    port_ct,
                    common[5:],  # everything after the five per-call tensor addresses
                    sig,
                    [(c.x, c.y) for c in port_list],
                    is_ln,
                    fp32_dest,
                    str(cfg.math_fidelity),
                    bool(cfg.math_approx_mode),
                    str(x_dtype),
                    str(out_dtype),
                    str(g_dtype),
                    crt_max,
                    cw_max,
                    tuple(defines),
                )
            ).encode()
        ).digest()
        program.custom_program_hash = int.from_bytes(h[:8], "little") & ((1 << 63) - 1)
        r, c = coord
        mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(r, c), ttnn.MeshCoordinate(r, c))] = program
    return mesh_desc, [t for t in (staging,) if t is not None]
