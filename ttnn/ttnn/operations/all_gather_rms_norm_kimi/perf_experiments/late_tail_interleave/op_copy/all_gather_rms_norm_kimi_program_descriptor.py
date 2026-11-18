# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm_kimi — host plan + MeshProgramDescriptor (one program per chip).

Regime R1 of op_design.md: L1-landing ring gather + fp32 stats-ahead, normalize-on-landing.

Core roles per chip. The compute cores have the same logical layout on every chip; the ports, receivers and hub sit
on logical row 0 per chip: each port at the row-0 worker nearest (NoC1 hops) to its link's Ethernet core, found on
the host from the fabric connection's router channel (Row0Placement, PORT_PLACEMENT), so a sender addresses the PEER
chip's ready port / receiver / hub by that chip's placement:
  ports      2L  one per (link l, direction d). Reader (RISCV_1, NoC0) reads chunks from compute cores' `landing`
                 (own slice, then the relay half once it landed) into cb_port_chunks; sender (RISCV_0, NoC1) runs the
                 ready fence, then sends every chunk into the SAME landing offset of the SAME logical core on the peer
                 chip (fused arrival increment every INC_EVERY chunks and at each entry end), and on link 0 the fp32
                 stats entries the hub hands it (polled between chunks, so they never wait behind the bulk).
  receivers  2L  one per incoming (link, direction): walk the upstream's static chunk stream; when the arrival counter
                 covers a chunk, increment `landed` on that chunk's compute core (+ relay_ready on the local port for
                 blocks it relays). Then re-arm the arrival counter.
  hub        1   collects the own fp32 partials, hands slot p (and the relayed slots) to the link-0 ports, multicasts
                 the whole stats vector to every compute core once all G slots are in, then re-arms its counters.
  compute    G * nb   core (m, q) owns row block m of slice q: block_h x block_w tiles resident in `landing`.

Knobs (single source of truth; every CB size / loop count / RT arg derives from them):
  BLOCK_W (None = the whole slice width), block_h (derived from the grid left after the ports), chunk_tiles (derived
  from the router payload), PORT_PLACEMENT, PORT_BATCH_CHUNKS, INC_EVERY, OUT_DEPTH_CHUNKS, num_links; the OVERLAP
  knobs of Refinement 2: GAMMA_HUB_MCAST, XG_LATE_STREAM_CHUNKS, FUSE_LATE_CHUNKS, STATS_BULK_WINDOW; Refinement 3:
  COMBINE_COL0, COMPUTE_STORE_NOC, OWN_STORE_GATE.
"""

from __future__ import annotations

import hashlib
import math
import os
import struct
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# Per-stage device zones (kernels/all_gather_rms_norm_kimi_perf.hpp, MaybeDeviceZoneScope) are compiled in only when
# AGRN_PERF_ZONES=1 at op build time: the breakdown runs turn them on; the ccl_perf harness (500 traced calls per
# experiment) leaves them off, because their markers overflow the profiler's DRAM buffer and drop whole programs.
PERF_ZONES = os.environ.get("AGRN_PERF_ZONES", "0") == "1"

# ---- block / transport knobs (host constants, defined once) ----
SLICES_PER_BLOCK = 1  # mechanism cap: a block never straddles slices
BLOCK_W = None  # tile columns of a slice per block; None = the whole slice (Wl_t)
PORT_PLACEMENT = "eth_nearest"  # "eth_nearest" (each port next to its link's Ethernet core) | "centred"  [TRANSPORT]
PORT_BATCH_CHUNKS = 2  # chunks per port batch (cb_port_chunks holds 2 batches); R1: 2 best of 1-4  [TRANSPORT]
PORT_BATCH_CHUNKS_MAX = 8  # packet-header pool cap
# While a port still owes stats slots, it keeps at most this many bulk packets queued in its router channel, so a
# stats packet (the relay hop of the stats round) waits behind that many, not a full channel. 0 = off.  [OVERLAP]
STATS_BULK_WINDOW = 2
INC_EVERY = 8  # fused arrival increment every this many chunks (+ every entry end)  [TRANSPORT]
OUT_DEPTH_CHUNKS = 2  # cb_out depth in chunks
# Look-ahead by stream position: a remote block computes xg = x * gamma BEFORE the stats combine for its chunks that
# are not among the last XG_LATE_STREAM_CHUNKS chunks of their upstream port's static stream; those last ones (they
# land at the very end) do it after the combine so the combine (~2 us of SFPU) never sits on the landing tail. Own
# cores do their whole block ahead. After the stats only out = xg * rstd is left for the chunks done ahead. The
# distance from the stream end is the combine time over the per-chunk landing interval (~6 chunks measured; the
# PERF FOCUS sweep: 10 late -> 19.01 us, 6 -> 18.94, 4 -> 20.74 cliff), independent of the stream length.
# None = no remote chunk ahead (every remote chunk after the combine).  [OVERLAP]
XG_LATE_STREAM_CHUNKS = 8
# A chunk whose xg was not computed ahead runs x * gamma * rstd as ONE chain (x * gamma into DEST, then * rstd_full
# with DEST reused as srcA): one unpack/pack pass on the landing tail instead of an fp32 round trip through cb_xg.
FUSE_LATE_CHUNKS = False
# combine_stats_block's SFPU tail (G-way fp32 sum, * 1/W, + eps, rsqrt) as ONE pass over the column-0 vectors of the
# partial tile (8 vector ops) instead of G + 2 whole-tile passes (32 each); False = the whole-tile chain.  [R3]
COMBINE_COL0 = True
# Normalize mode. "xg": x * gamma ahead of the stats (XG_LATE_STREAM_CHUNKS look-ahead), out = xg * rstd per chunk
# after the combine: two passes for a chunk that lands after the combine. "scale": S = rstd * gamma (one block pass
# right after the combine), out = x * S: one pass per landed chunk, so the landing tail is one multiply.  [R3]
NORMALIZE_MODE = "xg"
# xg mode: a block with no chunk ahead runs one gamma + one normalize pass over the whole block (pushed per chunk)
# instead of two passes per chunk: half the per-pass init / format reconfig on the landing tail.  [R3]
LATE_BLOCK = os.environ.get("LTI_LATE_BLOCK", "0") == "1"
# The trailing blocks of a stream's last entry that hold any of its last TAIL_INTERLEAVE_CHUNKS chunks are sent
# chunk-major (A0 B0 C0 D0 A1 B1 ...): each late block's first chunks land well before its last one, so only its last
# chunk's gamma + normalize + store stays after the stream's last landing. 0 = off.  [OVERLAP, late_tail_interleave]
TAIL_INTERLEAVE_CHUNKS = int(os.environ.get("LTI_INTERLEAVE", "8"))
# Each chunk-major round of that tail ends with a fused arrival increment, so one notification releases the first
# chunks of every late block (INC_EVERY alone would cover them only at the stream end).  [OVERLAP]
TAIL_ROUND_INC = os.environ.get("LTI_ROUND_INC", "1") == "1"
# NoC of the compute cores' writer (the DRAM output store + the partial export); the compute reader takes the other
# one. The port senders and receivers write on NoC1, the port readers pull the landed chunks on NoC0.  [R3]
COMPUTE_STORE_NOC = 1
# An own-slice core starts its DRAM store only once every port that sends its block has issued the reads of the
# block (1) or of its link's whole own entry (2); the ports increment the core's `landed`, which own cores do not
# otherwise use. 0 = store as soon as normalized. Keeps the own-store burst off the bulk.  [R3]
OWN_STORE_GATE = 2
# gamma source: True = the hub reads the Wt sticks once per chip and multicasts them into every compute core's
# stats-scratch tail (op_design.md regime R6); False = every compute core reads its block_w sticks from DRAM. The
# per-core reads (20 cores per slice on the same 12 pages) took ~4.5 us and held up the stats chain.  [OVERLAP]
GAMMA_HUB_MCAST = True
GAMMA_STICK_BYTES = 64  # one 32-element bf16 stick = one DRAM-aligned gamma page (checked per call)
STATS_SCRATCH_CAP_BYTES = 64 * 1024
STATS_ROW_BYTES = 32 * 4  # 32 fp32 row sums per tile row (kernels/..._common.hpp mirrors it)

FWD, BWD = 0, 1
_OPP = {FWD: BWD, BWD: FWD}
_NOCS = (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)

# ---- compute-core CB indices ----
CB_X_BLOCK = 0  # backed on `landing` (zero-copy)
CB_GAMMA = 1
CB_OUT = 2
CB_PARTIALS = 3  # G * block_h fp32 col tiles (UnpackToDestFp32)
CB_XSQ_SUM = 4  # fp32, UnpackToDestFp32
CB_RSTD = 5
CB_PARTIAL = 6
CB_RSTD_FULL = 7  # rstd column-broadcast over the tile (shares cb_partial's allocation)
CB_XG = 8  # x * gamma (fp32), computed before the stats land; out = xg * rstd after
CB_SCALER = 9
# ---- port-core CB ----
CB_PORT_CHUNKS = 0

# ---- program-local semaphore ids (declared on every op core, before the fabric connections take theirs) ----
SEM_OWN_LOADED = 0  # port: own cores of its link loaded x (fan-in, waited for its total)
SEM_STATS_TX = 1  # port: stats slots the hub handed over (one ordered sender: the hub)
SEM_RELAY_READY = 2  # port: relay blocks landed (one ordered sender: the matching receiver)
SEM_LANDED = 3  # compute: landed chunks of this core's block (one ordered sender: one receiver)
SEM_STATS_READY = 4  # compute: hub multicast of the stats landed
SEM_OWN_PARTIALS = 5  # hub: own partial rows exported (fan-in, waited for its total)
SEM_GAMMA_GO = 6  # compute: the hub's gamma multicast landed in the stats-scratch tail (GAMMA_HUB_MCAST)
NUM_LOCAL_SEMS = 7


def _crs(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _f32_bits(v):
    return struct.unpack("<I", struct.pack("<f", float(v)))[0]


def _xy(c):
    """Pack a virtual NoC coordinate (x, y) into one word."""
    return (int(c.x) << 8) | int(c.y)


def _schedule(p, G):
    """(entries toward p+1, entries toward p-1) as (slice q, part): own slice first, in the order they arrive
    downstream. Balanced ring (even G >= 4): the slice opposite each receiver goes half forward (part 1, the first
    half of the link's row blocks), half backward (part 2, the second half)."""
    if G % 2 == 0 and G >= 4:
        h = G // 2
        fwd = [((p - i) % G, 1 if i == h - 1 else 0) for i in range(h)]
        bwd = [((p + i) % G, 2 if i == h - 1 else 0) for i in range(h)]
        return fwd, bwd
    kf = G // 2
    kb = G - 1 - kf
    return [((p - i) % G, 0) for i in range(kf)], [((p + i) % G, 0) for i in range(kb)]


@dataclass
class Geometry:
    mesh_device: object
    G: int
    L: int
    rows: int  # mesh rows (groups for cluster_axis 1)
    cols: int
    Mt: int
    Wl_t: int
    Wt: int
    block_w: int
    block_h: int
    nb: int
    last_h: int
    chunk_tiles: int
    tile_bytes: int
    max_payload: int
    gx: int
    gy: int
    placement: dict  # chip coord -> Row0Placement (ports / receivers / hub differ per chip: they follow its links)
    compute: list  # logical CoreCoord, index i = q * nb + m
    row_base: list  # per logical row y: index of core (0, y) or 0xFFFFFFFF
    key: tuple = field(default=())

    @property
    def cpl(self):  # row blocks per link
        return -(-self.nb // self.L)

    def h(self, m):
        return self.last_h if m == self.nb - 1 else self.block_h

    def nchunks(self, m):
        return self.h(m) * self.block_w // self.chunk_tiles

    def link_rows(self, l):
        return list(range(l * self.cpl, min(self.nb, (l + 1) * self.cpl)))

    def link_of(self, m):
        return m // self.cpl

    def core(self, q, m):
        return self.compute[q * self.nb + m]

    @property
    def row0(self):  # every row-0 core: the union of the chips' port / receiver / hub placements fits in it
        return [ttnn.CoreCoord(x, 0) for x in range(self.gx)]

    @property
    def op_cores(self):
        return self.row0 + self.compute

    @property
    def scratch_bytes(self):  # G stats slots, then (64 B aligned on device) the Wt gamma sticks, then the hub's G
        # stats_ready mask words (one 16 B L1-aligned slot per published slot)
        return self.G * self.slot_bytes + 64 + self.Wt * GAMMA_STICK_BYTES + 16 * self.G

    @property
    def slot_bytes(self):  # one slice's fp32 partial vector: Mt rows x 32 values (kernels: STATS_ROW_BYTES)
        return self.Mt * STATS_ROW_BYTES


@dataclass
class Row0Placement:
    ports: dict  # (l, d) -> logical CoreCoord
    receivers: dict  # (l, d) -> logical CoreCoord
    hub: object


def _link_eth_cores(mesh_device, coord, peers, L):
    """{(l, d): physical NoC (x, y) of the Ethernet core link l toward peers[d] forwards through}, found on the host
    with no device dispatch: the first fabric connection RT arg is the router's Ethernet channel (append_fabric_
    connection_rt_args, a throwaway ProgramDescriptor absorbs its semaphores), and the channel indexes the chip's
    non-harvested Ethernet list (examples/fabric_all_gather `_chip_maps`, the cluster + SoC descriptors)."""
    from ttnn.operations.examples.fabric_all_gather.program_descriptor_with_inline_kernels import _chip_maps

    eth_list = _chip_maps(mesh_device, coord)[0]
    me = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))
    out = {}
    for d in (FWD, BWD):
        pn = mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*peers[d]))
        links = ttnn.get_forwarding_link_indices(me, pn)
        for l in range(L):
            chan = int(
                ttnn.setup_fabric_connection(me, pn, links[l], ttnn.ProgramDescriptor(), ttnn.CoreCoord(0, 0))[0]
            )
            out[(l, d)] = eth_list[chan]
    return out


def _place_row0(mesh_device, coord, peers, L, gx):
    """Each port at the row-0 worker with the fewest NoC1 hops to its link's Ethernet core (examples/fabric_all_gather
    `_worker_nearest_noc1`, report.md: per-link streams then share no NoC1 segment into the Ethernet row); the
    receivers and the hub on the row-0 cores left. Falls back to ports centred on row 0 when the Ethernet cores
    cannot be resolved on the host (e.g. an emulator without a cluster descriptor)."""
    keys = [(l, d) for l in range(L) for d in (FWD, BWD)]
    row0 = {(x, 0) for x in range(gx)}
    ports = {}
    if PORT_PLACEMENT == "eth_nearest":
        try:
            from ttnn.operations.examples.fabric_all_gather.program_descriptor_with_inline_kernels import (
                _worker_nearest_noc1,
            )

            eth = _link_eth_cores(mesh_device, coord, peers, L)
            taken = set()
            for k in keys:
                c = _worker_nearest_noc1(mesh_device, coord, eth[k], taken, row0)
                ports[k] = c
                taken.add((c.x, c.y))
        except Exception:  # host-side lookup unavailable: the centred placement below
            ports = {}
    if not ports:
        x0 = max(0, gx // 2 - L)
        ports = {k: ttnn.CoreCoord(x0 + 2 * k[0] + k[1], 0) for k in keys}
    taken = {c.x for c in ports.values()}
    free0 = [ttnn.CoreCoord(x, 0) for x in range(gx) if x not in taken]
    receivers = {k: free0[i] for i, k in enumerate(keys)}
    return Row0Placement(ports=ports, receivers=receivers, hub=free0[len(keys)])


def build_geometry(mesh_device, padded_shape, cluster_axis, num_links):
    rows, cols = tuple(mesh_device.shape)
    assert cluster_axis == 1
    G = cols
    L = num_links
    M, W_local = padded_shape[-2], padded_shape[-1]
    Mt, Wl_t = M // 32, W_local // 32
    tile_bytes = ttnn.tile_size(ttnn.bfloat16)
    max_payload = int(ttnn.get_tt_fabric_max_payload_size_bytes())
    grid = mesh_device.compute_with_storage_grid_size()
    gx, gy = int(grid.x), int(grid.y)
    key = (id(mesh_device), tuple(padded_shape), G, L, gx, gy, max_payload, PORT_PLACEMENT)
    if key in _GEO_CACHE:
        return _GEO_CACHE[key][1]

    block_w = Wl_t if BLOCK_W is None else BLOCK_W
    if Wl_t % block_w != 0 or block_w != Wl_t:
        # Phase 0: block_w is the whole slice width (a narrower block needs same-slice partial combining).
        raise ValueError(f"all_gather_rms_norm_kimi: block_w {block_w} must equal the slice width {Wl_t} tiles")
    if G > 32:  # the compute cores' stats_ready is a mask of the G slots
        raise ValueError(f"all_gather_rms_norm_kimi: group of {G} devices exceeds the 32-slot stats mask")
    if G * Mt * STATS_ROW_BYTES > STATS_SCRATCH_CAP_BYTES:
        raise ValueError(f"all_gather_rms_norm_kimi: M = {M} too tall for the stats scratch (G*M*4 B > 64 KiB)")

    # ---- placement: ports + receivers + hub on logical row 0 (next to the Ethernet row), compute on the rest ----
    n_aux = 2 * L + 1
    if 2 * L + n_aux > gx:
        raise ValueError(f"all_gather_rms_norm_kimi: grid row of {gx} cores cannot hold {2 * L} ports + {n_aux} aux")
    placement = {
        (r, p): _place_row0(mesh_device, (r, p), ((r, (p + 1) % G), (r, (p - 1) % G)), L, gx)
        for r in range(rows)
        for p in range(G)
    }
    avail = (gy - 1) * gx
    slots = G * SLICES_PER_BLOCK * (Wl_t // block_w)
    if slots > avail:
        raise ValueError(f"all_gather_rms_norm_kimi: grid has {avail} free cores, {slots} needed for one row block")
    per = avail // slots
    block_h = -(-Mt // per)
    nb = -(-Mt // block_h)
    last_h = Mt - (nb - 1) * block_h
    n_compute = slots * nb
    compute = [ttnn.CoreCoord(i % gx, 1 + i // gx) for i in range(n_compute)]
    row_base = [0xFFFFFFFF] + [(y - 1) * gx for y in range(1, gy)]

    cap = max(1, max_payload // tile_bytes)
    # a chunk is a whole number of block rows or a run inside one block row: the compute addresses it as one
    # (rows x cols) window of the block (gamma column offset, rstd row offset)
    chunk_tiles = max(
        d
        for d in range(1, cap + 1)
        if (block_h * block_w) % d == 0 and (last_h * block_w) % d == 0 and (block_w % d == 0 or d % block_w == 0)
    )
    geo = Geometry(
        mesh_device=mesh_device,
        G=G,
        L=L,
        rows=rows,
        cols=cols,
        Mt=Mt,
        Wl_t=Wl_t,
        Wt=G * Wl_t,
        block_w=block_w,
        block_h=block_h,
        nb=nb,
        last_h=last_h,
        chunk_tiles=chunk_tiles,
        tile_bytes=tile_bytes,
        max_payload=max_payload,
        gx=gx,
        gy=gy,
        placement=placement,
        compute=compute,
        row_base=row_base,
        key=key,
    )
    _GEO_CACHE[key] = (mesh_device, geo)
    return geo


_GEO_CACHE = {}
_RES_CACHE = {}


@dataclass
class Resources:
    landing: object
    stats: object
    sems: dict  # name -> global semaphore

    def addr(self, name):
        return int(ttnn.get_global_semaphore_address(self.sems[name]))


def get_resources(mesh_device, geo):
    """Persistent per-(mesh, geometry) scratch: the `landing` shards (compute cores), the stats scratch (every op
    core) and the global counters. Created once, then one synchronize so every chip's counters are 0 before any
    chip's first call can increment a neighbour's."""
    key = geo.key
    if key in _RES_CACHE:
        return _RES_CACHE[key][1]
    block_tiles = geo.block_h * geo.block_w
    comp_crs = compute_crs(geo)
    landing_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(comp_crs, (block_tiles * 32, 32), ttnn.ShardOrientation.ROW_MAJOR),
    )
    landing = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(geo.compute) * block_tiles * 32, 32]),
        ttnn.bfloat16,
        ttnn.TILE_LAYOUT,
        mesh_device,
        landing_mc,
    )
    op_cores = geo.op_cores
    op_crs = op_core_crs(geo)
    words = geo.scratch_bytes // 4
    stats_mc = ttnn.MemoryConfig(
        ttnn.TensorMemoryLayout.HEIGHT_SHARDED,
        ttnn.BufferType.L1,
        ttnn.ShardSpec(op_crs, (1, words), ttnn.ShardOrientation.ROW_MAJOR),
    )
    stats = ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(op_cores), words]), ttnn.uint32, ttnn.ROW_MAJOR_LAYOUT, mesh_device, stats_mc
    )
    sems = {name: ttnn.create_global_semaphore(mesh_device, op_crs, 0) for name in ("ready", "arrival", "sfwd", "sbwd")}
    ttnn.synchronize_device(mesh_device)
    res = Resources(landing=landing, stats=stats, sems=sems)
    _RES_CACHE[key] = (mesh_device, res)
    return res


# ----------------------------------------------------------------------------------------------------------------------
# Streams: what each port sends, what each receiver sees
# ----------------------------------------------------------------------------------------------------------------------


def port_stream(geo, p, l, d):
    """Entries [(q, [m, ...])] port (l, d) of group position p sends, in order. Entry 0 = own slice rows of link l;
    entry k >= 1 relays (part of) the upstream's entry k - 1."""
    sched = _schedule(p, geo.G)[d]
    rows = geo.link_rows(l)
    half = len(rows) // 2
    out = []
    for q, part in sched:
        rs = rows if part == 0 else (rows[:half] if part == 1 else rows[half:])
        out.append((q, list(rs)))
    return out


def own_block_reads(geo, p):
    """How many port streams read each own block of chip p (each such port increments the block's `landed` once);
    the same for every own block (every link's own rows ride entry 0 of both directions)."""
    counts = {
        m: sum(m in ms for l in range(geo.L) for d in (FWD, BWD) for q, ms in port_stream(geo, p, l, d)[:1] if q == p)
        for m in range(geo.nb)
    }
    assert len(set(counts.values())) == 1, counts
    return counts[0]


def tail_blocks(geo, entries):
    """How many trailing blocks of the stream are sent chunk-major: those of the last entry that hold any of the
    stream's last TAIL_INTERLEAVE_CHUNKS chunks."""
    if TAIL_INTERLEAVE_CHUNKS <= 0 or not entries:
        return 0
    total = sum(geo.nchunks(m) for _, ms in entries for m in ms)
    n, t = 0, total
    for m in reversed(entries[-1][1]):
        if t <= total - TAIL_INTERLEAVE_CHUNKS:
            break
        n += 1
        t -= geo.nchunks(m)
    return n


@dataclass
class StreamChunk:
    q: int
    m: int
    c: int  # chunk index within the block
    entry: int
    inc: bool = False  # carries a fused arrival increment
    last_of_block: bool = False  # last chunk of its block in send order


def stream_chunks(geo, entries):
    """Every chunk of a port stream in send order, with its increment flag: the host mirror of the port sender's walk
    (blocks in order, each block's chunks in order; the tail_blocks() trailing blocks chunk-major). The sender
    increments on every INC_EVERY-th chunk, on each entry's last chunk, and (TAIL_ROUND_INC) on the last chunk of each
    chunk-major round of the tail."""
    blocks = [(k, q, m) for k, (q, ms) in enumerate(entries) for m in ms]
    n_tail = tail_blocks(geo, entries)
    head, tail = blocks[: len(blocks) - n_tail], blocks[len(blocks) - n_tail :]
    out = []
    for k, q, m in head:
        n = geo.nchunks(m)
        out += [StreamChunk(q, m, c, k, last_of_block=c == n - 1) for c in range(n)]
        out[-1].inc = m == entries[k][1][-1]  # entry end
    for c in range(max((geo.nchunks(m) for _, _, m in tail), default=0)):
        out += [
            StreamChunk(q, m, c, k, last_of_block=c == geo.nchunks(m) - 1) for k, q, m in tail if c < geo.nchunks(m)
        ]
        out[-1].inc = TAIL_ROUND_INC
    if tail:
        out[-1].inc = True  # the tail ends the last entry
    for i, ch in enumerate(out):
        ch.inc = ch.inc or (i + 1) % INC_EVERY == 0
    return out


def stream_incs(geo, entries):
    """Per chunk of the stream (in send order): the arrival-counter value that covers it; and the total increments."""
    thresholds = []
    incs = pending = 0
    for ch in stream_chunks(geo, entries):
        pending += 1
        if ch.inc:
            incs += 1
            thresholds += [incs] * pending
            pending = 0
    assert pending == 0
    return thresholds, incs


def xg_ahead_table(geo, p):
    """Per compute core (index q * nb + m) of chip p: chunks of its block that compute xg before the stats combine,
    packed 4 per word (8 bits each). Own blocks: all. Remote blocks: their chunks sent before the last
    XG_LATE_STREAM_CHUNKS of their upstream port's stream (each remote block arrives in exactly one upstream stream,
    its chunks in order, so this is a prefix of the block)."""
    G = geo.G
    ahead = [0] * (G * geo.nb)
    for m in range(geo.nb):
        ahead[p * geo.nb + m] = geo.nchunks(m)
    for l in range(geo.L):
        for d in (FWD, BWD):
            up_p = (p - 1) % G if d == FWD else (p + 1) % G
            chunks = stream_chunks(geo, port_stream(geo, up_p, l, d))
            thr = 0 if XG_LATE_STREAM_CHUNKS is None else len(chunks) - XG_LATE_STREAM_CHUNKS
            for ch in chunks[: max(0, thr)]:
                ahead[ch.q * geo.nb + ch.m] += 1
    assert max(ahead) < 256
    words = [0] * (-(-len(ahead) // 4))
    for i, a in enumerate(ahead):
        words[i // 4] |= a << (8 * (i % 4))
    return words


def stats_slots(geo, p, d):
    """Stats slots the link-0 port of direction d sends: own first, then the relayed ones, as the bulk schedule."""
    G = geo.G
    kf = G // 2
    kb = G - 1 - kf
    if d == FWD:
        return [(p - i) % G for i in range(kf)]
    return [(p + i) % G for i in range(kb)]


# ----------------------------------------------------------------------------------------------------------------------
# Program descriptor
# ----------------------------------------------------------------------------------------------------------------------


def _hash(*parts):
    return int.from_bytes(hashlib.sha1("|".join(str(x) for x in parts).encode()).digest()[:8], "little") >> 1


def _mcast_rects(geo):
    """Compute cores as rectangles (logical): full rows merged, then the partial last row."""
    gx = geo.gx
    n = len(geo.compute)
    full_rows = n // gx
    rects = []
    if full_rows:
        rects.append((0, 1, gx - 1, full_rows, full_rows * gx))
    rem = n - full_rows * gx
    if rem:
        rects.append((0, 1 + full_rows, rem - 1, 1 + full_rows, rem))
    return rects


def compute_crs(geo):
    """The compute cores as merged rectangles (one CoreRange per rectangle: cheap dispatch)."""
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, y0, x1, y1, _ in _mcast_rects(geo)]
    )


def op_core_crs(geo):
    """Every op core: logical row 0 (ports / receivers / hub of every chip) + the compute rectangles."""
    row0 = ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(geo.gx - 1, 0))
    return ttnn.CoreRangeSet([row0] + list(compute_crs(geo).ranges()))


def create_mesh_program_descriptor(geo, res, input_tensor, gamma, output, epsilon, cfg):
    mesh = geo.mesh_device
    G, L = geo.G, geo.L
    virt = lambda c: mesh.worker_core_from_logical_core(c)
    node = lambda coord: mesh.get_fabric_node_id(ttnn.MeshCoordinate(*coord))

    tile_bytes = geo.tile_bytes
    chunk = geo.chunk_tiles
    batch = max(1, min(PORT_BATCH_CHUNKS, PORT_BATCH_CHUNKS_MAX))
    landing_addr = int(res.landing.buffer_address())
    stats_addr = int(res.stats.buffer_address())
    ready_addr, arrival_addr = res.addr("ready"), res.addr("arrival")
    sfwd_addr, sbwd_addr = res.addr("sfwd"), res.addr("sbwd")
    in_addr, g_addr, out_addr = (int(t.buffer_address()) for t in (input_tensor, gamma, output))
    gamma_page = int(gamma.buffer_aligned_page_size())
    if gamma_page > GAMMA_STICK_BYTES:
        raise ValueError(
            f"all_gather_rms_norm_kimi: gamma page {gamma_page} B exceeds the {GAMMA_STICK_BYTES} B stick slot"
        )
    W = G * geo.Wl_t * 32

    comp_crs = compute_crs(geo)
    op_crs = op_core_crs(geo)
    # row-0 roles per chip (virtual coords): a sender addresses the PEER chip's ready port / receiver / hub
    port_vs = {c: {k: virt(v) for k, v in pl.ports.items()} for c, pl in geo.placement.items()}
    recv_vs = {c: {k: virt(v) for k, v in pl.receivers.items()} for c, pl in geo.placement.items()}
    hub_vs = {c: virt(pl.hub) for c, pl in geo.placement.items()}

    # geometry CT args shared by the compute-core kernels
    geo_ct = [geo.block_h, geo.block_w, geo.Mt, geo.Wl_t, G, geo.nb, chunk, geo.cpl, geo.gy, L]

    unpack_modes = [ttnn.UnpackToDestMode.Default] * 64
    unpack_modes[CB_PARTIALS] = ttnn.UnpackToDestMode.UnpackToDestFp32
    unpack_modes[CB_XSQ_SUM] = ttnn.UnpackToDestMode.UnpackToDestFp32

    mesh_desc = ttnn.MeshProgramDescriptor()
    for r in range(geo.rows):
        for p in range(G):
            coord = (r, p)
            prev_c, next_c = (r, (p - 1) % G), (r, (p + 1) % G)
            pl = geo.placement[coord]
            port_v, hub_v = port_vs[coord], hub_vs[coord]
            port_crs = _crs([pl.ports[(l, d)] for l in range(L) for d in (FWD, BWD)])
            recv_crs = _crs([pl.receivers[(l, d)] for l in range(L) for d in (FWD, BWD)])
            hub_crs = _crs([pl.hub])
            program = ttnn.ProgramDescriptor()
            program.custom_program_hash = _hash(
                "all_gather_rms_norm_kimi",
                geo.key[1:],
                coord,
                epsilon,
                cfg.math_fidelity,
                cfg.math_approx_mode,
                BLOCK_W,
                sorted((k, (c.x, c.y)) for k, c in pl.ports.items()),
                PORT_BATCH_CHUNKS,
                INC_EVERY,
                STATS_BULK_WINDOW,
                OUT_DEPTH_CHUNKS,
                GAMMA_HUB_MCAST,
                XG_LATE_STREAM_CHUNKS,
                FUSE_LATE_CHUNKS,
                COMBINE_COL0,
                NORMALIZE_MODE,
                LATE_BLOCK,
                TAIL_INTERLEAVE_CHUNKS,
                TAIL_ROUND_INC,
                COMPUTE_STORE_NOC,
                OWN_STORE_GATE,
                gamma_page,
                PERF_ZONES,
            )
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=s, core_ranges=op_crs, initial_value=0) for s in range(NUM_LOCAL_SEMS)
            ]

            # ---------------- CBs ----------------
            def cb(idxs, dtype, pages, cores):
                """One allocation of `pages` tiles; several indices on it = an alias (disjoint lifetimes, see
                l1_ledger.md). Each index keeps its own FIFO pointers, all starting at the allocation base."""
                page = ttnn.tile_size(dtype)
                idxs = idxs if isinstance(idxs, (list, tuple)) else [idxs]
                return ttnn.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=cores,
                    format_descriptors=[
                        ttnn.CBFormatDescriptor(buffer_index=i, data_format=dtype, page_size=page) for i in idxs
                    ],
                )

            bh, bw = geo.block_h, geo.block_w
            # cb_out's capacity must stay a multiple of its push quantum (chunk) when the gamma side is wider
            program.cbs = [
                ttnn.cb_descriptor_from_sharded_tensor(CB_X_BLOCK, res.landing),
                cb(CB_GAMMA, ttnn.bfloat16, bw, comp_crs),  # held until the last chunk's xg
                cb(CB_OUT, ttnn.bfloat16, OUT_DEPTH_CHUNKS * chunk, comp_crs),
                cb(CB_PARTIALS, ttnn.float32, G * bh, comp_crs),
                cb([CB_XSQ_SUM, CB_RSTD], ttnn.float32, bh, comp_crs),  # xsq popped by the reduce before combine
                cb([CB_PARTIAL, CB_RSTD_FULL], ttnn.float32, bh, comp_crs),  # partial exported before the stats land
                cb(CB_XG, ttnn.float32, bh * bw, comp_crs),  # own cores hold the whole block of xg
                cb(CB_SCALER, ttnn.bfloat16, 1, comp_crs),
                cb(CB_PORT_CHUNKS, ttnn.bfloat16, 2 * batch * chunk, port_crs),
            ]

            # ---------------- compute cores ----------------
            port_xy_common = [_xy(port_v[(l, FWD)]) for l in range(L)] + [_xy(port_v[(l, BWD)]) for l in range(L)]
            reader_common = [in_addr, g_addr, stats_addr, p] + geo.row_base + port_xy_common
            reader_ct = geo_ct + [
                CB_X_BLOCK,
                CB_GAMMA,
                CB_PARTIALS,
                CB_SCALER,
                SEM_OWN_LOADED,
                SEM_STATS_READY,
                SEM_LANDED,
                tile_bytes,
                gamma_page,
                SEM_GAMMA_GO,
                int(GAMMA_HUB_MCAST),
            ]
            reader_ct += list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
            reader_ct += list(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
            writer_common = [out_addr, stats_addr, p, _xy(hub_v)] + geo.row_base
            writer_ct = geo_ct + [CB_PARTIAL, CB_OUT, SEM_OWN_PARTIALS, tile_bytes, SEM_LANDED]
            writer_ct += [own_block_reads(geo, p) if OWN_STORE_GATE else 0]
            writer_ct += list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
            compute_common = [p] + geo.row_base + xg_ahead_table(geo, p)
            compute_ct = geo_ct + [_f32_bits(1.0 / W), _f32_bits(epsilon)]
            compute_ct += [CB_X_BLOCK, CB_GAMMA, CB_OUT, CB_PARTIALS, CB_XSQ_SUM, CB_RSTD, CB_PARTIAL, CB_XG]
            compute_ct += [CB_SCALER, CB_RSTD_FULL, int(FUSE_LATE_CHUNKS), int(COMBINE_COL0)]
            compute_ct += [int(NORMALIZE_MODE == "scale"), int(LATE_BLOCK)]

            ccfg = ttnn.ComputeConfigDescriptor()
            ccfg.math_fidelity = cfg.math_fidelity
            ccfg.fp32_dest_acc_en = bool(cfg.fp32_dest_acc_en)
            ccfg.math_approx_mode = bool(cfg.math_approx_mode)
            ccfg.unpack_to_dest_mode = unpack_modes

            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_reader.cpp"),
                    core_ranges=comp_crs,
                    compile_time_args=reader_ct,
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=reader_common,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=_NOCS[1 - COMPUTE_STORE_NOC]
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_writer.cpp"),
                    core_ranges=comp_crs,
                    compile_time_args=writer_ct,
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=writer_common,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=_NOCS[COMPUTE_STORE_NOC]
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_compute.cpp"),
                    core_ranges=comp_crs,
                    compile_time_args=compute_ct,
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=compute_common,
                    config=ccfg,
                ),
            ]

            # ---------------- ports ----------------
            preader_rt, psender_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
            for l in range(L):
                for d in (FWD, BWD):
                    core = pl.ports[(l, d)]
                    peer = next_c if d == FWD else prev_c
                    entries = port_stream(geo, p, l, d)
                    blocks = []  # (q, m, relay, entry_end)
                    for k, (q, ms) in enumerate(entries):
                        for i, m in enumerate(ms):
                            blocks.append((q, m, k > 0, i == len(ms) - 1))
                    n_chunks = sum(geo.nchunks(m) for _, m, _, _ in blocks)
                    slots = stats_slots(geo, p, d) if l == 0 else []
                    active = n_chunks > 0 or len(slots) > 0
                    # reader
                    n_tail = tail_blocks(geo, entries)
                    rargs = [landing_addr, len(geo.link_rows(l)), n_chunks, len(blocks), n_tail]
                    for q, m, relay, end in blocks:
                        rargs += [_xy(virt(geo.core(q, m))), geo.nchunks(m) | (int(relay) << 16) | (int(end) << 17)]
                    preader_rt[core.x][core.y] = rargs
                    # sender
                    sargs = [landing_addr, stats_addr, int(active), n_chunks, len(blocks), n_tail, len(slots)]
                    sargs += [ready_addr, _xy(port_vs[peer][(l, _OPP[d])]), arrival_addr, _xy(recv_vs[peer][(l, d)])]
                    sargs += [sfwd_addr if d == FWD else sbwd_addr, _xy(hub_vs[peer])]
                    pn = node(peer)
                    sargs += [int(pn.mesh_id), int(pn.chip_id)]
                    sargs += slots
                    for q, m, relay, end in blocks:
                        sargs += [_xy(virt(geo.core(q, m))), geo.nchunks(m) | (int(end) << 17)]
                    if active:
                        links = ttnn.get_forwarding_link_indices(node(coord), pn)
                        sargs += list(ttnn.setup_fabric_connection(node(coord), pn, links[l], program, core))
                    psender_rt[core.x][core.y] = sargs
            port_ct = [
                CB_PORT_CHUNKS,
                chunk,
                tile_bytes,
                batch,
                INC_EVERY,
                SEM_OWN_LOADED,
                SEM_RELAY_READY,
                SEM_STATS_TX,
                geo.slot_bytes,
                geo.max_payload,
                STATS_BULK_WINDOW,
                SEM_LANDED,
                OWN_STORE_GATE,
                int(TAIL_ROUND_INC),
            ]
            kernels += [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_port_reader.cpp"),
                    core_ranges=port_crs,
                    compile_time_args=port_ct,
                    runtime_args=preader_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_port_sender.cpp"),
                    core_ranges=port_crs,
                    compile_time_args=port_ct,
                    runtime_args=psender_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                ),
            ]

            # ---------------- receivers ----------------
            recv_rt = ttnn.RuntimeArgs()
            for l in range(L):
                for d in (FWD, BWD):
                    core = pl.receivers[(l, d)]
                    up_p = (p - 1) % G if d == FWD else (p + 1) % G
                    up_entries = port_stream(geo, up_p, l, d)
                    local_entries = port_stream(geo, p, l, d)
                    thresholds, incs = stream_incs(geo, up_entries)
                    events = []
                    for t, ch in enumerate(stream_chunks(geo, up_entries)):
                        relayed = False
                        if ch.entry + 1 < len(local_entries):
                            rq, rms = local_entries[ch.entry + 1]
                            assert rq == ch.q
                            relayed = ch.m in rms
                        flag = 1 if (ch.last_of_block and relayed) else 0
                        events += [thresholds[t], _xy(virt(geo.core(ch.q, ch.m))) | (flag << 16)]
                    recv_rt[core.x][core.y] = [arrival_addr, incs, _xy(port_v[(l, d)]), len(events) // 2] + events
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_receiver.cpp"),
                    core_ranges=recv_crs,
                    compile_time_args=[SEM_LANDED, SEM_RELAY_READY],
                    runtime_args=recv_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
                    ),
                )
            )

            # ---------------- hub ----------------
            kf = len(stats_slots(geo, p, FWD))
            kb = len(stats_slots(geo, p, BWD))
            rects = _mcast_rects(geo)
            hub_args = [
                stats_addr,
                geo.nb,
                p,
                _xy(port_v[(0, FWD)]),
                _xy(port_v[(0, BWD)]),
                kf,
                kb,
                sfwd_addr,
                sbwd_addr,
                g_addr,
                len(rects),
            ]
            for x0, y0, x1, y1, n in rects:
                a, b = virt(ttnn.CoreCoord(x0, y0)), virt(ttnn.CoreCoord(x1, y1))
                hub_args += [min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y), n]
            hub_rt = ttnn.RuntimeArgs()
            hub_rt[pl.hub.x][pl.hub.y] = hub_args
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_hub.cpp"),
                    core_ranges=hub_crs,
                    compile_time_args=[
                        G,
                        geo.slot_bytes,
                        SEM_OWN_PARTIALS,
                        SEM_STATS_TX,
                        SEM_STATS_READY,
                        SEM_GAMMA_GO,
                        int(GAMMA_HUB_MCAST),
                        gamma_page,
                        geo.Wt,
                    ]
                    + list(ttnn.TensorAccessorArgs(gamma).get_compile_time_args()),
                    runtime_args=hub_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                )
            )
            if PERF_ZONES:
                for k in kernels:
                    k.defines = list(k.defines) + [("AGRN_PERF_ZONES", "1")]
            program.kernels = kernels
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = program
    return mesh_desc
