# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""a2a_wkv_b2 — MeshProgramDescriptor (one ProgramDescriptor per chip).

Regime R1 of op_design.md (matmul-first, heads over cores, streamed mt rounds):

  worker cores (one head each at 96 workers):
    reader  (NCRISC, NoC 0 or 1 per READ_NOC_SPLIT): W[h] (Kt x Nt bf8 tiles) once per head, then attn
                           sticks per mt block
    compute               : tilize(mt_block x Kt) -> matmul_block([mt_block x Kt] @ [Kt x Nt]) -> cb_out
    writer  (BRISC, the other NoC): local head -> output L1 tiles; remote head block -> bump the head
                           slot's block counter (GlobalSemaphore)
  fabric_tx cores (one per stream; a stream = the heads sent to one destination over one link):
    reader  (NCRISC, NoC 0): poll the heads' block counters, pull published blocks from cb_out into
                             cb_tx_stage in TX_ORDER (ready: whatever is published, round-robin), one
                             cb_tx_meta page per batch naming its packets; reset each head's counter to 0
                             after its last block
    writer  (BRISC)        : ready handshake, destination address table (cb_tx_addrs, filled once), scatter-
                             write each packet to the destination's output tiles over the fabric, done
                             handshake (flush), wait for the peer's done. Direct connection to its link's
                             router, or (MUX_OVERFLOW_HEADS > 0, regime R6) a client of the link's fabric mux
  fabric mux cores (R6 only): one per (direction, link) shared by more than one stream
  Worker heads are placed by HEAD_PLACEMENT (local heads on the slowest core rows).

Every block knob (mt_block, attn_rm_depth, tx_batch, tx_buffers, tiles_per_packet, heads_per_core, and the R3 knobs
TX_ORDER, HEAD_PLACEMENT, MUX_OVERFLOW_HEADS, MUX_NUM_BUFFERS) is defined
once here and every dependent CB size / loop count derives from it.
"""

from __future__ import annotations

import os
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"  # launch_gap2 copies (variant code behind defines)

# launch_gap2 knobs (defaults = the op as shipped)
KN = dict(
    merge=False,  # one worker reader + one worker writer kernel over every worker core (no NoC-split pair)
    tx_common=False,  # tx reader + tx writer per-core runtime args -> one common table per kernel
    cb_split=False,
    cb_box=False,  # every CB on the bounding rectangle of the program cores (one CB-config multicast range)  # worker-only CBs on the worker cores, tx-only CBs on the tx cores (cb_out on both)
    info=False,  # print the descriptor structure
)

# ---------------------------------------------------------------------------
# Block knobs (single source of truth)
# ---------------------------------------------------------------------------
MT_BLOCK = 1  # token tile rows per block (op_design.md: mt_block)
ATTN_RM_DEPTH = 2  # cb_attn_rm depth in blocks (reader fetches mt+1 while compute tilizes mt)
TX_BATCH = 8  # packets per fabric_tx batch: one pull barrier (reader), one flush + pop (writer)
TX_BUFFERS = 2  # cb_tx_stage depth in batches (pull batch b+1 overlaps send of batch b)
# fabric_tx send order (Refinement 3): "ready" = each batch takes whatever packets are published, round-robin
# over the stream's heads (worker cores finish at very different times, so a strict mt-major stream is gated
# by its slowest head every round); "strict" = mt-major, heads in order (Phase 0 - Refinement 2 order).
TX_ORDER = "ready"
# Which worker core gets which head (Refinement 3): "by_stream" = local heads on the first core rows, the
# heaviest fabric streams on the last; "interleaved" = local heads first, remote destinations round-robin
# over the rest (see _head_order; the two measured equal); "rowmajor" = head h on core h.
HEAD_PLACEMENT = "by_stream"
# R6 (Refinement 3): heads of each neighbour stream moved onto link 1 as a second stream; link 1 is then
# shared by two fabric_tx cores through a fabric mux core (ttnn.experimental.fabric_mux). 0 = one stream per
# link (no mux), the Phase 0 - Refinement 2 topology.
MUX_OVERFLOW_HEADS = 0
MUX_NUM_BUFFERS = 8  # fabric mux buffers per client channel
# Worker read NoC split (Refinement 2): which worker cores swap their reader onto NoC 1 (and their writer
# onto NoC 0), so the DRAM read stream's responses use both NoCs' links. Each kernel still uses one NoC
# (dedicated mode), so no cross-NoC counter state is involved. "none" = every reader on NoC 0 (Phase 0);
# "halves" (shipped) = right half of the grid reads on NoC 1; also "halves_left", "rows", "cols", "checker",
# "all" — measured in changelog.md Refinement 2 (checker defeats dispatch multicast; all-NoC-1 is slow).
READ_NOC_SPLIT = "halves"
MAX_MATMUL_CORES = 110  # matmul di/dt cap
SCATTER_MAX_CHUNKS = 4  # NOC_SCATTER_WRITE_MAX_CHUNKS
DEST_TILES_HALF_SYNC_16B = 8  # DEST capacity (half sync, 16-bit)
MAX_COMMON_RUNTIME_ARGS = 341  # per-kernel common runtime-arg cap

# CB indices (any distinct values)
CB_WEIGHT = 0
CB_ATTN_RM = 1
CB_ATTN_TILED = 2
CB_OUT = 3
CB_TX_STAGE = 4
CB_TX_ADDRS = 5  # fabric_tx writer: per-packet destination NoC address table (scratch, not a FIFO)
CB_TX_META = 6  # fabric_tx reader -> writer: per batch [k, packet code x tx_batch]

TILE = 32

# TEMP (Perf 1 ablation): comma list of NO_READ, NO_COMPUTE, NO_PULL, NO_SEND -> A2A_ABL_<x> defines on every kernel.
_ABL_DEFINES = [(f"A2A_ABL_{x.strip()}", "1") for x in os.environ.get("A2A_ABLATE", "").split(",") if x.strip()]


def _div_up(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Cross-chip semaphores: created once per mesh device on the first (eager) call
# ---------------------------------------------------------------------------
_SEM_CACHE = []  # [(mesh_device, sem_ready, sem_done)]
_COUNTER_CACHE = []  # [(mesh_device, [block counter per head slot])]


def get_cross_chip_semaphores(mesh_device):
    for dev, ready, done in _SEM_CACHE:
        if dev is mesh_device:
            return ready, done
    # make_ccl_semaphore runs a host Synchronize — never called during a trace capture because the
    # first call of a mesh device is always eager.
    ready = ttnn._ttnn.ccl_host.make_ccl_semaphore(mesh_device, 0)
    done = ttnn._ttnn.ccl_host.make_ccl_semaphore(mesh_device, 0)
    _SEM_CACHE.append((mesh_device, ready, done))
    return ready, done


def get_block_counter_semaphores(mesh_device, num_slots):
    """One GlobalSemaphore per worker head slot: the slot's published-block count (zero between calls).

    Persistent addresses (unlike a program semaphore, which a worker re-initialises only at its own
    launch, possibly after a fabric_tx core has polled it). Created on the first (eager) call."""
    for dev, sems in _COUNTER_CACHE:
        if dev is mesh_device and len(sems) >= num_slots:
            return sems[:num_slots]
    sems = [ttnn._ttnn.ccl_host.make_ccl_semaphore(mesh_device, 0) for _ in range(num_slots)]
    _COUNTER_CACHE.append((mesh_device, sems))
    return sems


# ---------------------------------------------------------------------------
# Ring route plan (derived from the mesh; never hard-coded)
# ---------------------------------------------------------------------------


def _ring_streams(n, hpc, overflow=0):
    """Streams leaving one chip of an n-ring. Each stream = (direction, hops, link, dst_offset, heads).

    direction +1 = towards ring position r+1, -1 = towards r-1. The antipode (2k == n) is split in
    two halves, one per direction. Within a direction, streams sorted by hop count get links 0, 1, ...
    `overflow` (MUX_OVERFLOW_HEADS) moves heads of the neighbour stream onto link 1 (see below).
    """
    raw = []
    for k in range(1, n):
        if 2 * k < n:
            raw.append((+1, k, k, list(range(hpc))))
        elif 2 * k > n:
            raw.append((-1, n - k, k, list(range(hpc))))
        else:
            half = hpc // 2
            raw.append((+1, k, k, list(range(half))))
            raw.append((-1, k, k, list(range(half, hpc))))
    streams = []
    for direction in (+1, -1):
        mine = sorted([s for s in raw if s[0] == direction], key=lambda s: s[1])
        links = len(mine)
        for link, (d, hops, offset, heads) in enumerate(mine):
            if link == 0 and hops == 1 and links > 1 and 0 < overflow < len(heads):
                # R6 (Refinement 3): the neighbour stream's last `overflow` heads go out on link 1 as a
                # second stream, sharing that link (with the antipode stream) through a fabric mux.
                streams.append({"dir": d, "hops": hops, "link": 1, "offset": offset, "heads": heads[-overflow:]})
                heads = heads[:-overflow]
            streams.append({"dir": d, "hops": hops, "link": link, "offset": offset, "heads": heads})
    # slot = position among the streams sharing one (direction, link), by hop count. The plan is the same
    # on every chip, so a stream's peer (the destination's stream targeting this chip) is the stream of the
    # opposite direction with the same link and slot.
    for s in streams:
        group = sorted((t for t in streams if t["dir"] == s["dir"] and t["link"] == s["link"]), key=lambda t: t["hops"])
        s["slot"] = next(i for i, t in enumerate(group) if t is s)
    return sorted(streams, key=lambda s: (-s["dir"], s["link"], s["slot"]))


def _reads_on_noc1(core, grid):
    if READ_NOC_SPLIT == "none":
        return False
    if READ_NOC_SPLIT == "all":
        return True
    if READ_NOC_SPLIT == "checker":
        return (core.x + core.y) % 2 == 1
    if READ_NOC_SPLIT == "rows":
        return core.y % 2 == 1
    if READ_NOC_SPLIT == "cols":
        return core.x % 2 == 1
    if READ_NOC_SPLIT == "halves":
        return core.x >= grid.x // 2
    if READ_NOC_SPLIT == "halves_left":
        return core.x < grid.x // 2
    raise ValueError(f"unknown READ_NOC_SPLIT {READ_NOC_SPLIT!r}")


# Worker reader per-bank NoC choice (Perf 1). Per worker core (indexed [logical y][logical x]) a 16-bit mask:
# bit (bank + 8 * ((page / 8) & 1)) set = that DRAM page is read on NoC 1, else NoC 0 (the reader issues on
# both NoCs from NCRISC, kernels/a2a_wkv_b2_reader.cpp). The R2 "halves" split leaves the NoC 1 northbound
# links of the two DRAM columns saturated (link-load model, perf_experiments/reader_issue/ri_linkmodel.py);
# these masks (searched with that model, ri_masksearch*.py, seed "s1") spread every core's DRAM responses so
# no link binds. The read itself is at a DRAM ceiling either way (reader alone ~52.4 us); what moves is the
# NoC 1 bandwidth left for the fabric_tx pulls / sends: 82.4 -> 78.6 us per call (seeds u4 / u6: 78.2-79.4).
# The table is specific to this placement: a 12-wide worker grid of 8 rows on the Blackhole p150b layout
# (harvested physical columns, DRAM endpoints) the search modelled. Any mask is CORRECT on any grid (it only
# picks a NoC per page); off that layout the table has no meaning and _noc1_bank_mask falls back to the
# "halves" equivalent (the core's default reader NoC for every page).
NOC1_BANK_MASKS = (
    (0x844D, 0x5155, 0x81F0, 0x864D, 0xC09C, 0x4C88, 0x7474, 0x5D40, 0x45CD, 0xC3AC, 0x45B6, 0x0905),
    (0x8C0C, 0x5ACC, 0x94B2, 0x9455, 0x5C5B, 0xF576, 0xC85E, 0x4C4F, 0x45CE, 0x068D, 0x8D45, 0x80C4),
    (0x1D0A, 0xD49C, 0x3C1D, 0x1434, 0x18C4, 0x1414, 0xE288, 0xC9CE, 0x68D6, 0xED21, 0x4D60, 0x2C4C),
    (0x3130, 0xB01A, 0x3913, 0x1818, 0x1801, 0x918A, 0x9190, 0xA129, 0x0881, 0x2101, 0x8001, 0x8000),
    (0x0909, 0x1F91, 0x2118, 0x908A, 0x9302, 0x1119, 0xA998, 0x8909, 0x2E08, 0x09A1, 0x3981, 0x8929),
    (0x1281, 0x98B8, 0x1000, 0x309A, 0x8B00, 0x008A, 0xB131, 0x1B8A, 0x80C9, 0x0903, 0x4381, 0xC9A3),
    (0x0000, 0x0122, 0x1252, 0x1030, 0x1003, 0x3131, 0x3012, 0x0141, 0x2302, 0x2230, 0x8000, 0xB121),
    (0x1011, 0x2201, 0x7210, 0x1050, 0x0242, 0x3312, 0x1113, 0x0101, 0xA363, 0x0120, 0x0301, 0x0009),
)


def _noc1_bank_mask(core, grid, worker_cores):
    if grid.x == len(NOC1_BANK_MASKS[0]) and all(c.y < len(NOC1_BANK_MASKS) for c in worker_cores):
        return NOC1_BANK_MASKS[core.y][core.x]
    return 0xFFFF if _reads_on_noc1(core, grid) else 0x0000


def _core_ranges(cores):
    """Pack cores into as few rectangles as possible (x-runs per row, merged over consecutive rows), so
    dispatch can multicast binaries / launch messages to a core set instead of unicasting per core."""
    rows = {}
    for c in cores:
        rows.setdefault(c.y, set()).add(c.x)
    runs = []  # (x0, x1, y)
    for y in sorted(rows):
        xs = sorted(rows[y])
        start = prev = xs[0]
        for x in xs[1:] + [None]:
            if x is not None and x == prev + 1:
                prev = x
                continue
            runs.append((start, prev, y))
            if x is not None:
                start = prev = x
    rects = []  # [x0, x1, y0, y1]
    for x0, x1, y in runs:
        for r in rects:
            if r[0] == x0 and r[1] == x1 and r[3] == y - 1:
                r[3] = y
                break
        else:
            rects.append([x0, x1, y, y])
    return ttnn.CoreRangeSet(
        [ttnn.CoreRange(ttnn.CoreCoord(x0, y0), ttnn.CoreCoord(x1, y1)) for x0, x1, y0, y1 in rects]
    )


def _head_order(r, n, hpc, streams):
    """Global head order in which heads fill the worker cores (row-major), for chip r (HEAD_PLACEMENT).

    "by_stream" (Refinement 3): destination blocks of hpc heads in the order local, then remote
    destinations by ascending per-stream head count (antipode halves before whole-chunk neighbours), so
    the heaviest fabric streams sit on the last core rows and the local heads (no fabric send) on the
    first. Measured on this grid (READ_NOC_SPLIT = "halves"): worker DRAM streams finish from ~24 us
    (bottom rows) to ~60 us (top rows), so the heads whose late blocks must still cross a link are the
    ones that finish first. "rowmajor": head h on core h (Phase 0 - Refinement 2).
    """
    if HEAD_PLACEMENT == "rowmajor":
        return list(range(n * hpc))
    assert HEAD_PLACEMENT in ("by_stream", "interleaved"), HEAD_PLACEMENT
    stream_heads = {}  # destination offset k -> max heads of one of its streams
    for s in streams:
        stream_heads[s["offset"]] = max(stream_heads.get(s["offset"], 0), len(s["heads"]))
    offsets = sorted(range(1, n), key=lambda k: (stream_heads[k], k))
    local = [r * hpc + h for h in range(hpc)]
    if HEAD_PLACEMENT == "by_stream":
        return local + [((r + k) % n) * hpc + h for k in offsets for h in range(hpc)]
    # "interleaved": remote destinations round-robin over the remaining cores, so every stream gets
    # heads on fast and slow cores alike (each starts sending as early as the fastest cores publish).
    return local + [((r + k) % n) * hpc + h for h in range(hpc) for k in offsets]


def _place_heads(core_slots, order):
    """Give each core its head count from `order`; a core's heads must be consecutive global heads (the
    reader takes head_start + count). Falls back to row-major if `order` breaks that."""
    assign, i = [], 0
    for core, cnt in core_slots:
        chunk = order[i : i + cnt]
        if chunk != list(range(chunk[0], chunk[0] + cnt)):
            return _place_heads(core_slots, sorted(order))
        assign.append((core, chunk[0], cnt))
        i += cnt
    owner = {}  # global head -> (worker core, head index within that core)
    for core, start, cnt in assign:
        for j in range(cnt):
            owner[start + j] = (core, j)
    return assign, owner


def _merge_semaphores(pd):
    """One SemaphoreDescriptor per (id, initial value, core type), its cores packed into rectangles (Perf 1,
    launch gap). The fabric connection args add one single-core semaphore descriptor per tx core; dispatch
    writes one transfer per descriptor range (~0.25-0.45 us each at a trace-replay boundary), and a
    CoreRangeSet of single-core ranges is not merged by itself, so the ranges are re-packed here
    (measured: 8 single-core semaphores cost ~1.9 us of launch gap)."""
    by_key = {}
    for sd in pd.semaphores:
        by_key.setdefault((sd.id, sd.initial_value, sd.core_type), []).extend(
            ttnn.corerange_to_cores(sd.core_ranges, None, True)
        )
    pd.semaphores = [
        ttnn.SemaphoreDescriptor(id=i, core_type=ct, core_ranges=_core_ranges(cores), initial_value=v)
        for (i, v, ct), cores in by_key.items()
    ]


def _tx_to_common(pd, first_tx_kernel, tx_cores, grid, counters):
    """launch_gap2: every tx kernel's per-core runtime args -> one common table (kernel indexes its block by its
    logical core, see the A2A_TX_COMMON block in the tx kernels). The tx reader's per-head args are packed to one
    word (x | y << 8 | head idx << 16) with the counter addresses shared, so the table fits the common-arg cap."""
    idx = [c.y * grid.x + c.x for c in tx_cores]
    idx0, nslots = min(idx), max(idx) - min(idx) + 1
    kernels = list(pd.kernels)
    for k in range(first_tx_kernel, len(kernels)):
        kd = kernels[k]
        if "a2a_wkv_b2_tx_" not in str(kd.kernel_source):
            continue  # fabric mux kernels keep their own args
        is_reader = "tx_reader" in str(kd.kernel_source)
        blocks = {}
        for c in tx_cores:
            try:
                a = list(kd.runtime_args[c.x][c.y])
            except Exception:
                continue
            if not a and not kd.core_ranges.contains(c):
                continue
            if is_reader:
                nh = a[0]
                a = [nh] + [a[1 + 4 * s] | (a[2 + 4 * s] << 8) | (a[3 + 4 * s] << 16) for s in range(nh)]
            blocks[c.y * grid.x + c.x] = a
        head = 3 + nslots
        table, offs = [], [0] * nslots
        for i in sorted(blocks):
            offs[i - idx0] = head + len(table)
            table += blocks[i]
        counters_base = head + len(table)
        common = (
            [grid.x, idx0, counters_base if is_reader else 0] + offs + table + (list(counters) if is_reader else [])
        )
        assert len(common) <= MAX_COMMON_RUNTIME_ARGS, len(common)
        kernels[k] = ttnn.KernelDescriptor(
            kernel_source=kd.kernel_source,
            core_ranges=kd.core_ranges,
            compile_time_args=list(kd.compile_time_args),
            defines=list(kd.defines) + [("A2A_TX_COMMON", "1")],
            common_runtime_args=common,
            config=kd.config,
        )
    pd.kernels = kernels


def _info(pd):
    print("\nLG2_INFO kernels:")
    for kd in pd.kernels:
        n_rt = 0
        try:
            for c in ttnn.corerange_to_cores(kd.core_ranges, None, True):
                if len(kd.runtime_args[c.x][c.y]):
                    n_rt += 1
        except Exception:
            n_rt = -1
        print(
            f"  {Path(str(kd.kernel_source)).name} ranges={len(kd.core_ranges.ranges())} cores={kd.core_ranges.num_cores()}"
            f" crta={len(kd.common_runtime_args)} rt_cores={n_rt}"
        )
    print("LG2_INFO cbs: " + "; ".join(f"ranges={len(c.core_ranges.ranges())}" for c in pd.cbs))
    print("LG2_INFO sems: " + "; ".join(f"id={s.id} ranges={len(s.core_ranges.ranges())}" for s in pd.semaphores))


def _coord_with(coord_tuple, axis, value):
    c = list(coord_tuple)
    c[axis] = value
    return ttnn.MeshCoordinate(*c)


# ---------------------------------------------------------------------------
# Mesh program descriptor
# ---------------------------------------------------------------------------


def create_mesh_program_descriptor(
    input_tensor,
    weight,
    output_tensor,
    *,
    cluster_axis,
    num_links,
    compute_kernel_config,
    sem_ready,
    sem_done,
):
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    n = mesh_shape[cluster_axis]

    _, H, T, L = tuple(input_tensor.shape)
    V = tuple(weight.shape)[-1]
    hpc = H // n  # heads owned by each chip after the exchange
    Mt, Kt, Nt = _div_up(T, TILE), _div_up(L, TILE), _div_up(V, TILE)

    mt_block = MT_BLOCK
    assert 1 <= mt_block <= Mt
    num_mt_blocks = _div_up(Mt, mt_block)
    # Output subblock 1 x Nt must fit DEST (the helper's subblock legality).
    assert Nt <= DEST_TILES_HALF_SYNC_16B, "Nt must fit DEST as one output subblock"

    attn_tile_bytes = ttnn.tile_size(ttnn.bfloat16)
    w_tile_bytes = ttnn.tile_size(ttnn.bfloat8_b)
    out_tile_bytes = ttnn.tile_size(ttnn.bfloat8_b)
    stick_bytes = L * input_tensor.element_size()
    attn_page_bytes = input_tensor.buffer_aligned_page_size()

    max_payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    tiles_per_packet = min(Nt, SCATTER_MAX_CHUNKS, max_payload // out_tile_bytes)
    assert tiles_per_packet >= 1, "fabric payload smaller than one tile"
    l1_alignment = ttnn.get_l1_alignment()

    # ---------------- streams & cores ----------------
    streams = _ring_streams(n, hpc, MUX_OVERFLOW_HEADS)
    num_streams = len(streams)
    # (direction, link) groups with more than one stream share their link through one fabric mux core.
    link_groups = sorted({(s["dir"], s["link"]) for s in streams}, key=lambda g: (-g[0], g[1]))
    mux_groups = [g for g in link_groups if sum(1 for s in streams if (s["dir"], s["link"]) == g) > 1]
    # fabric_tx writer's destination address table: one entry (8 B NoC address) per tile slot of every
    # packet of the largest stream, in send order (same packet count derivation as the tx kernels).
    addr_entry_bytes = 8
    packets_per_head = sum(
        _div_up(min(mt_block, Mt - mb * mt_block) * Nt, tiles_per_packet) for mb in range(num_mt_blocks)
    )
    max_stream_packets = max(len(s["heads"]) for s in streams) * packets_per_head
    meta_page_bytes = _div_up((1 + TX_BATCH) * 4, l1_alignment) * l1_alignment
    assert TX_ORDER in ("ready", "strict"), TX_ORDER
    per_dir = max(len({s["link"] for s in streams if s["dir"] == d}) for d in (+1, -1))
    assert per_dir == num_links, "validate() guarantees num_links == links per direction"

    grid = mesh_device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    all_logical = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(total_cores)]
    tx_cores = all_logical[total_cores - num_streams :]
    mux_cores = all_logical[total_cores - num_streams - len(mux_groups) : total_cores - num_streams]
    num_workers = min(H, MAX_MATMUL_CORES, total_cores - num_streams - len(mux_groups))
    worker_crs = ttnn.num_cores_to_corerangeset(num_workers, grid, True)
    (_, _, g1, g2, heads_g1, heads_g2) = ttnn.split_work_to_cores(worker_crs, H, True)
    core_slots = []  # [(core, num_heads)] in row-major core order
    for group, cnt in ((g1, heads_g1), (g2, heads_g2)):
        if cnt == 0:
            continue
        for core in sorted(ttnn.corerange_to_cores(group, None, True), key=lambda c: (c.y, c.x)):
            core_slots.append((core, cnt))
    assert sum(cnt for _, cnt in core_slots) == H
    heads_per_core = max(cnt for _, cnt in core_slots)
    worker_cores = [c for c, _ in core_slots]
    worker_set = _core_ranges(worker_cores)
    tx_set = _core_ranges(tx_cores)
    program_set = worker_set.merge(tx_set)

    tx_core_of = {(s["dir"], s["link"], s["slot"]): tx_cores[i] for i, s in enumerate(streams)}
    mux_core_of = {g: mux_cores[i] for i, g in enumerate(mux_groups)}
    direct_tx = [
        tx_core_of[(s["dir"], s["link"], s["slot"])] for s in streams if (s["dir"], s["link"]) not in mux_core_of
    ]
    muxed_tx = [tx_core_of[(s["dir"], s["link"], s["slot"])] for s in streams if (s["dir"], s["link"]) in mux_core_of]
    if mux_groups:
        fabric_mux = ttnn.experimental.fabric_mux
        mux_channel = fabric_mux.ChannelType.FULL_SIZE
        mux_clients = max(sum(1 for s in streams if (s["dir"], s["link"]) == g) for g in mux_groups)
        # Fixed-address map at the L1 base of the mux cores, which carry no CB (CBs span workers + tx only).
        mux_config = fabric_mux.Config(
            num_full_size_channels=mux_clients,
            num_header_only_channels=0,
            num_buffers_per_full_size_channel=MUX_NUM_BUFFERS,
            num_buffers_per_header_only_channel=0,
            full_size_channel_buffer_size_bytes=fabric_mux.channel_buffer_size_bytes(mesh_device),
            base_l1_address=ttnn.get_allocator_base_address(mesh_device, ttnn.BufferType.L1),
        )
        mux_client_ct = fabric_mux.client_compile_time_args(
            num_clients=mux_clients, channel_type=mux_channel, config=mux_config
        )

    counter_addrs = [
        ttnn.get_global_semaphore_address(c) for c in get_block_counter_semaphores(mesh_device, heads_per_core)
    ]
    sem_ready_addr = ttnn.get_global_semaphore_address(sem_ready)
    sem_done_addr = ttnn.get_global_semaphore_address(sem_done)
    fabric_defines = list(ttnn.get_fabric_kernel_defines("Linear"))

    out_ct_accessor = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    worker_only = (CB_WEIGHT, CB_ATTN_RM, CB_ATTN_TILED)
    tx_only = (CB_TX_STAGE, CB_TX_ADDRS, CB_TX_META)

    def cb(index, dtype, page_size, num_pages):
        crs = program_set
        if KN["cb_box"]:
            pc = ttnn.corerange_to_cores(program_set, None, True)
            crs = ttnn.CoreRangeSet(
                [
                    ttnn.CoreRange(
                        ttnn.CoreCoord(min(c.x for c in pc), min(c.y for c in pc)),
                        ttnn.CoreCoord(max(c.x for c in pc), max(c.y for c in pc)),
                    )
                ]
            )
        elif KN["cb_split"]:
            crs = worker_set if index in worker_only else tx_set if index in tx_only else program_set
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=crs,  # uniform CB addresses on every program core (pull_block)
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    mesh_pd = ttnn.MeshProgramDescriptor()

    for flat in range(mesh_device.get_num_devices()):
        idx = []
        rem = flat
        for dim in reversed(mesh_shape):
            idx.append(rem % dim)
            rem //= dim
        idx = tuple(reversed(idx))
        coord = ttnn.MeshCoordinate(*idx)
        r = idx[cluster_axis]
        my_node = mesh_device.get_fabric_node_id(coord)
        worker_assign, head_owner = _place_heads(core_slots, _head_order(r, n, hpc, streams))

        # ---------------- CBs ----------------
        cbs = [
            cb(CB_TX_ADDRS, ttnn.uint32, addr_entry_bytes * tiles_per_packet, max_stream_packets),
            cb(CB_TX_META, ttnn.uint32, meta_page_bytes, TX_BUFFERS),
            cb(CB_WEIGHT, ttnn.bfloat8_b, w_tile_bytes, Kt * Nt),
            cb(CB_ATTN_RM, ttnn.bfloat16, attn_tile_bytes, ATTN_RM_DEPTH * mt_block * Kt),
            cb(CB_ATTN_TILED, ttnn.bfloat16, attn_tile_bytes, mt_block * Kt),
            cb(CB_OUT, ttnn.bfloat8_b, out_tile_bytes, heads_per_core * Mt * Nt),
            cb(CB_TX_STAGE, ttnn.bfloat8_b, out_tile_bytes, TX_BUFFERS * TX_BATCH * tiles_per_packet),
        ]
        semaphores = []  # block counters are GlobalSemaphores (counter_addrs)

        # ---------------- worker kernels ----------------
        w_reader_ct = [CB_WEIGHT, CB_ATTN_RM, Kt, Nt, Mt, mt_block, T, stick_bytes, attn_page_bytes, w_tile_bytes, TILE]
        w_reader_ct += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
        w_reader_ct += ttnn.TensorAccessorArgs(weight).get_compile_time_args()
        w_writer_ct = [CB_OUT, Mt, Nt, mt_block, hpc, n, out_tile_bytes] + out_ct_accessor
        compute_ct = [CB_ATTN_RM, CB_ATTN_TILED, CB_WEIGHT, CB_OUT, Kt, Nt, Mt, mt_block]

        # Worker args (Perf 1, launch gap): no per-core runtime args. Every per-core value sits in ONE common
        # runtime-arg table per kernel, one word per logical core index y * grid.x + x (the kernel indexes it
        # with get_relative_logical_x/y), so dispatch multicasts one copy per kernel group instead of
        # writing every worker core (trace-replay launch gap 11.5 -> 9.5 us with the table alone).
        core_table = [0] * (max(c.y * grid.x + c.x for c in worker_cores) + 1)
        for core, start, cnt in worker_assign:
            assert start < 256 and cnt < 256
            core_table[core.y * grid.x + core.x] = start | (cnt << 8)
        reader_table = list(core_table)  # + the core's per-bank NoC 1 mask (Perf 1, worker reader)
        for core in worker_cores:
            reader_table[core.y * grid.x + core.x] |= _noc1_bank_mask(core, grid, worker_cores) << 16
        w_reader_common = [input_tensor.buffer_address(), weight.buffer_address(), grid.x] + reader_table
        w_writer_common = [output_tensor.buffer_address(), r, grid.x, 4 + heads_per_core]
        w_writer_common += counter_addrs[:heads_per_core] + core_table
        compute_common = [grid.x] + core_table
        assert max(len(w_reader_common), len(w_writer_common)) <= MAX_COMMON_RUNTIME_ARGS

        worker_kernels = []
        if KN["merge"]:
            # launch_gap2: one kernel per RISC over every worker core. The reader picks the NoC per page from its
            # mask (DM_DYNAMIC_NOC), so its kernel default NoC is unused; the writer takes the NoC its group used
            # to default to from its table word (bit 16).
            w_writer_table = list(core_table)
            for core in worker_cores:
                w_writer_table[core.y * grid.x + core.x] |= (0 if _reads_on_noc1(core, grid) else 1) << 16
            w_writer_common_m = w_writer_common[: len(w_writer_common) - len(core_table)] + w_writer_table
            dyn = ttnn.NOC_MODE.DM_DYNAMIC_NOC
            worker_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_reader.cpp"),
                    core_ranges=worker_set,
                    compile_time_args=w_reader_ct,
                    defines=_ABL_DEFINES,
                    common_runtime_args=w_reader_common,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0, noc_mode=dyn
                    ),
                )
            )
            worker_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_writer.cpp"),
                    core_ranges=worker_set,
                    compile_time_args=w_writer_ct,
                    defines=[("A2A_W_TABLE_NOC", "1")],
                    common_runtime_args=w_writer_common_m,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1, noc_mode=dyn
                    ),
                )
            )
        for on_noc1 in (False, True) if not KN["merge"] else ():
            cores = [c for c in worker_cores if _reads_on_noc1(c, grid) == on_noc1]
            if not cores:
                continue
            crs = _core_ranges(cores)
            # Dynamic NoC mode (Perf 1): the reader issues DRAM reads on BOTH NoCs (per-bank NoC mask), so the
            # two RISCs share a NoC; in dedicated mode the writer's end-of-kernel idle check (and any barrier
            # on its NoC) would count the reader's transactions as its own. Each kernel's default NoC is
            # unchanged: reader NoC 0 / writer NoC 1, swapped on the on_noc1 set.
            read_noc, write_noc = (ttnn.NOC.NOC_1, ttnn.NOC.NOC_0) if on_noc1 else (ttnn.NOC.NOC_0, ttnn.NOC.NOC_1)
            reader_cfg = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_1, noc=read_noc, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC
            )
            writer_cfg = ttnn.DataMovementConfigDescriptor(
                processor=ttnn.DataMovementProcessor.RISCV_0, noc=write_noc, noc_mode=ttnn.NOC_MODE.DM_DYNAMIC_NOC
            )
            worker_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_reader.cpp"),
                    core_ranges=crs,
                    compile_time_args=w_reader_ct,
                    defines=_ABL_DEFINES,
                    common_runtime_args=w_reader_common,
                    config=reader_cfg,
                )
            )
            worker_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_writer.cpp"),
                    core_ranges=crs,
                    compile_time_args=w_writer_ct,
                    common_runtime_args=w_writer_common,
                    config=writer_cfg,
                )
            )
        worker_compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_compute.cpp"),
            core_ranges=worker_set,
            compile_time_args=compute_ct,
            defines=_ABL_DEFINES,
            common_runtime_args=compute_common,
            config=compute_kernel_config,
        )

        # ---------------- fabric_tx kernels ----------------
        max_stream_heads = max(len(s["heads"]) for s in streams)
        tx_reader_ct = [CB_TX_STAGE, CB_OUT, Mt, Nt, mt_block, tiles_per_packet, out_tile_bytes]
        tx_reader_ct += [TX_BATCH, max_stream_heads, CB_TX_META, l1_alignment, int(TX_ORDER == "ready")]
        tx_writer_ct = [CB_TX_STAGE, Mt, Nt, mt_block, tiles_per_packet, out_tile_bytes, n, l1_alignment, TX_BATCH]
        tx_writer_ct += [CB_TX_ADDRS, CB_TX_META]
        tx_writer_ct += out_ct_accessor
        tx_reader_rt = ttnn.RuntimeArgs()
        tx_writer_rt = ttnn.RuntimeArgs()
        tx_plan = []  # (stream, tx core, neighbour node, writer op args)
        for s in streams:
            tx_core = tx_core_of[(s["dir"], s["link"], s["slot"])]
            d = (r + s["offset"]) % n  # destination ring position
            dst_coord = _coord_with(idx, cluster_axis, d)
            dst_node = mesh_device.get_fabric_node_id(dst_coord)
            nbr_node = mesh_device.get_fabric_node_id(_coord_with(idx, cluster_axis, (r + s["dir"]) % n))
            # Peer: the destination's stream that targets this chip (opposite direction, same link and slot).
            peer_core = tx_core_of[(-s["dir"], s["link"], s["slot"])]
            peer_virt = mesh_device.worker_core_from_logical_core(peer_core)

            reader_args = [len(s["heads"])]
            for h_local in s["heads"]:
                wcore, hi = head_owner[d * hpc + h_local]
                wv = mesh_device.worker_core_from_logical_core(wcore)
                reader_args += [wv.x, wv.y, hi, counter_addrs[hi]]
            tx_reader_rt[tx_core.x][tx_core.y] = reader_args

            writer_args = [
                int(dst_node.mesh_id),
                dst_node.chip_id,
                peer_virt.x,
                peer_virt.y,
                sem_ready_addr,
                sem_done_addr,
                output_tensor.buffer_address(),
                r,
                len(s["heads"]),
            ] + list(s["heads"])
            tx_writer_rt[tx_core.x][tx_core.y] = []
            tx_plan.append((s, tx_core, nbr_node, writer_args))

        tx_reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_tx_reader.cpp"),
            core_ranges=tx_set,
            compile_time_args=tx_reader_ct,
            defines=_ABL_DEFINES,
            runtime_args=tx_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )

        def _tx_writer(cores, ct, defines):
            sub = ttnn.RuntimeArgs()
            for c in cores:
                sub[c.x][c.y] = []
            return ttnn.KernelDescriptor(
                kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_tx_writer.cpp"),
                core_ranges=_core_ranges(cores),
                compile_time_args=ct,
                defines=defines,
                runtime_args=sub,
                config=ttnn.WriterConfigDescriptor(),
            )

        tx_writers = {}  # True (mux client) / False (direct) -> kernel descriptor index
        tx_kernels = []
        if direct_tx:
            tx_kernels.append(_tx_writer(direct_tx, tx_writer_ct, fabric_defines + _ABL_DEFINES))
            tx_writers[False] = len(worker_kernels) + 2 + len(tx_kernels) - 1
        if muxed_tx:
            tx_kernels.append(
                _tx_writer(muxed_tx, tx_writer_ct + list(mux_client_ct), fabric_defines + [("A2A_TX_MUX", "1")])
            )
            tx_writers[True] = len(worker_kernels) + 2 + len(tx_kernels) - 1
        mux_kernels = []
        if mux_groups:  # one kernel over every mux core (same CT args; RT args per core below)
            mux_rt0 = ttnn.RuntimeArgs()
            for g in mux_groups:
                mux_rt0[mux_core_of[g].x][mux_core_of[g].y] = []
            mux_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source="tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    core_ranges=_core_ranges(mux_cores),
                    compile_time_args=mux_config.kernel_compile_time_args(),
                    runtime_args=mux_rt0,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                    ),
                    opt_level=fabric_mux.KernelBuildOptLevel.O3,
                )
            )

        pd = ttnn.ProgramDescriptor(
            kernels=[*worker_kernels, worker_compute, tx_reader, *tx_kernels, *mux_kernels],
            semaphores=semaphores,
            cbs=cbs,
        )
        MUX_KERNEL_BASE = len(worker_kernels) + 2 + len(tx_kernels)
        for i, g in enumerate(mux_groups):
            mux_core = mux_core_of[g]
            nbr_node = mesh_device.get_fabric_node_id(_coord_with(idx, cluster_axis, (r + g[0]) % n))
            mux_rt = mux_config.kernel_runtime_args(
                source_node_id=my_node,
                destination_node_id=nbr_node,
                link_index=g[1],
                program_descriptor=pd,
                mux_logical_core=mux_core,
            )
            pd.kernels[MUX_KERNEL_BASE].runtime_args[mux_core.x][mux_core.y].extend(list(mux_rt))
        term_sem = {}  # mux group -> termination-master semaphore id
        for s, tx_core, nbr_node, writer_args in tx_plan:
            g = (s["dir"], s["link"])
            if g in mux_core_of:
                # Mux client block FIRST (MuxConn consumes it from arg 0). Slot 0 is the termination master.
                master = tx_core_of[(s["dir"], s["link"], 0)]
                conn_args = fabric_mux.client_runtime_args(
                    connection_valid=True,
                    is_termination_master=s["slot"] == 0,
                    channel_type=mux_channel,
                    mux_virtual_core=mesh_device.worker_core_from_logical_core(mux_core_of[g]),
                    client_index=s["slot"],
                    client_logical_core=tx_core,
                    config=mux_config,
                    program_descriptor=pd,
                    termination_master_virtual_core=mesh_device.worker_core_from_logical_core(master),
                    termination_master_semaphore_id=term_sem.get(g),
                )
                if s["slot"] == 0:
                    term_sem[g] = conn_args[10]
                kidx = tx_writers[True]
            else:
                # Fabric connection block FIRST (FabricStreamSender consumes it from arg 0).
                conn_args = ttnn._ttnn.ccl_host.build_ccl_fabric_rt_args(
                    mesh_device, my_node, nbr_node, s["link"], pd, tx_core, True
                )
                kidx = tx_writers[False]
            rt_ref = pd.kernels[kidx].runtime_args[tx_core.x][tx_core.y]
            rt_ref.extend(list(conn_args))
            rt_ref.extend(writer_args)

        _merge_semaphores(pd)
        if KN["tx_common"]:
            _tx_to_common(pd, len(worker_kernels) + 1, tx_cores, grid, counter_addrs[:heads_per_core])
        if KN["info"] and flat == 0:
            _info(pd)
        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd

    return mesh_pd
