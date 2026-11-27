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

KERNEL_DIR = Path(__file__).parent / "kernels"  # experiment copy of the op kernels + tx_writer_mr

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
HEAD_PLACEMENT = os.environ.get("TXR_PLACEMENT", "by_stream")
# R6 (Refinement 3): heads of each neighbour stream moved onto link 1 as a second stream; link 1 is then
# shared by two fabric_tx cores through a fabric mux core (ttnn.experimental.fabric_mux). 0 = one stream per
# link (no mux), the Phase 0 - Refinement 2 topology.
MUX_OVERFLOW_HEADS = 0  # experiment: the mux path is not used
# tx_rebalance_direct: last x heads of each neighbour stream go to the link-1 tx core (same DIRECT router
# connection as the antipode half, a second route). 0 = baseline topology.
DIRECT_OVERFLOW_HEADS = int(os.environ.get("TXR_OVERFLOW", "0"))
# "base" = the op's tx writer (one route per core; requires DIRECT_OVERFLOW_HEADS == 0); "mr" = multi-route writer
TX_WRITER = os.environ.get("TXR_WRITER", "mr")
# where the moved heads sit inside their destination block: "last" (fast rows, by_stream default) or "first"
OVERFLOW_POS = os.environ.get("TXR_OVERFLOW_POS", "last")
# multi-route writer: send each route's done right after its last packet (A2A_TXR_EARLY_DONE)
EARLY_DONE = os.environ.get("TXR_EARLY_DONE", "0") == "1"
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
    """Streams leaving one chip of an n-ring (tx_rebalance_direct variant).

    A stream = one fabric_tx core = (direction, link) with one or more destination segments
    ("segs": [{"offset", "hops", "heads"}]); "heads" is the flattened [(offset, h_local)] list in send-
    table order. The antipode (2k == n) is split in two halves, one per direction; within a direction,
    destinations sorted by hop count get links 0, 1, ... `overflow` (DIRECT_OVERFLOW_HEADS) moves x heads
    of the link-0 neighbour destination onto the link-1 stream as a second segment (second route on the
    same direct router connection).
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
        per_link = [[{"offset": off, "hops": hops, "heads": list(heads)}] for (_, hops, off, heads) in mine]
        if len(per_link) > 1 and 0 < overflow < hpc and mine[0][1] == 1:
            nb = per_link[0][0]
            if OVERFLOW_POS == "first":
                moved, nb["heads"] = nb["heads"][:overflow], nb["heads"][overflow:]
            else:
                moved, nb["heads"] = nb["heads"][-overflow:], nb["heads"][:-overflow]
            per_link[1].append({"offset": nb["offset"], "hops": nb["hops"], "heads": moved})
        for link, segs in enumerate(per_link):
            streams.append(
                {
                    "dir": direction,
                    "link": link,
                    "slot": 0,
                    "segs": segs,
                    "heads": [(g["offset"], h) for g in segs for h in g["heads"]],
                }
            )
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
        for g in s["segs"]:
            stream_heads[g["offset"]] = max(stream_heads.get(g["offset"], 0), len(g["heads"]))
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
    streams = _ring_streams(n, hpc, DIRECT_OVERFLOW_HEADS)
    assert TX_WRITER == "mr" or DIRECT_OVERFLOW_HEADS == 0, "the base tx writer has one route per core"
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

    def cb(index, dtype, page_size, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=program_set,  # uniform CB addresses on every program core (pull_block)
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
        w_reader_rt = {}  # (x, y) -> args; split per read-NoC set below
        compute_rt = ttnn.RuntimeArgs()
        w_writer_rt = {}
        for core, start, cnt in worker_assign:
            w_reader_rt[(core.x, core.y)] = [input_tensor.buffer_address(), weight.buffer_address(), start, cnt]
            compute_rt[core.x][core.y] = [cnt]
            w_writer_rt[(core.x, core.y)] = [output_tensor.buffer_address(), start, cnt, r] + counter_addrs[:cnt]
        w_writer_ct = [CB_OUT, Mt, Nt, mt_block, hpc, n, out_tile_bytes] + out_ct_accessor
        compute_ct = [CB_ATTN_RM, CB_ATTN_TILED, CB_WEIGHT, CB_OUT, Kt, Nt, Mt, mt_block]

        # Worker kernels: one reader/writer descriptor pair per read-NoC set (same sources and args).
        def _subset(rt, cores):
            sub = ttnn.RuntimeArgs()
            for c in cores:
                sub[c.x][c.y] = rt[(c.x, c.y)]
            return sub

        worker_kernels = []
        for on_noc1 in (False, True):
            cores = [c for c in worker_cores if _reads_on_noc1(c, grid) == on_noc1]
            if not cores:
                continue
            crs = _core_ranges(cores)
            if on_noc1:
                reader_cfg = ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.RISCV_1_default
                )
                writer_cfg = ttnn.DataMovementConfigDescriptor(
                    processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.RISCV_0_default
                )
            else:
                reader_cfg = ttnn.ReaderConfigDescriptor()
                writer_cfg = ttnn.WriterConfigDescriptor()
            worker_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_reader.cpp"),
                    core_ranges=crs,
                    compile_time_args=w_reader_ct,
                    defines=_ABL_DEFINES,
                    runtime_args=_subset(w_reader_rt, cores),
                    config=reader_cfg,
                )
            )
            worker_kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_writer.cpp"),
                    core_ranges=crs,
                    compile_time_args=w_writer_ct,
                    runtime_args=_subset(w_writer_rt, cores),
                    config=writer_cfg,
                )
            )
        worker_compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_compute.cpp"),
            core_ranges=worker_set,
            compile_time_args=compute_ct,
            defines=_ABL_DEFINES,
            runtime_args=compute_rt,
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
            nbr_node = mesh_device.get_fabric_node_id(_coord_with(idx, cluster_axis, (r + s["dir"]) % n))
            # Peer per destination: the destination's stream that targets this chip (opposite direction,
            # same link and slot) -- by the symmetric plan, the same logical core for every segment.
            peer_core = tx_core_of[(-s["dir"], s["link"], s["slot"])]
            peer_virt = mesh_device.worker_core_from_logical_core(peer_core)

            reader_args = [len(s["heads"])]
            for off, h_local in s["heads"]:
                d = (r + off) % n  # destination ring position
                wcore, hi = head_owner[d * hpc + h_local]
                wv = mesh_device.worker_core_from_logical_core(wcore)
                reader_args += [wv.x, wv.y, hi, counter_addrs[hi]]
            tx_reader_rt[tx_core.x][tx_core.y] = reader_args

            route_args = []
            seg_route = {}
            for gi, g in enumerate(s["segs"]):
                dst_node = mesh_device.get_fabric_node_id(_coord_with(idx, cluster_axis, (r + g["offset"]) % n))
                route_args += [int(dst_node.mesh_id), dst_node.chip_id, peer_virt.x, peer_virt.y]
                seg_route[g["offset"]] = gi
            tail = [sem_ready_addr, sem_done_addr, output_tensor.buffer_address(), r, len(s["heads"])]
            if TX_WRITER == "mr":
                writer_args = [len(s["segs"])] + route_args + tail + [h | (seg_route[o] << 16) for o, h in s["heads"]]
            else:
                assert len(s["segs"]) == 1
                writer_args = route_args + tail + [h for _, h in s["heads"]]
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
                kernel_source=str(
                    KERNEL_DIR / ("a2a_wkv_b2_tx_writer_mr.cpp" if TX_WRITER == "mr" else "a2a_wkv_b2_tx_writer.cpp")
                ),
                core_ranges=_core_ranges(cores),
                compile_time_args=ct,
                defines=defines,
                runtime_args=sub,
                config=ttnn.WriterConfigDescriptor(),
            )

        tx_writers = {}  # True (mux client) / False (direct) -> kernel descriptor index
        tx_kernels = []
        if direct_tx:
            tx_kernels.append(
                _tx_writer(
                    direct_tx,
                    tx_writer_ct,
                    fabric_defines
                    + _ABL_DEFINES
                    + ([("A2A_TXR_EARLY_DONE", "1")] if EARLY_DONE and TX_WRITER == "mr" else []),
                )
            )
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

        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd

    return mesh_pd
