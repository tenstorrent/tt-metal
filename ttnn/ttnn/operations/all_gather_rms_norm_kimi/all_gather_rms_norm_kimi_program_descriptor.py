# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""all_gather_rms_norm_kimi — host plan + MeshProgramDescriptor (one program per chip).

Regime R1 of op_design.md: L1-landing ring gather + fp32 stats-ahead, normalize-on-landing.

Core roles per chip (the same logical layout on every chip):
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
  from the router payload), PORT_BATCH_CHUNKS, INC_EVERY, OUT_DEPTH_CHUNKS, num_links.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---- block / transport knobs (host constants, defined once) ----
SLICES_PER_BLOCK = 1  # mechanism cap: a block never straddles slices
BLOCK_W = None  # tile columns of a slice per block; None = the whole slice (Wl_t)
PORT_BATCH_CHUNKS = 4  # chunks per port batch (cb_port_chunks holds 2 batches)  [TRANSPORT]
PORT_BATCH_CHUNKS_MAX = 8  # packet-header pool cap
INC_EVERY = 8  # fused arrival increment every this many chunks (+ every entry end)  [TRANSPORT]
OUT_DEPTH_CHUNKS = 2  # cb_out depth in chunks
STATS_SCRATCH_CAP_BYTES = 64 * 1024

FWD, BWD = 0, 1
_OPP = {FWD: BWD, BWD: FWD}

# ---- compute-core CB indices ----
CB_X_BLOCK = 0  # backed on `landing` (zero-copy)
CB_GAMMA = 1
CB_OUT = 2
CB_PARTIALS = 3  # G * block_h fp32 col tiles (UnpackToDestFp32)
CB_XSQ_SUM = 4  # fp32, UnpackToDestFp32
CB_RSTD = 5
CB_PARTIAL = 6
CB_RSTD_FULL = 7
CB_SCALE = 8
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
NUM_LOCAL_SEMS = 6


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
    ports: dict  # (l, d) -> logical CoreCoord
    receivers: dict  # (l, d) -> logical CoreCoord
    hub: object
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
    def op_cores(self):
        return list(self.ports.values()) + list(self.receivers.values()) + [self.hub] + self.compute

    @property
    def slot_bytes(self):  # one slice's fp32 partial vector: Mt rows x 32 values
        return self.Mt * 128


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
    key = (id(mesh_device), tuple(padded_shape), G, L, gx, gy, max_payload)
    if key in _GEO_CACHE:
        return _GEO_CACHE[key][1]

    block_w = Wl_t if BLOCK_W is None else BLOCK_W
    if Wl_t % block_w != 0 or block_w != Wl_t:
        # Phase 0: block_w is the whole slice width (a narrower block needs same-slice partial combining).
        raise ValueError(f"all_gather_rms_norm_kimi: block_w {block_w} must equal the slice width {Wl_t} tiles")
    if G * Mt * 128 > STATS_SCRATCH_CAP_BYTES:
        raise ValueError(f"all_gather_rms_norm_kimi: M = {M} too tall for the stats scratch (G*M*4 B > 64 KiB)")

    # ---- placement: ports + receivers + hub on logical row 0 (next to the Ethernet row), compute on the rest ----
    n_aux = 2 * L + 1
    if 2 * L + n_aux > gx:
        raise ValueError(f"all_gather_rms_norm_kimi: grid row of {gx} cores cannot hold {2 * L} ports + {n_aux} aux")
    x0 = max(0, gx // 2 - L)
    port_cores = [ttnn.CoreCoord(x0 + i, 0) for i in range(2 * L)]
    ports = {(l, d): port_cores[2 * l + d] for l in range(L) for d in (FWD, BWD)}
    free0 = [ttnn.CoreCoord(x, 0) for x in range(gx) if not (x0 <= x < x0 + 2 * L)]
    receivers = {(l, d): free0[2 * l + d] for l in range(L) for d in (FWD, BWD)}
    hub = free0[2 * L]
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
    chunk_tiles = max(d for d in range(1, cap + 1) if (block_h * block_w) % d == 0 and (last_h * block_w) % d == 0)
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
        ports=ports,
        receivers=receivers,
        hub=hub,
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
    words = geo.G * geo.Mt * 32
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


def stream_incs(geo, entries):
    """Per chunk of the stream (in order): the arrival-counter value that covers it; and the total increments.
    The sender increments on every INC_EVERY-th chunk and on each entry's last chunk."""
    thresholds = []
    sent = incs = 0
    pending = 0
    for _, ms in entries:
        n_entry = sum(geo.nchunks(m) for m in ms)
        k = 0
        for m in ms:
            for _ in range(geo.nchunks(m)):
                sent += 1
                k += 1
                pending += 1
                if sent % INC_EVERY == 0 or k == n_entry:
                    incs += 1
                    thresholds += [incs] * pending
                    pending = 0
    assert pending == 0
    return thresholds, incs


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
    """Every op core: the compute rectangles + the row-0 ports / receivers / hub (merged into x-runs)."""
    row0 = sorted({c.x for c in list(geo.ports.values()) + list(geo.receivers.values()) + [geo.hub]})
    runs, start = [], row0[0]
    for a, b in zip(row0, row0[1:] + [None]):
        if b != a + 1:
            runs.append((start, a))
            start = b
    ranges = [ttnn.CoreRange(ttnn.CoreCoord(x0, 0), ttnn.CoreCoord(x1, 0)) for x0, x1 in runs]
    return ttnn.CoreRangeSet(ranges + list(compute_crs(geo).ranges()))


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
    W = G * geo.Wl_t * 32

    comp_crs = compute_crs(geo)
    port_list = [geo.ports[(l, d)] for l in range(L) for d in (FWD, BWD)]
    port_crs = _crs(port_list)
    recv_crs = _crs([geo.receivers[(l, d)] for l in range(L) for d in (FWD, BWD)])
    hub_crs = _crs([geo.hub])
    op_crs = op_core_crs(geo)

    hub_v = virt(geo.hub)
    port_v = {k: virt(c) for k, c in geo.ports.items()}
    recv_v = {k: virt(c) for k, c in geo.receivers.items()}

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
            program = ttnn.ProgramDescriptor()
            program.custom_program_hash = _hash(
                "all_gather_rms_norm_kimi",
                geo.key[1:],
                coord,
                epsilon,
                cfg.math_fidelity,
                cfg.math_approx_mode,
                BLOCK_W,
                PORT_BATCH_CHUNKS,
                INC_EVERY,
                OUT_DEPTH_CHUNKS,
                gamma_page,
            )
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=s, core_ranges=op_crs, initial_value=0) for s in range(NUM_LOCAL_SEMS)
            ]

            # ---------------- CBs ----------------
            def cb(idx, dtype, pages, cores):
                page = ttnn.tile_size(dtype)
                return ttnn.CBDescriptor(
                    total_size=pages * page,
                    core_ranges=cores,
                    format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=idx, data_format=dtype, page_size=page)],
                )

            bh, bw = geo.block_h, geo.block_w
            program.cbs = [
                ttnn.cb_descriptor_from_sharded_tensor(CB_X_BLOCK, res.landing),
                cb(CB_GAMMA, ttnn.bfloat16, bw, comp_crs),
                cb(CB_OUT, ttnn.bfloat16, OUT_DEPTH_CHUNKS * chunk, comp_crs),
                cb(CB_PARTIALS, ttnn.float32, G * bh, comp_crs),
                cb(CB_XSQ_SUM, ttnn.float32, bh, comp_crs),
                cb(CB_RSTD, ttnn.float32, bh, comp_crs),
                cb(CB_PARTIAL, ttnn.float32, bh, comp_crs),
                cb(CB_RSTD_FULL, ttnn.float32, bh, comp_crs),
                cb(CB_SCALE, ttnn.float32, bh * bw, comp_crs),
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
            ]
            reader_ct += list(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
            reader_ct += list(ttnn.TensorAccessorArgs(gamma).get_compile_time_args())
            writer_common = [out_addr, stats_addr, p, _xy(hub_v)] + geo.row_base
            writer_ct = geo_ct + [CB_PARTIAL, CB_OUT, SEM_OWN_PARTIALS, tile_bytes]
            writer_ct += list(ttnn.TensorAccessorArgs(output).get_compile_time_args())
            compute_common = [p] + geo.row_base
            compute_ct = geo_ct + [_f32_bits(1.0 / W), _f32_bits(epsilon)]

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
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=ttnn.NOC.NOC_0
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_writer.cpp"),
                    core_ranges=comp_crs,
                    compile_time_args=writer_ct,
                    runtime_args=ttnn.RuntimeArgs(),
                    common_runtime_args=writer_common,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_1
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
                    core = geo.ports[(l, d)]
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
                    rargs = [landing_addr, len(geo.link_rows(l)), n_chunks, len(blocks)]
                    for q, m, relay, end in blocks:
                        rargs += [_xy(virt(geo.core(q, m))), geo.nchunks(m) | (int(relay) << 16) | (int(end) << 17)]
                    preader_rt[core.x][core.y] = rargs
                    # sender
                    sargs = [landing_addr, stats_addr, int(active), n_chunks, len(blocks), len(slots)]
                    sargs += [ready_addr, _xy(port_v[(l, _OPP[d])]), arrival_addr, _xy(recv_v[(l, d)])]
                    sargs += [sfwd_addr if d == FWD else sbwd_addr, _xy(hub_v)]
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
                    core = geo.receivers[(l, d)]
                    up_p = (p - 1) % G if d == FWD else (p + 1) % G
                    up_entries = port_stream(geo, up_p, l, d)
                    local_entries = port_stream(geo, p, l, d)
                    thresholds, incs = stream_incs(geo, up_entries)
                    events = []
                    t = 0
                    for k, (q, ms) in enumerate(up_entries):
                        relayed = set()
                        if k + 1 < len(local_entries):
                            rq, rms = local_entries[k + 1]
                            assert rq == q
                            relayed = set(rms)
                        for m in ms:
                            n = geo.nchunks(m)
                            xy = _xy(virt(geo.core(q, m)))
                            for c in range(n):
                                flag = 1 if (c == n - 1 and m in relayed) else 0
                                events += [thresholds[t], xy | (flag << 16)]
                                t += 1
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
                len(rects),
            ]
            for x0, y0, x1, y1, n in rects:
                a, b = virt(ttnn.CoreCoord(x0, y0)), virt(ttnn.CoreCoord(x1, y1))
                hub_args += [min(a.x, b.x), min(a.y, b.y), max(a.x, b.x), max(a.y, b.y), n]
            hub_rt = ttnn.RuntimeArgs()
            hub_rt[geo.hub.x][geo.hub.y] = hub_args
            kernels.append(
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "all_gather_rms_norm_kimi_hub.cpp"),
                    core_ranges=hub_crs,
                    compile_time_args=[G, geo.slot_bytes, SEM_OWN_PARTIALS, SEM_STATS_TX, SEM_STATS_READY],
                    runtime_args=hub_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=ttnn.NOC.NOC_0
                    ),
                )
            )
            program.kernels = kernels
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = program
    return mesh_desc
