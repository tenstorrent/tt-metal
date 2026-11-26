# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""a2a_wkv_b2 — MeshProgramDescriptor (one ProgramDescriptor per chip).

Regime R1 of op_design.md (matmul-first, heads over cores, streamed mt rounds):

  worker cores (one head each at 96 workers):
    reader  (NCRISC, NoC 0): W[h] (Kt x Nt bf8 tiles) once per head, then attn sticks per mt block
    compute               : tilize(mt_block x Kt) -> matmul_block([mt_block x Kt] @ [Kt x Nt]) -> cb_out
    writer  (BRISC,  NoC 1): local head -> output L1 tiles; every block -> bump sem_blocks_done
  fabric_tx cores (one per stream = (direction, link)):
    reader  (NCRISC, NoC 0): poll a worker's sem_blocks_done, pull its block from cb_out into cb_tx_stage
    writer  (BRISC)        : ready handshake, scatter-write packets to the destination's output tiles
                             over the fabric, done handshake (flush), wait for the peer's done

Every block knob (mt_block, attn_rm_depth, tx_depth, tiles_per_packet, heads_per_core) is defined
once here and every dependent CB size / loop count derives from it.
"""

from __future__ import annotations

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Block knobs (single source of truth)
# ---------------------------------------------------------------------------
MT_BLOCK = 1  # token tile rows per block (op_design.md: mt_block)
ATTN_RM_DEPTH = 2  # cb_attn_rm depth in blocks (reader fetches mt+1 while compute tilizes mt)
TX_DEPTH = 2  # cb_tx_stage depth in packets (pull p+1 overlaps send p)
MAX_MATMUL_CORES = 110  # matmul di/dt cap
SCATTER_MAX_CHUNKS = 4  # NOC_SCATTER_WRITE_MAX_CHUNKS
DEST_TILES_HALF_SYNC_16B = 8  # DEST capacity (half sync, 16-bit)

# CB indices (any distinct values)
CB_WEIGHT = 0
CB_ATTN_RM = 1
CB_ATTN_TILED = 2
CB_OUT = 3
CB_TX_STAGE = 4

# Program semaphore ids
SEM_BLOCKS_DONE = 0

TILE = 32


def _div_up(a, b):
    return (a + b - 1) // b


# ---------------------------------------------------------------------------
# Cross-chip semaphores: created once per mesh device on the first (eager) call
# ---------------------------------------------------------------------------
_SEM_CACHE = []  # [(mesh_device, sem_ready, sem_done)]


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


# ---------------------------------------------------------------------------
# Ring route plan (derived from the mesh; never hard-coded)
# ---------------------------------------------------------------------------


def _ring_streams(n, hpc):
    """Streams leaving one chip of an n-ring. Each stream = (direction, hops, link, dst_offset, heads).

    direction +1 = towards ring position r+1, -1 = towards r-1. The antipode (2k == n) is split in
    two halves, one per direction. Within a direction, streams sorted by hop count get links 0, 1, ...
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
        for link, (d, hops, offset, heads) in enumerate(mine):
            streams.append({"dir": d, "hops": hops, "link": link, "offset": offset, "heads": heads})
    return streams


def _stream_core_index(direction, link, num_links):
    return (0 if direction > 0 else 1) * num_links + link


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
    streams = _ring_streams(n, hpc)
    num_streams = len(streams)
    per_dir = max(sum(1 for s in streams if s["dir"] == d) for d in (+1, -1))
    assert per_dir == num_links, "validate() guarantees num_links == streams per direction"

    grid = mesh_device.compute_with_storage_grid_size()
    total_cores = grid.x * grid.y
    all_logical = [ttnn.CoreCoord(i % grid.x, i // grid.x) for i in range(total_cores)]
    tx_cores = all_logical[total_cores - num_streams :]
    num_workers = min(H, MAX_MATMUL_CORES, total_cores - num_streams)
    worker_crs = ttnn.num_cores_to_corerangeset(num_workers, grid, True)
    (_, _, g1, g2, heads_g1, heads_g2) = ttnn.split_work_to_cores(worker_crs, H, True)
    worker_assign = []  # [(core, head_start, num_heads)]
    head = 0
    for group, cnt in ((g1, heads_g1), (g2, heads_g2)):
        if cnt == 0:
            continue
        for core in sorted(ttnn.corerange_to_cores(group, None, True), key=lambda c: (c.y, c.x)):
            worker_assign.append((core, head, cnt))
            head += cnt
    assert head == H
    heads_per_core = max(cnt for _, _, cnt in worker_assign)
    head_owner = {}  # global head -> (worker core, head index within that core)
    for core, start, cnt in worker_assign:
        for i in range(cnt):
            head_owner[start + i] = (core, i)

    worker_cores = [c for c, _, _ in worker_assign]
    worker_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in worker_cores])
    tx_set = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in tx_cores])
    program_set = worker_set.merge(tx_set)

    tx_core_of = {}
    for s in streams:
        tx_core_of[(s["dir"], s["link"])] = tx_cores[_stream_core_index(s["dir"], s["link"], num_links)]

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

        # ---------------- CBs ----------------
        cbs = [
            cb(CB_WEIGHT, ttnn.bfloat8_b, w_tile_bytes, Kt * Nt),
            cb(CB_ATTN_RM, ttnn.bfloat16, attn_tile_bytes, ATTN_RM_DEPTH * mt_block * Kt),
            cb(CB_ATTN_TILED, ttnn.bfloat16, attn_tile_bytes, mt_block * Kt),
            cb(CB_OUT, ttnn.bfloat8_b, out_tile_bytes, heads_per_core * Mt * Nt),
            cb(CB_TX_STAGE, ttnn.bfloat8_b, out_tile_bytes, TX_DEPTH * tiles_per_packet),
        ]
        semaphores = [
            ttnn.SemaphoreDescriptor(
                id=SEM_BLOCKS_DONE, core_type=ttnn.CoreType.WORKER, core_ranges=program_set, initial_value=0
            )
        ]

        # ---------------- worker kernels ----------------
        w_reader_ct = [CB_WEIGHT, CB_ATTN_RM, Kt, Nt, Mt, mt_block, T, stick_bytes, attn_page_bytes, w_tile_bytes]
        w_reader_ct += ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
        w_reader_ct += ttnn.TensorAccessorArgs(weight).get_compile_time_args()
        w_reader_rt = ttnn.RuntimeArgs()
        compute_rt = ttnn.RuntimeArgs()
        w_writer_rt = ttnn.RuntimeArgs()
        for core, start, cnt in worker_assign:
            w_reader_rt[core.x][core.y] = [input_tensor.buffer_address(), weight.buffer_address(), start, cnt]
            compute_rt[core.x][core.y] = [cnt]
            w_writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start, cnt, r]
        w_writer_ct = [CB_OUT, Mt, Nt, mt_block, hpc, n, SEM_BLOCKS_DONE, out_tile_bytes] + out_ct_accessor
        compute_ct = [CB_ATTN_RM, CB_ATTN_TILED, CB_WEIGHT, CB_OUT, Kt, Nt, Mt, mt_block]

        worker_reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_reader.cpp"),
            core_ranges=worker_set,
            compile_time_args=w_reader_ct,
            runtime_args=w_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        worker_compute = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_compute.cpp"),
            core_ranges=worker_set,
            compile_time_args=compute_ct,
            runtime_args=compute_rt,
            config=compute_kernel_config,
        )
        worker_writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_writer.cpp"),
            core_ranges=worker_set,
            compile_time_args=w_writer_ct,
            runtime_args=w_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )

        # ---------------- fabric_tx kernels ----------------
        tx_reader_ct = [CB_TX_STAGE, CB_OUT, SEM_BLOCKS_DONE, Mt, Nt, mt_block, tiles_per_packet, out_tile_bytes]
        tx_writer_ct = [CB_TX_STAGE, Mt, Nt, mt_block, tiles_per_packet, out_tile_bytes, n, l1_alignment]
        tx_writer_ct += out_ct_accessor
        tx_reader_rt = ttnn.RuntimeArgs()
        tx_writer_rt = ttnn.RuntimeArgs()
        tx_plan = []  # (tx core, neighbour node, link, writer op args)
        for s in streams:
            tx_core = tx_core_of[(s["dir"], s["link"])]
            d = (r + s["offset"]) % n  # destination ring position
            dst_coord = _coord_with(idx, cluster_axis, d)
            dst_node = mesh_device.get_fabric_node_id(dst_coord)
            nbr_node = mesh_device.get_fabric_node_id(_coord_with(idx, cluster_axis, (r + s["dir"]) % n))
            # Peer: the destination's stream that targets this chip (opposite direction, same link).
            peer_core = tx_core_of[(-s["dir"], s["link"])]
            peer_virt = mesh_device.worker_core_from_logical_core(peer_core)

            reader_args = [len(s["heads"])]
            for h_local in s["heads"]:
                wcore, hi = head_owner[d * hpc + h_local]
                wv = mesh_device.worker_core_from_logical_core(wcore)
                reader_args += [wv.x, wv.y, hi]
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
            tx_plan.append((tx_core, nbr_node, s["link"], writer_args))

        tx_reader = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_tx_reader.cpp"),
            core_ranges=tx_set,
            compile_time_args=tx_reader_ct,
            runtime_args=tx_reader_rt,
            config=ttnn.ReaderConfigDescriptor(),
        )
        tx_writer = ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "a2a_wkv_b2_tx_writer.cpp"),
            core_ranges=tx_set,
            compile_time_args=tx_writer_ct,
            defines=fabric_defines,
            runtime_args=tx_writer_rt,
            config=ttnn.WriterConfigDescriptor(),
        )

        pd = ttnn.ProgramDescriptor(
            kernels=[worker_reader, worker_compute, worker_writer, tx_reader, tx_writer],
            semaphores=semaphores,
            cbs=cbs,
        )
        TX_WRITER_IDX = 4
        for tx_core, nbr_node, link, writer_args in tx_plan:
            # Fabric connection block FIRST (FabricStreamSender consumes it from arg 0).
            fabric_args = ttnn._ttnn.ccl_host.build_ccl_fabric_rt_args(
                mesh_device, my_node, nbr_node, link, pd, tx_core, True
            )
            rt_ref = pd.kernels[TX_WRITER_IDX].runtime_args[tx_core.x][tx_core.y]
            rt_ref.extend(list(fabric_args))
            rt_ref.extend(writer_args)

        mesh_pd[ttnn.MeshCoordinateRange(coord, coord)] = pd

    return mesh_pd
