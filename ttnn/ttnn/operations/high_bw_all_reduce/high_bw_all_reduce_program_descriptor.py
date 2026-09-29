# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""high_bw_all_reduce — MeshProgramDescriptor for the chain schemes of op_design.md:
R1 `chain_line` (axis line), R2 `chain_snake_line` (cluster_axis=None, Linear) and R3
`rotated_chain_ring` (Ring: G slices, slice j's chain starts at ring position j).

Per device, per lane l (one Fabric link per hop and direction):
  * 1 port core : port_fwd (RISCV_0, connection toward p+1) + port_bwd (RISCV_1, toward p-1)
  * W reducer cores : reader (RISCV_1) / compute / writer (RISCV_0)

Every chunk is reduced hop by hop along its slice's chain (partials over Fabric into
the downstream reducer's landing CB, always p -> p+1) and the final sum is relayed
back (p -> p-1) by the port cores, each device writing its output DRAM on the way.
Chunk roles (head / middle / tail) are per chunk: kernels/high_bw_all_reduce_roles.hpp.

All block knobs are single-source host constants below; every CB page count,
ring size and trip count is derived from them.
"""

from __future__ import annotations

from pathlib import Path

import ttnn
from ttnn.operations.ccl import Topology

KERNEL_DIR = Path(__file__).parent / "kernels"

# ---------------------------------------------------------------------------
# Block-model knobs (single source of truth)
# ---------------------------------------------------------------------------
CHUNK_BYTES_TARGET = 65536  # `tile` block: chunk_tiles derived from this
REDUCERS_PER_LANE = 4  # `tile` -> reducer sub-assignment cap (W)
# Credit coalescing (Refinement 3). Every cross-device credit is a header-only Fabric packet that
# costs the EDM a full packet slot: on BH FABRIC_2D one worker channel moves ~237 cycles/packet
# whatever its size, so per-chunk credits take ~1/packets_per_chunk of each link direction
# (measured on a send-only stream: +1 credit per 16 data packets = +5.5%). Both credit streams —
# landing grants (p+1 -> p) and final-landing grants (p -> p+1) — send one packet per
# `credit_batch` blocks of a reducer; batching only one of them leaves the other direction binding.
# A withheld credit lowers the sender's effective ring depth by credit_batch - 1, so both ring
# depths derive from it (see _ring_depths), keeping the Phase 0 effective depths.
CREDIT_BATCH_CHUNKS = 4  # 1 = per-chunk credits (the Phase 0 protocol, byte-identical)
RECV_EFFECTIVE_DEPTH = 2  # landing slots per reducer the upstream sender can always fill
FINAL_EFFECTIVE_DEPTH = 1  # final-landing slots per reducer the downstream relay can always fill
INPUT_DEPTH_CHUNKS = 2  # cb_local_input depth
REDUCED_DEPTH_CHUNKS = 2  # cb_reduced depth
STAGING_DEPTH_PER_REDUCER = 1  # port staging ring slots per reducer
# Bank-run chunk layout (Refinement 4, kernels/high_bw_all_reduce_chunk_io.hpp): chunks are stored
# bank-major in L1 so every DRAM read / write of a chunk is one transfer per DRAM bank (run_tiles
# pages) instead of one per page. 0 = page-by-page identity layout (the pre-Refinement-4 transfers).
# Parked at 0 (measured, BH 2x2): it won 5-6% while the relay RISC also wrote DRAM (co-located
# ports), but once port_drain owns that write (SPLIT_PORTS) it is flat at 1 link and costs 2-6% at
# 2 links (both lanes' 8 KiB bursts hit the same bank at once). A live knob for other regimes.
BANK_RUN_LAYOUT = 0

CONTROL_WORD_STRIDE = 16  # bytes between control-array words (L1 alignment)
NUM_CONTROL_ARRAYS = 5  # staged[], final_ready[], partial_granted[], final_freed[], drained[]
# Fabric data-packet headers in flight per port connection (<= the per-RISC PacketHeaderPool
# budget minus the one credit header). Passed to both port kernels as a CT arg.
MAX_DATA_HEADERS = 8

# CB indices (semantic)
CB_REMOTE_PARTIAL = 0
CB_LOCAL_INPUT = 1
CB_REDUCED = 2
NUM_CIRCULAR_BUFFERS = 64  # length of ComputeConfigDescriptor.unpack_to_dest_mode (one entry per CB id)

# Program semaphore ids (fabric connection setup appends its own after these)
SEM_GO = 0
SEM_EGRESS_CREDIT = 1
SEM_FINAL_EGRESS = 2

# Reducer NoC placement (perf lamp "Placement / NoC selection"): reader on NoC0, writer on NoC1.
REDUCER_READER_NOC = ttnn.NOC.RISCV_0_default  # NoC0
REDUCER_WRITER_NOC = ttnn.NOC.RISCV_1_default  # NoC1
# Port NoC placement: port_fwd (RISCV_0) and port_bwd (RISCV_1).
PORT_FWD_NOC = ttnn.NOC.RISCV_1_default  # NoC1 (the WriterConfigDescriptor default)
PORT_BWD_NOC = ttnn.NOC.RISCV_0_default  # NoC0 (the ReaderConfigDescriptor default)
# Split ports (Refinement 4, design Perf lamp "port co-location"): 1 puts port_fwd on its own core
# and runs port_drain (output DRAM writes of the relayed finals, from local L1) on the backward-port
# core's RISCV_0 next to port_bwd, so the relay RISC only forwards over Fabric and the DRAM write
# runs concurrently on the other NoC. 0 = one port core per lane running port_fwd + port_bwd, with
# port_bwd writing DRAM itself (the pre-Refinement-4 layout).
SPLIT_PORTS = 1
PORT_DRAIN_NOC = ttnn.NOC.RISCV_1_default  # NoC1: the NoC port_bwd's Fabric sends are not on
# Lane placement: 0 packs lanes row-major (lane l at linear core l * (REDUCERS_PER_LANE + 1));
# k > 0 starts lane l on grid row l * k.
LANE_ROW_STRIDE = 1

PER_REDUCER_GSEMS = ("partial_credit", "final_arrival", "final_credit")

_GSEM_CACHE = {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _div_up(a, b):
    return (a + b - 1) // b


def _axis_groups(mesh_shape, axis):
    rows, cols = mesh_shape
    if axis == 0:
        return [[(r, c) for r in range(rows)] for c in range(cols)]
    return [[(r, c) for c in range(cols)] for r in range(rows)]


def _snake_line(rows, cols):
    """Row snake: every edge is a direct one-hop link."""
    path = []
    for r in range(rows):
        cs = range(cols) if r % 2 == 0 else range(cols - 1, -1, -1)
        path += [(r, c) for c in cs]
    return path


def _snake_cycle(rows, cols):
    """Hamiltonian cycle of one-hop links (needs an even mesh dimension): row 0 left to right,
    snake columns 1.. over rows 1.., then return up column 0 to close on (0, 0)."""
    if rows % 2 != 0:
        return [(r, c) for (c, r) in _snake_cycle(cols, rows)]
    path = [(0, c) for c in range(cols)]
    for r in range(1, rows):
        cs = range(cols - 1, 0, -1) if r % 2 == 1 else range(1, cols)
        path += [(r, c) for c in cs]
    path += [(r, 0) for r in range(rows - 1, 0, -1)]
    return path


def _wrapped_axes():
    cfg = ttnn.get_fabric_config()
    wraps = {
        "FABRIC_2D_TORUS_Y": (0,),
        "FABRIC_2D_TORUS_X": (1,),
        "FABRIC_2D_TORUS_XY": (0, 1),
    }
    for name, axes in wraps.items():
        if hasattr(ttnn.FabricConfig, name) and cfg == getattr(ttnn.FabricConfig, name):
            return axes
    return ()


def route(mesh_shape, cluster_axis, topology):
    """(groups, is_ring): every group is an ordered path of one-hop neighbours; a ring also uses
    the closing edge path[-1] -> path[0]. Raises ValueError for a Ring the cluster cannot close."""
    rows, cols = mesh_shape
    want_ring = topology == Topology.Ring
    axis = cluster_axis
    if axis is None:
        if rows >= 2 and cols >= 2:
            if not want_ring:
                return [_snake_line(rows, cols)], False
            if rows % 2 and cols % 2:
                raise ValueError(
                    f"high_bw_all_reduce: Ring over cluster_axis=None needs an even mesh dimension, mesh {rows}x{cols}"
                )
            return [_snake_cycle(rows, cols)], True
        axis = 1 if rows == 1 else 0  # a 1-D mesh: the whole mesh is the single axis line
    groups = _axis_groups(mesh_shape, axis)
    if want_ring:
        if axis not in _wrapped_axes():
            raise ValueError(
                f"high_bw_all_reduce: Ring along mesh axis {axis} needs a fabric config that wraps it "
                f"(got {ttnn.get_fabric_config()})"
            )
        if len(groups[0]) >= 3:
            return groups, True
        # A 2-device ring is the line: same link pair, same bytes per direction.
    return groups, False


def _edges(groups, is_ring):
    for path in groups:
        pairs = list(zip(path, path[1:]))
        if is_ring:
            pairs.append((path[-1], path[0]))
        yield from pairs


def _node(mesh_device, coord):
    return mesh_device.get_fabric_node_id(ttnn.MeshCoordinate(*coord))


def usable_links(mesh_device, groups, is_ring):
    """min over every edge of the route (snake corners and the closing edge included), both
    directions, of the forwarding link count."""
    best = None
    for a, b in _edges(groups, is_ring):
        na, nb = _node(mesh_device, a), _node(mesh_device, b)
        for s, d in ((na, nb), (nb, na)):
            n = len(ttnn.get_forwarding_link_indices(s, d))
            best = n if best is None else min(best, n)
    return best


def _linear_core(idx, grid):
    return ttnn.CoreCoord(idx % grid.x, idx // grid.x)


def _split_ports(group_size):
    """Path gate for SPLIT_PORTS (measured, BH 2x2): only a chain with middle devices (G >= 3: the
    None snake and ring) has a device that both relays a final over Fabric and writes it to DRAM;
    there the split wins 5-14%. A G = 2 axis line never does both on one device (flat to +0.6%),
    so it keeps the co-located ports. Returns 0 / 1 (the extra core per lane)."""
    return SPLIT_PORTS if group_size >= 3 else 0


def _lane_stride(split):
    """Cores per lane: the port core(s), then REDUCERS_PER_LANE reducers."""
    return 1 + split + REDUCERS_PER_LANE


def _lane_core(lane, i, grid, split):
    """Core i of lane `lane` (i = 0: port_fwd, `split`: port_bwd, then the reducers). LANE_ROW_STRIDE = 0 packs lanes
    row-major one after another; k > 0 starts lane l on grid row l * k."""
    stride = _lane_stride(split)
    if LANE_ROW_STRIDE and lane * LANE_ROW_STRIDE < grid.y and stride <= grid.x:
        return ttnn.CoreCoord(i, lane * LANE_ROW_STRIDE)
    return _linear_core(lane * stride + i, grid)


def _core_set(cores):
    return ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])


def _global_semaphores(mesh_device, cluster_axis, route_kind, num_lanes, grid, split):
    """Persistent per-config cross-device counters (cumulative within an invocation,
    atomically decremented by exactly the invocation total at its end). Keyed on the real
    route kind (line / ring), so Linear and Ring configs never share counters."""
    key = (id(mesh_device), cluster_axis, route_kind, num_lanes, REDUCERS_PER_LANE, LANE_ROW_STRIDE, split)
    if key not in _GSEM_CACHE:
        cores = _core_set([_lane_core(l, i, grid, split) for l in range(num_lanes) for i in range(_lane_stride(split))])
        # Per reducer index: the port kernels serve reducers independently, so every credit /
        # arrival counter between ports is per reducer (the reducer landing arrival lives on the
        # reducer core itself and needs only one).
        names = ["partial_arrival"]
        names += [f"{kind}_{r}" for kind in PER_REDUCER_GSEMS for r in range(REDUCERS_PER_LANE)]
        sems = {name: ttnn.create_global_semaphore(mesh_device, cores, 0) for name in names}
        _GSEM_CACHE[key] = (mesh_device, sems)  # keep the mesh alive alongside its semaphores
    return {k: int(ttnn.get_global_semaphore_address(v)) for k, v in _GSEM_CACHE[key][1].items()}


def _height_sharded_l1(mesh_device, cores, pages_per_core, dtype):
    """Per-call L1 scratch: one shard of `pages_per_core` tiles per core, uniform address."""
    core_set = _core_set(cores)
    shard_h = pages_per_core * 32
    mem_cfg = ttnn.create_sharded_memory_config(
        shape=(shard_h, 32),
        core_grid=core_set,
        strategy=ttnn.ShardStrategy.HEIGHT,
        orientation=ttnn.ShardOrientation.ROW_MAJOR,
        use_height_and_width_as_shard_shape=True,
    )
    return ttnn.allocate_tensor_on_device(
        ttnn.Shape([len(cores) * shard_h, 32]), dtype, ttnn.TILE_LAYOUT, mesh_device, mem_cfg
    )


def _credit_batch(group_size, num_lanes):
    """Credit batch per path. Refinement 3 gated the 2-link snake / ring (multi-lane chains with
    middle devices) to per-chunk credits because deeper rings measured +2-7% there; Refinement 4
    traced that to both lanes sharing one core row (LANE_ROW_STRIDE = 0). With a row per lane the
    batch wins on those cells too (BH 2x2, 32 MB: None-Linear 913 -> 894 us, None-Ring 781 ->
    713 us), so every path uses CREDIT_BATCH_CHUNKS. Kept as the single per-path hook."""
    return CREDIT_BATCH_CHUNKS


def _ring_depths(credit_batch):
    """(landing ring chunks per reducer, final-landing ring chunks per reducer). Deadlock freedom
    of a withheld credit needs each ring >= credit_batch (see port_fwd / port_bwd)."""
    recv_depth = RECV_EFFECTIVE_DEPTH + credit_batch - 1
    final_depth = FINAL_EFFECTIVE_DEPTH + credit_batch - 1
    assert recv_depth >= credit_batch and final_depth >= credit_batch
    return recv_depth, final_depth


def _scratch_cb(cb_index, scratch, first_page, pages, tile_bytes, core_set):
    """A reducer CB overlaid on the shared per-call scratch shard at page offset `first_page`."""
    return ttnn.cb_descriptor_from_sharded_tensor(
        cb_index, scratch, address_offset=first_page * tile_bytes, total_size=pages * tile_bytes, core_ranges=core_set
    )


# ---------------------------------------------------------------------------
# Build + dispatch (exactly one generic_op)
# ---------------------------------------------------------------------------


def build_and_dispatch(input_tensor, *, cluster_axis, topology, num_links):
    mesh_device = input_tensor.device()
    mesh_shape = tuple(mesh_device.shape)
    groups, is_ring = route(mesh_shape, cluster_axis, topology)
    group_size = len(groups[0])
    num_slices = group_size if is_ring else 1  # R3: one rotated chain per ring position
    placement = {coord: (path, p) for path in groups for p, coord in enumerate(path)}

    # --- lanes (num_links) ---
    if group_size >= 2:
        links_avail = usable_links(mesh_device, groups, is_ring)
        if num_links is None:
            num_lanes = links_avail
        else:
            if num_links <= 0 or num_links > links_avail:
                raise ValueError(
                    f"high_bw_all_reduce: num_links={num_links} must be in [1, {links_avail}] "
                    "(usable links on every edge of the route)"
                )
            num_lanes = num_links
    else:
        if num_links is not None and num_links <= 0:
            raise ValueError(f"high_bw_all_reduce: num_links={num_links} must be > 0")
        num_lanes = num_links or 1
    if num_lanes < 1:
        raise ValueError("high_bw_all_reduce: no usable fabric links on the selected route")

    # --- derived block quantities ---
    dtype = input_tensor.dtype
    # fp32 end-to-end: every CB page and the wire carry the input dtype (never downcast). For fp32
    # the add runs on the SFPU with both operand CBs unpacked straight to fp32 DEST (the FPU's
    # SrcA/SrcB would truncate to tf32). fp32_dest_acc_en stays hard-wired (spec: fp32 accumulate).
    fp32_path = dtype == ttnn.float32
    compute_config = ttnn.ComputeConfigDescriptor(fp32_dest_acc_en=True)
    if fp32_path:
        modes = [ttnn.UnpackToDestMode.Default] * NUM_CIRCULAR_BUFFERS
        modes[CB_REMOTE_PARTIAL] = ttnn.UnpackToDestMode.UnpackToDestFp32
        modes[CB_LOCAL_INPUT] = ttnn.UnpackToDestMode.UnpackToDestFp32
        compute_config.unpack_to_dest_mode = modes
    tile_bytes = input_tensor.tile.get_tile_size(dtype)
    max_payload = ttnn.get_tt_fabric_max_payload_size_bytes()
    packet_tiles = max_payload // tile_bytes
    assert packet_tiles >= 1, "fabric payload smaller than one tile"
    chunk_tiles = max(packet_tiles, ((CHUNK_BYTES_TARGET // tile_bytes) // packet_tiles) * packet_tiles)
    chunk_bytes = chunk_tiles * tile_bytes

    padded = list(input_tensor.padded_shape)
    lead = 1
    for d in padded[:-2]:
        lead *= d
    tensor_tiles = lead * _div_up(padded[-2], 32) * _div_up(padded[-1], 32)

    lane_nominal = _div_up(tensor_tiles, num_lanes)
    lane_start = [l * lane_nominal for l in range(num_lanes)]
    lane_tiles = [max(0, min(tensor_tiles - s, lane_nominal)) for s in lane_start]
    lane_blocks = [_div_up(t, chunk_tiles) for t in lane_tiles]
    W = max(1, min(REDUCERS_PER_LANE, max(lane_blocks)))

    def reducer_blocks(l, r):
        n = lane_blocks[l]
        return _div_up(n - r, W) if n > r else 0

    # --- core layout (identical on every device; port position fixed per lane) ---
    grid = mesh_device.compute_with_storage_grid_size()
    split = _split_ports(group_size)
    if num_lanes * _lane_stride(split) > grid.x * grid.y:
        raise ValueError("high_bw_all_reduce: compute grid too small for the requested lanes")
    fwd_cores = [_lane_core(l, 0, grid, split) for l in range(num_lanes)]
    bwd_cores = [_lane_core(l, split, grid, split) for l in range(num_lanes)]  # == fwd_cores if co-located
    reducer_cores = [[_lane_core(l, 1 + split + r, grid, split) for r in range(W)] for l in range(num_lanes)]
    all_reducers = [c for lane in reducer_cores for c in lane]
    port_cores = fwd_cores + (bwd_cores if split else [])
    all_cores = port_cores + all_reducers
    fwd_set, bwd_set = _core_set(fwd_cores), _core_set(bwd_cores)
    reducer_set, all_set = _core_set(all_reducers), _core_set(all_cores)

    def noc_xy(core):
        v = mesh_device.worker_core_from_logical_core(core)
        return v.x, v.y

    # Persistent cross-device counters first: created before the per-call scratch, their (tiny,
    # cached) L1 words never split the free block the scratch below needs.
    gsem = _global_semaphores(mesh_device, cluster_axis, "ring" if is_ring else "line", num_lanes, grid, split)

    # --- per-call L1 scratch (uniform address on every device) ---
    # ONE lockstep L1 tensor sharded over every op core (Refinement 3: "re-place the port scratch").
    # An L1 allocation reserves its address range on every core, so separate reducer and port
    # scratch tensors (plus the reducers' program CBs) cost their SUM on every core; with a shared
    # shard the reducers' three CBs and the port's control array + rings overlay one address range
    # and the footprint is their MAX. That is what lets the credit-batch-sized rings fit.
    control_bytes = _div_up(NUM_CONTROL_ARRAYS * W * CONTROL_WORD_STRIDE, tile_bytes) * tile_bytes

    def scratch_layout(batch):
        recv_depth, final_depth = _ring_depths(batch)
        reducer = [recv_depth * chunk_tiles, INPUT_DEPTH_CHUNKS * chunk_tiles, REDUCED_DEPTH_CHUNKS * chunk_tiles]
        port = [control_bytes // tile_bytes, W * STAGING_DEPTH_PER_REDUCER * chunk_tiles, W * final_depth * chunk_tiles]
        return recv_depth, final_depth, reducer, port, max(sum(reducer), sum(port))

    # The deepest rings (largest credit batch) the gate allows that fit the largest free L1 block
    # (lockstep over the mesh, so every device derives the same batch); batch 1 is the Phase 0
    # footprint.
    l1_free = ttnn.get_memory_view(mesh_device, ttnn.BufferType.L1).largest_contiguous_bytes_free_per_bank
    credit_batch = _credit_batch(group_size, num_lanes)
    while credit_batch > 1 and scratch_layout(credit_batch)[-1] * tile_bytes > l1_free:
        credit_batch -= 1
    recv_depth, final_depth, (recv_pages, input_pages, reduced_pages), port_layout, shard_pages = scratch_layout(
        credit_batch
    )
    staging_pages = port_layout[1]
    op_scratch = _height_sharded_l1(mesh_device, all_cores, shard_pages, dtype)

    output_tensor = ttnn.allocate_tensor_on_device(input_tensor.spec, mesh_device)

    scratch_addr = int(op_scratch.buffer_address())
    landing_addr = scratch_addr  # reducer cores: cb_remote_partial at shard offset 0
    ctrl_base = scratch_addr  # port cores: control array at shard offset 0
    staged_arr = ctrl_base
    final_ready_arr = ctrl_base + W * CONTROL_WORD_STRIDE
    granted_arr = ctrl_base + 2 * W * CONTROL_WORD_STRIDE
    final_freed_arr = ctrl_base + 3 * W * CONTROL_WORD_STRIDE
    drained_arr = ctrl_base + 4 * W * CONTROL_WORD_STRIDE
    staging_addr = ctrl_base + control_bytes
    final_addr = staging_addr + staging_pages * tile_bytes

    per_reducer = {kind: [gsem[f"{kind}_{r}"] for r in range(W)] for kind in PER_REDUCER_GSEMS}

    input_addr = int(input_tensor.buffer_address())
    output_addr = int(output_tensor.buffer_address())
    input_ta = ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args()
    output_ta = ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()

    mesh_desc = ttnn.MeshProgramDescriptor()
    rows, cols = mesh_shape
    for mr in range(rows):
        for mc in range(cols):
            coord = (mr, mc)
            path, p = placement[coord]
            G = len(path)
            has_next = is_ring or p < G - 1
            has_prev = is_ring or p > 0
            me = _node(mesh_device, coord)
            nxt = _node(mesh_device, path[(p + 1) % G]) if has_next else None
            prv = _node(mesh_device, path[(p - 1) % G]) if has_prev else None
            roles_ct = [p, G, num_slices]
            fwd_links = ttnn.get_forwarding_link_indices(me, nxt) if nxt is not None else []
            bwd_links = ttnn.get_forwarding_link_indices(me, prv) if prv is not None else []

            program = ttnn.ProgramDescriptor()
            program.semaphores = [
                ttnn.SemaphoreDescriptor(id=SEM_GO, core_ranges=all_set, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_EGRESS_CREDIT, core_ranges=all_set, initial_value=0),
                ttnn.SemaphoreDescriptor(id=SEM_FINAL_EGRESS, core_ranges=all_set, initial_value=0),
            ]
            program.cbs = [
                _scratch_cb(CB_REMOTE_PARTIAL, op_scratch, 0, recv_pages, tile_bytes, reducer_set),
                _scratch_cb(CB_LOCAL_INPUT, op_scratch, recv_pages, input_pages, tile_bytes, reducer_set),
                _scratch_cb(CB_REDUCED, op_scratch, recv_pages + input_pages, reduced_pages, tile_bytes, reducer_set),
            ]

            reader_rt = ttnn.RuntimeArgs()
            writer_rt = ttnn.RuntimeArgs()
            compute_rt = ttnn.RuntimeArgs()
            fwd_rt = ttnn.RuntimeArgs()
            bwd_rt = ttnn.RuntimeArgs()
            drain_rt = ttnn.RuntimeArgs()

            for l in range(num_lanes):
                fwd_core, bwd_core = fwd_cores[l], bwd_cores[l]
                fx, fy = noc_xy(fwd_core)
                bx, by = noc_xy(bwd_core)
                red_xy = []
                for r, core in enumerate(reducer_cores[l]):
                    rx, ry = noc_xy(core)
                    red_xy += [rx, ry]
                    nb = reducer_blocks(l, r)
                    reader_rt[core.x][core.y] = [
                        input_addr,
                        lane_start[l],
                        lane_tiles[l],
                        r,
                        W,
                        nb,
                        bx,
                        by,
                        granted_arr + r * CONTROL_WORD_STRIDE,
                        gsem["partial_arrival"],
                    ]
                    writer_rt[core.x][core.y] = [
                        r,
                        W,
                        nb,
                        bx,
                        by,
                        fx,
                        fy,
                        staging_addr,
                        staged_arr + r * CONTROL_WORD_STRIDE,
                        final_addr,
                        final_ready_arr + r * CONTROL_WORD_STRIDE,
                        output_addr,
                        lane_start[l],
                        lane_tiles[l],
                    ]
                    compute_rt[core.x][core.y] = [nb, r, W]

                fwd_args = (
                    [
                        lane_blocks[l],
                        W,
                        fx,
                        fy,
                        bx,
                        by,
                        staging_addr,
                        staged_arr,
                        landing_addr,
                        gsem["partial_arrival"],
                        final_freed_arr,
                        int(nxt.mesh_id) if nxt is not None else 0,
                        int(nxt.chip_id) if nxt is not None else 0,
                    ]
                    + red_xy
                    + per_reducer["partial_credit"]
                    + per_reducer["final_credit"]
                )
                if nxt is not None:
                    fwd_args += list(ttnn.setup_fabric_connection(me, nxt, fwd_links[l], program, fwd_core))
                fwd_rt[fwd_core.x][fwd_core.y] = fwd_args

                bwd_args = (
                    [
                        lane_blocks[l],
                        W,
                        bx,
                        by,
                        fx,
                        fy,
                        drained_arr,
                        ctrl_base,
                        final_addr,
                        final_ready_arr,
                        granted_arr,
                        final_freed_arr,
                        output_addr,
                        lane_start[l],
                        lane_tiles[l],
                        int(prv.mesh_id) if prv is not None else 0,
                        int(prv.chip_id) if prv is not None else 0,
                    ]
                    + red_xy
                    + per_reducer["partial_credit"]
                    + per_reducer["final_arrival"]
                    + per_reducer["final_credit"]
                )
                if prv is not None:
                    bwd_args += list(ttnn.setup_fabric_connection(me, prv, bwd_links[l], program, bwd_core))
                bwd_rt[bwd_core.x][bwd_core.y] = bwd_args
                drain_rt[bwd_core.x][bwd_core.y] = [
                    lane_blocks[l],
                    W,
                    final_addr,
                    drained_arr,
                    output_addr,
                    lane_start[l],
                    lane_tiles[l],
                ] + per_reducer["final_arrival"]

            reader_ct = (
                [
                    CB_REMOTE_PARTIAL,
                    CB_LOCAL_INPUT,
                    chunk_tiles,
                    recv_depth,
                    SEM_GO,
                    tile_bytes,
                ]
                + roles_ct
                + [BANK_RUN_LAYOUT]
                + list(input_ta)
            )
            writer_ct = (
                [
                    CB_REDUCED,
                    chunk_tiles,
                    tile_bytes,
                    STAGING_DEPTH_PER_REDUCER,
                    final_depth,
                    SEM_GO,
                    SEM_EGRESS_CREDIT,
                    SEM_FINAL_EGRESS,
                ]
                + roles_ct
                + [BANK_RUN_LAYOUT]
                + list(output_ta)
            )
            compute_ct = [CB_REMOTE_PARTIAL, CB_LOCAL_INPUT, CB_REDUCED, chunk_tiles] + roles_ct + [int(fp32_path)]
            fwd_ct = (
                [
                    chunk_tiles,
                    packet_tiles,
                    tile_bytes,
                    STAGING_DEPTH_PER_REDUCER,
                    recv_depth,
                    final_depth,
                    int(has_next),
                    SEM_GO,
                    SEM_EGRESS_CREDIT,
                    CONTROL_WORD_STRIDE,
                    MAX_DATA_HEADERS,
                ]
                + roles_ct
                + [REDUCERS_PER_LANE, credit_batch]
            )
            bwd_ct = (
                [
                    chunk_tiles,
                    packet_tiles,
                    tile_bytes,
                    final_depth,
                    int(has_prev),
                    SEM_GO,
                    SEM_FINAL_EGRESS,
                    NUM_CONTROL_ARRAYS,
                    CONTROL_WORD_STRIDE,
                    MAX_DATA_HEADERS,
                ]
                + roles_ct
                + [REDUCERS_PER_LANE, credit_batch, BANK_RUN_LAYOUT, split]
                + list(output_ta)
            )

            kernels = [
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "high_bw_all_reduce_reader.cpp"),
                    core_ranges=reducer_set,
                    compile_time_args=reader_ct,
                    runtime_args=reader_rt,
                    # DRAM reads on NoC0 (preferred read NoC); staging writes to the port on NoC1,
                    # off the NoC0 the port's port_fwd streams Fabric payload on.
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=REDUCER_READER_NOC
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "high_bw_all_reduce_writer.cpp"),
                    core_ranges=reducer_set,
                    compile_time_args=writer_ct,
                    runtime_args=writer_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=REDUCER_WRITER_NOC
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "high_bw_all_reduce_compute.cpp"),
                    core_ranges=reducer_set,
                    compile_time_args=compute_ct,
                    runtime_args=compute_rt,
                    config=compute_config,
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "high_bw_all_reduce_port_fwd.cpp"),
                    core_ranges=fwd_set,
                    compile_time_args=fwd_ct,
                    runtime_args=fwd_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_0, noc=PORT_FWD_NOC
                    ),
                ),
                ttnn.KernelDescriptor(
                    kernel_source=str(KERNEL_DIR / "high_bw_all_reduce_port_bwd.cpp"),
                    core_ranges=bwd_set,
                    compile_time_args=bwd_ct,
                    runtime_args=bwd_rt,
                    config=ttnn.DataMovementConfigDescriptor(
                        processor=ttnn.DataMovementProcessor.RISCV_1, noc=PORT_BWD_NOC
                    ),
                ),
            ]
            if split:
                drain_ct = [
                    chunk_tiles,
                    tile_bytes,
                    final_depth,
                    SEM_GO,
                    CONTROL_WORD_STRIDE,
                    *roles_ct,
                    REDUCERS_PER_LANE,
                    BANK_RUN_LAYOUT,
                ] + list(output_ta)
                kernels.append(
                    ttnn.KernelDescriptor(
                        kernel_source=str(KERNEL_DIR / "high_bw_all_reduce_port_drain.cpp"),
                        core_ranges=bwd_set,
                        compile_time_args=drain_ct,
                        runtime_args=drain_rt,
                        config=ttnn.DataMovementConfigDescriptor(
                            processor=ttnn.DataMovementProcessor.RISCV_0, noc=PORT_DRAIN_NOC
                        ),
                    )
                )
            program.kernels = kernels
            mesh_desc[ttnn.MeshCoordinateRange(ttnn.MeshCoordinate(*coord), ttnn.MeshCoordinate(*coord))] = program

    return ttnn.generic_op([input_tensor, op_scratch, output_tensor], mesh_desc)
