# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for paged_scaled_dot_product_attention_decode (op_design.md, this directory).

The scheme is a parameterized flash-decode. For one user its key/value heads' position tiles are laid end to end
as one *virtual key sequence* of N = NKV * user_pos_tiles units (unit v = head v // user_pos_tiles, position tile
v % user_pos_tiles). A unit of head c adds a large negative number to the scores of every query row outside head
group c, so the heads share one output tile. The sequence is split over the workers of a cluster group (a worker
is one NEO of one cluster): worker w runs an online softmax over a contiguous range and emits a partial
(O unnormalized, row max m, row sum l) into an L1 scratch tensor; the group's rank-0 cluster (the combiner)
waits on a semaphore for every partial, merges them, each of its NEOs taking a share of the head dimension, and
writes the result.

user_pos_tiles depends on cur_pos, a device tensor the host does not read (rule R6), so the split is evaluated on
the device by kernels/decode_plan.hpp (one function shared by the three kernels). The host sizes everything for
the largest position, N_max = NKV * tensor_pos_tiles, with the same worker-count rule (`clusters_for`, mirrored
from decode_plan.hpp).

Every knob is defined once below and passed to the kernels as a compile-time argument; every ring capacity, loop
trip count and grid size is derived from them.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# ---------------------------------------------------------------------------------------------------------------
# Knobs (op_design.md, "Every knob is a parameter")
# ---------------------------------------------------------------------------------------------------------------

# (R reader harts, C NEOs, W writer harts) per cluster: compute-bound (two matmuls, softmax, merges per unit).
THREAD_PLAN = (4, 4, 2)
# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
NEOS_PER_CLUSTER = 4
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 is the remapper
# L1 kept free of this op's buffers on every cluster (firmware, other programs' allocations).
L1_RESERVE_BYTES = 65536
# The cost of one partial merge in units of one position tile of work (the sqrt(N) worker rule).
MERGE_COST_UNITS = 1
# K/V ring depth in position tile-rows (each tile-row is tensor_d_tiles tiles).
KV_DEPTH_ROWS = 2
# Score-mask ring: masks per head segment (body, last) and segments in flight.
MASKS_PER_SEGMENT = 2
MASK_DEPTH_SEGMENTS = 2
# Partials-in ring depth in partials.
PARTIAL_DEPTH_PARTIALS = 2
# Streaming windows (q in, output out) and the running-state rings (pop then push one tile).
STREAM_DEPTH = 2
STATE_DEPTH = 2
# Users per block: no operand is shared across users, so a multi-user block amortizes nothing (design, Axes).
USERS_PER_BLOCK = 1
# Partials merged per step of the combine.
PARTIALS_PER_MERGE = 1
# Where the partials scratch lives; DRAM is the fallback when its L1 share does not fit.
PARTIALS_MEMORY = ttnn.L1_MEMORY_CONFIG
# Test overrides: the most clusters one user may engage (None: the device's grid), the thread plan, and a hook
# that keeps the last partials scratch tensor so a test can see which worker slots were written.
_MAX_CLUSTERS_PER_USER = None
_THREAD_PLAN_OVERRIDE = None
_DEBUG_KEEP_SCRATCH = False
_SCRATCH_SENTINEL = 12345.0
_last_scratch = None

# DEST tiles of one half-sync section (kernel_lib dest_helpers.hpp DEST_AUTO_LIMIT; the compute kernel asserts the
# PV sub-block against it): the PV output sub-block, whose width the reader's V order follows.
DEST_TILES = {True: 4, False: 8}

# The compute kernel is built for size: its unpack binary at O2 exceeds the 24 KiB compute limit (target page).
COMPUTE_OPT_LEVEL = "Os"

# Rings a NEO's compute kernel both produces and consumes: 16 Tensix-only tile counters, two per ring.
MAX_INTRA_NEO_RINGS = 8

TILE = 32
NUM_SEMAPHORES = NEOS_PER_CLUSTER  # one "partials ready" semaphore per combiner reader hart


def validate_thread_plan(threads):
    r, c, w = threads
    ok = (
        c in (1, 2, 4)
        and c <= NEOS_PER_CLUSTER
        and r >= 1
        and w >= 1
        and r + w <= FREE_DM_HARTS
        and c % r == 0
        and c % w == 0
        and r == c  # each reader hart feeds exactly one NEO: per-NEO reader streams differ in length
    )
    if not ok:
        raise ValueError(
            f"thread plan (R, C, W) = {threads}: need C in {{1, 2, 4}}, R + W <= {FREE_DM_HARTS}, R | C, W | C, R == C"
        )


def isqrt_ceil(x):
    r = math.isqrt(x)
    return r if r * r == x else r + 1


def clusters_for(units, compute_threads, cluster_cap, merge_cost_units):
    """Clusters that minimize N/W + kappa*W for N units on W = clusters * C workers (decode_plan.hpp mirrors it)."""
    workers = isqrt_ceil(-(-units // merge_cost_units))
    clusters = -(-workers // compute_threads)
    return max(1, min(clusters, cluster_cap))


def pv_subblock_tiles(tensor_d_tiles, fp32_dest_acc_en):
    """The PV output sub-block: the largest divisor of tensor_d_tiles that fits one DEST section. A divisor, so
    the reader's V blocks never wrap the V ring (its capacity is a multiple of tensor_d_tiles)."""
    cap = DEST_TILES[bool(fp32_dest_acc_en)]
    return max(d for d in range(1, min(cap, tensor_d_tiles) + 1) if tensor_d_tiles % d == 0)


def _tile_bytes(dtype):
    return {ttnn.bfloat16: 2048, ttnn.float32: 4096, ttnn.int32: 4096}[dtype]


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _rings(C, io_dtype, inter_dtype, tensor_d_tiles, d_share_tiles, k_chunk_tiles):
    """(name, entries per NEO, dtype) of every ring: the single list the specs and the L1 footprint come from."""
    return [
        ("cur_pos", 1, ttnn.int32),
        ("reduce_scaler", 1, ttnn.bfloat16),
        ("q_tiles", tensor_d_tiles, io_dtype),  # one block: the reader writes all of q at once
        ("k_tiles", KV_DEPTH_ROWS * tensor_d_tiles, io_dtype),
        ("v_tiles", KV_DEPTH_ROWS * tensor_d_tiles, io_dtype),
        ("score_masks", MASK_DEPTH_SEGMENTS * MASKS_PER_SEGMENT, inter_dtype),
        ("partials_in", PARTIAL_DEPTH_PARTIALS * (2 + d_share_tiles), inter_dtype),
        ("q_scaled", tensor_d_tiles, io_dtype),
        ("scores", k_chunk_tiles, inter_dtype),
        ("probs", k_chunk_tiles, io_dtype),
        ("max_run", STATE_DEPTH, inter_dtype),
        # One column-vector tile in flight per step, by role: m_chunk, then alpha (phase 1); [beta, alpha]
        # (phase 2); 1/l (normalize).
        ("col_step", STATE_DEPTH, inter_dtype),
        ("sum_run", STATE_DEPTH, inter_dtype),
        ("out_acc", tensor_d_tiles, inter_dtype),
        # One step's [O_step, l_step]: [O_chunk, l_chunk] (phase 1) or [beta O_k share, beta l_k] (phase 2).
        ("step", tensor_d_tiles + 1, inter_dtype),
        ("partial_out", tensor_d_tiles + 2, inter_dtype),
        ("output_tiles", STREAM_DEPTH, io_dtype),
    ]


def _ring_bytes(rings, C):
    return C * sum(entries * _tile_bytes(dtype) for _, entries, dtype in rings)


def _align(x, a):
    return -(-x // a) * a


def _group_assignment(grid, users, cluster_cap):
    """(G_u, {core: (user_start, num_users, rank, combiner core)}) with engaged clusters taken row-wise."""
    num_clusters = grid.x * grid.y
    cores = [ttnn.CoreCoord(x, y) for y in range(grid.y) for x in range(grid.x)]
    clusters_per_user = max(1, min(cluster_cap, num_clusters // users if users <= num_clusters else 1))
    assignment = {}
    if users * clusters_per_user <= num_clusters:
        for user in range(users):
            group = cores[user * clusters_per_user : (user + 1) * clusters_per_user]
            for rank, core in enumerate(group):
                assignment[core] = (user, 1, rank, group[0])
        return clusters_per_user, assignment
    # More users than clusters: one cluster per user, users dealt in contiguous rounds.
    _, _, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, users, row_wise=True)
    start = 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            assignment[core] = (start, per_core, 0, core)
            start += per_core
    return 1, assignment


def create_program_artifacts(
    q, k, v, page_table, cur_pos, output, *, scale, math_fidelity, fp32_dest_acc_en, math_approx_mode
):
    global _last_scratch
    threads = _THREAD_PLAN_OVERRIDE or THREAD_PLAN
    validate_thread_plan(threads)
    R, C, W = threads
    device = q.device()
    grid = device.compute_with_storage_grid_size()
    num_clusters = grid.x * grid.y

    # Geometry (alignment-aware; X9 makes block_size a multiple of 32).
    users, heads, head_dim = q.shape[1], q.shape[2], q.shape[3]
    num_blocks, kv_heads, block_size = k.shape[0], k.shape[1], k.shape[2]
    pages_per_user = page_table.shape[1]
    tensor_d_tiles = -(-head_dim // TILE)
    block_pos_tiles = block_size // TILE
    tensor_pos_tiles = pages_per_user * block_pos_tiles
    has_row_groups = int(kv_heads > 1 or heads < TILE)
    io_dtype = q.dtype
    inter_dtype = ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16

    # Work distribution: the worker-count rule at the largest position, capped by the grid.
    units_max = kv_heads * tensor_pos_tiles
    cap = num_clusters if _MAX_CLUSTERS_PER_USER is None else max(1, min(num_clusters, _MAX_CLUSTERS_PER_USER))
    cap = min(cap, max(1, num_clusters // users))
    clusters_per_user = clusters_for(units_max, C, cap, MERGE_COST_UNITS)
    clusters_per_user, assignment = _group_assignment(grid, users, clusters_per_user)
    max_workers_per_user = clusters_per_user * C
    d_share_tiles = -(-tensor_d_tiles // C)

    # Scratchpads: each DM hart reads the positions (and a reader hart its user's page-table row) itself.
    pos_page_bytes = cur_pos.buffer_aligned_page_size()
    table_page_bytes = page_table.buffer_aligned_page_size()
    reader_hart_scratch = _align(pos_page_bytes, 64) + _align(table_page_bytes, 64)
    writer_hart_scratch = _align(pos_page_bytes, 64)

    # The partials scratch: slot u*max_workers_per_user + w holds worker w's (O tiles, m, l) of user u.
    partial_tiles = tensor_d_tiles + 2
    partials_shape = ttnn.Shape([1, 1, users * max_workers_per_user * TILE, partial_tiles * TILE])
    partials_bytes = users * max_workers_per_user * partial_tiles * _tile_bytes(inter_dtype)

    # k_chunk_tiles: one chunk = the whole per-worker assignment at N_max, lowered until the cluster's L1 fits.
    budget = ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES
    partials_memory = PARTIALS_MEMORY
    dm_scratch = R * reader_hart_scratch + W * writer_hart_scratch
    k_chunk_tiles = max(1, min(tensor_pos_tiles, -(-units_max // max_workers_per_user)))

    def footprint(chunk, memory_config):
        share = -(-partials_bytes // num_clusters) if memory_config == ttnn.L1_MEMORY_CONFIG else 0
        rings = _rings(C, io_dtype, inter_dtype, tensor_d_tiles, d_share_tiles, chunk)
        return _ring_bytes(rings, C) + dm_scratch + share

    if footprint(1, partials_memory) > budget:
        partials_memory = ttnn.DRAM_MEMORY_CONFIG
    while k_chunk_tiles > 1 and footprint(k_chunk_tiles, partials_memory) > budget:
        k_chunk_tiles -= 1
    if footprint(k_chunk_tiles, partials_memory) > budget:
        raise NotImplementedError(
            "paged_scaled_dot_product_attention_decode: the head dimension is too wide for one cluster's L1 "
            f"(D={head_dim}): the wide-D regime (O state blocked along D) is not built"
        )

    if _DEBUG_KEEP_SCRATCH:
        # Test hook: the scratch pre-filled with a sentinel (a host-to-device write, not a program), kept so a test
        # can see which worker slots the program wrote.
        import torch

        partials = ttnn.from_torch(
            torch.full(tuple(partials_shape), _SCRATCH_SENTINEL, dtype=torch.float32),
            dtype=inter_dtype,
            layout=ttnn.TILE_LAYOUT,
            device=device,
            memory_config=partials_memory,
        )
        _last_scratch = partials
    else:
        partials = ttnn.allocate_tensor_on_device(
            ttnn.TensorSpec(partials_shape, inter_dtype, ttnn.TILE_LAYOUT, partials_memory), device
        )

    rings = _rings(C, io_dtype, inter_dtype, tensor_d_tiles, d_share_tiles, k_chunk_tiles)
    dataflow_buffers = [
        E.DataflowBufferSpec(
            unique_id=name, entry_size=_tile_bytes(dtype), num_entries=entries * C, data_format_metadata=dtype
        )
        for name, entries, dtype in rings
    ]

    core_set = ttnn.CoreRangeSet([ttnn.CoreRange(core, core) for core in assignment])
    semaphores = [E.SemaphoreSpec(unique_id=f"partials_ready_{i}", target_nodes=core_set) for i in range(NUM_SEMAPHORES)]
    semaphore_bindings = [
        E.SemaphoreBinding(semaphore_spec_name=f"partials_ready_{i}", accessor_name=f"partials_ready_{i}")
        for i in range(NUM_SEMAPHORES)
    ]
    scratchpads = [
        E.ScratchpadSpec(unique_id="reader_scratch", size_per_node=R * reader_hart_scratch),
        E.ScratchpadSpec(unique_id="writer_scratch", size_per_node=W * writer_hart_scratch),
    ]

    tensors = {
        "q": q,
        "k": k,
        "v": v,
        "page_table": page_table,
        "cur_pos": cur_pos,
        "partials": partials,
        "output": output,
    }
    tensor_parameters = [E.TensorParameter(unique_id=name, spec=t.spec) for name, t in tensors.items()]

    def bind(*names):
        return [E.TensorBinding(tensor_parameter_name=n, accessor_name=n) for n in names]

    plan = {
        "reader_threads": R,
        "compute_threads": C,
        "writer_threads": W,
        "num_users": users,
        "num_heads": heads,
        "kv_heads": kv_heads,
        "tensor_d_tiles": tensor_d_tiles,
        "d_share_tiles": d_share_tiles,
        "block_pos_tiles": block_pos_tiles,
        "pages_per_user": pages_per_user,
        "tensor_pos_tiles": tensor_pos_tiles,
        "num_blocks": num_blocks,
        "clusters_per_user": clusters_per_user,
        "max_workers_per_user": max_workers_per_user,
        "k_chunk_tiles": k_chunk_tiles,
        "merge_cost_units": MERGE_COST_UNITS,
        "has_row_groups": has_row_groups,
        "users_per_block": USERS_PER_BLOCK,
        "partials_per_merge": PARTIALS_PER_MERGE,
        "pv_subblock_tiles": pv_subblock_tiles(tensor_d_tiles, fp32_dest_acc_en),
        "inter_fp32": int(bool(fp32_dest_acc_en)),
    }
    dm_plan = {
        **plan,
        "pos_page_bytes": pos_page_bytes,
        "table_page_bytes": table_page_bytes,
        "reader_hart_scratch": reader_hart_scratch,
        "writer_hart_scratch": writer_hart_scratch,
    }
    work_schema = ["user_start", "user_count", "cluster_rank"]

    # Explicit sync on every reader ring: the reader harts' streams differ in length (reader kernel header).
    reader_hw = E.create_reader_datamovement_config()
    reader_hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for_all=True)
    writer_hw = E.create_writer_datamovement_config()
    writer_hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for_all=True)

    reader_rings = ["cur_pos", "reduce_scaler", "q_tiles", "k_tiles", "v_tiles", "score_masks", "partials_in"]
    compute_owned = [
        "q_scaled", "scores", "probs", "max_run", "col_step", "sum_run", "out_acc", "step",
    ]  # fmt: skip
    # A ring the compute kernel both produces and consumes costs two of a NEO's 16 Tensix-only tile counters
    # (dataflow_buffer_config.h TILE_COUNTERS_PER_INTRA_TENSIX_DFB): at most 8 per NEO.
    assert len(compute_owned) <= MAX_INTRA_NEO_RINGS, compute_owned
    writer_rings = ["partial_out", "output_tiles"]

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "paged_scaled_dot_product_attention_decode_reader.cpp"),
        hw_config=reader_hw,
        num_threads=R,
        dfb_bindings=[E.producer_of(n, n) for n in reader_rings],
        semaphore_bindings=semaphore_bindings,
        scratchpad_bindings=[E.ScratchpadBinding(scratchpad_spec_name="reader_scratch", accessor_name="scratch")],
        tensor_bindings=bind("q", "k", "v", "page_table", "cur_pos", "partials"),
        compile_time_args=dm_plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=work_schema),
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "paged_scaled_dot_product_attention_decode_compute.cpp"),
        hw_config=E.ComputeHardwareConfig(
            fpu_math_fidelity=math_fidelity,  # always set
            enable_32_bit_dest=fp32_dest_acc_en,
            # Every Float32 ring the compute kernel reads goes through the source registers (FPU ops read them);
            # UnpackToDest is not used (its reload and reconfig paths are not proven here).
            unpack_modes={
                name: E.UnpackMode.UnpackToSrc
                for name, _, dtype in rings
                if dtype == ttnn.float32 and name not in writer_rings
            },
        ),
        num_threads=C,
        # The fused kernel (two matmuls, softmax, two merges) exceeds the 24 KiB compute binary at O2.
        compiler_options=E.KernelSpec.CompilerOptions(opt_level=getattr(E.KernelBuildOptLevel, COMPUTE_OPT_LEVEL)),
        dfb_bindings=(
            [E.consumer_of(n, n) for n in reader_rings]
            + [b for n in compute_owned for b in (E.producer_of(n, n), E.consumer_of(n, n))]
            + [E.producer_of(n, n) for n in writer_rings]
        ),
        compile_time_args={**plan, "exp_approx": int(math_approx_mode)},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=work_schema, common_runtime_arg_names=["scale_bits"]
        ),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "paged_scaled_dot_product_attention_decode_writer.cpp"),
        hw_config=writer_hw,
        num_threads=W,
        dfb_bindings=[E.consumer_of(n, n) for n in writer_rings],
        semaphore_bindings=semaphore_bindings,
        scratchpad_bindings=[E.ScratchpadBinding(scratchpad_spec_name="writer_scratch", accessor_name="scratch")],
        tensor_bindings=bind("cur_pos", "partials", "output"),
        compile_time_args=dm_plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=work_schema + ["combiner_x", "combiner_y"]
        ),
    )

    spec = E.ProgramSpec(
        name="paged_scaled_dot_product_attention_decode",
        kernels=[reader, compute, writer],
        dataflow_buffers=dataflow_buffers,
        semaphores=semaphores,
        scratchpads=scratchpads,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )

    work_args = {
        "user_start": {core: a[0] for core, a in assignment.items()},
        "user_count": {core: a[1] for core, a in assignment.items()},
        "cluster_rank": {core: a[2] for core, a in assignment.items()},
    }
    combiner_noc = {core: device.worker_core_from_logical_core(a[3]) for core, a in assignment.items()}
    writer_args = {
        **work_args,
        "combiner_x": {core: xy.x for core, xy in combiner_noc.items()},
        "combiner_y": {core: xy.y for core, xy in combiner_noc.items()},
    }
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=work_args),
            E.KernelRunArgs(
                kernel="compute",
                runtime_arg_values=work_args,
                common_runtime_arg_values={"scale_bits": _float_bits(scale)},
            ),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=writer_args),
        ]
    )
    io_tensors = list(tensors.values())
    tensor_indices = {name: i for i, name in enumerate(tensors)}
    return io_tensors, spec, run_args, tensor_indices


def program_summary(q, k, page_table, *, fp32_dest_acc_en=True, threads=None, grid=None):
    """Structural counts of the plan for a call (for tests and the changelog): clusters per user, workers,
    k_chunk_tiles. Uses the same functions as create_program_artifacts."""
    R, C, W = threads or _THREAD_PLAN_OVERRIDE or THREAD_PLAN
    grid = grid or q.device().compute_with_storage_grid_size()
    num_clusters = grid.x * grid.y
    users = q.shape[1]
    kv_heads, block_pos_tiles = k.shape[1], k.shape[2] // TILE
    tensor_pos_tiles = page_table.shape[1] * block_pos_tiles
    units_max = kv_heads * tensor_pos_tiles
    cap = num_clusters if _MAX_CLUSTERS_PER_USER is None else max(1, min(num_clusters, _MAX_CLUSTERS_PER_USER))
    cap = min(cap, max(1, num_clusters // users))
    g = clusters_for(units_max, C, cap, MERGE_COST_UNITS)
    return {"clusters_per_user": g, "max_workers_per_user": g * C, "units_max": units_max}
