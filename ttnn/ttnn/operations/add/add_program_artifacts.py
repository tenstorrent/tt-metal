# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for add (regime `stream_flat` of op_design.md).

Phase 0: a, b and out have identical tile-padded page sequences, so page p of out = page p of a + page p of b.
The flat page axis is split in contiguous ranges over the full grid (rounds of C pages per cluster); inside a
cluster, the STRIDED rings deal the range round-robin over the thread plan (R readers, C NEOs, W writers):

    reader hart r   reads    pages r, r+R, ...   (a then b, one tagged implicit-sync read per page each)
    NEO t           adds     pages t, t+C, ...   (one eltwise `add` helper call, blocks of `block_tiles`)
    writer hart w   writes   pages w, w+W, ...   (one tagged implicit-sync write per page)

Knobs (each defined once here; every ring size, loop count and grid size derives from them):
    THREAD_PLAN         (R, C, W)
    BLOCK_TILES         tiles a NEO takes from the rings per block (one DEST section)
    RING_DEPTH_BLOCKS   blocks of capacity per NEO in each ring
    MIN_TILES_PER_CORE  smallest page count worth engaging a cluster for
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# --- Knobs (op_design.md, Parameters / Blocking Model) ---------------------------------------------------
THREAD_PLAN = (1, 1, 1)  # (R, C, W): data-movement bound, one FPU op per tile (op_design.md, Work Distribution)
BLOCK_TILES = 4  # one DEST section in every DEST mode (fp32 half-sync: 4)
RING_DEPTH_BLOCKS = 2  # double buffering: read block k+1 while adding block k
MIN_TILES_PER_CORE = 1  # occupancy lamp: engage ceil(num_tiles / MIN_TILES_PER_CORE) clusters at most

# --- Cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them ---
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 is the remapper
NEOS = 4
MAX_RING_BYTES = 1_048_560  # one DFB ring, compute side

# Semantic ring names (shared with the kernels as dfb::<name>).
CB_INPUT_A = "cb_input_a_tiles"
CB_INPUT_B = "cb_input_b_tiles"
CB_OUTPUT = "cb_output_tiles"


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in (1, 2, 4) and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(
            f"thread plan (R, C, W) = {threads}: need C in {{1, 2, 4}}, R + W <= {FREE_DM_HARTS}, R | C, W | C"
        )


def split_pages(num_tiles, grid_size, compute_threads, min_tiles_per_core=MIN_TILES_PER_CORE):
    """(core set, {core: pages}, {core: first page}) over the clusters that get work.

    The split is in rounds of C pages (whole rounds per cluster, so each NEO's share differs by at most one
    page), converted back to pages; the last cluster's last round may be short.
    """
    rounds = -(-num_tiles // compute_threads)
    min_rounds = max(1, -(-min_tiles_per_core // compute_threads))
    max_cores = grid_size.x * grid_size.y
    num_cores = min(max_cores, -(-rounds // min_rounds))
    cores = ttnn.num_cores_to_corerangeset(num_cores, grid_size, True)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(cores, rounds, row_wise=True)
    counts, starts, start_round = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            first = start_round * compute_threads
            counts[core] = min(per_core * compute_threads, num_tiles - first)
            starts[core] = first
            start_round += per_core
    return all_cores, counts, starts


def dest_section_tiles(config):
    """Tiles in one DEST section for this config (mirrors kernel_lib/dest_helpers.hpp get_dest_limit)."""
    full_sync = bool(config.dst_full_sync_en)
    fp32 = bool(config.fp32_dest_acc_en)
    return (8 if fp32 else 16) if full_sync else (4 if fp32 else 8)


def compute_hw_config(config):
    """ComputeConfigDescriptor -> ComputeHardwareConfig. fpu_math_fidelity is always set (ttsim gap)."""
    return E.ComputeHardwareConfig(
        fpu_math_fidelity=config.math_fidelity,
        sfpu_precision_mode=E.Precision.Approximate if config.math_approx_mode else E.Precision.Precise,
        enable_32_bit_dest=bool(config.fp32_dest_acc_en),
        double_buffer_dest=not bool(config.dst_full_sync_en),
    )


def create_program_artifacts(
    a, b, out, config, *, threads=THREAD_PLAN, block_tiles=BLOCK_TILES, ring_depth=RING_DEPTH_BLOCKS,
    min_tiles_per_core=MIN_TILES_PER_CORE
):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for one `ttnn.generic_op` call.

    The keyword overrides exist so tests and the perf pass can turn each knob without editing code.
    """
    validate_thread_plan(threads)
    R, C, W = threads
    if ring_depth < 2:
        raise ValueError(f"ring_depth={ring_depth}: need >= 2 blocks per NEO")

    num_tiles = out.buffer_num_pages()
    core_set, counts, starts = split_pages(
        num_tiles, a.device().compute_with_storage_grid_size(), C, min_tiles_per_core
    )

    # Clamp the block once: never larger than one DEST section (mechanism cap; the kernel static_asserts it),
    # nor than a NEO's share on the busiest cluster.
    max_core_tiles = max(counts.values())
    block = max(1, min(block_tiles, dest_section_tiles(config), -(-max_core_tiles // C)))
    entries = ring_depth * block * C  # a STRIDED ring is sliced per NEO: capacity is counted per NEO

    tensors = (("input_a", a), ("input_b", b), ("output", out))
    rings = ((CB_INPUT_A, a), (CB_INPUT_B, b), (CB_OUTPUT, out))
    footprint = 0
    for name, t in rings:
        ring_bytes = entries * t.buffer_page_size()
        if ring_bytes > MAX_RING_BYTES:
            raise ValueError(f"add: ring {name} of {ring_bytes} B exceeds {MAX_RING_BYTES} B")
        footprint += ring_bytes
    budget = ttnn.get_max_worker_l1_unreserved_size()
    if footprint > budget:
        raise ValueError(f"add: ring footprint {footprint} B exceeds the L1 budget {budget} B")

    dfbs = [
        E.DataflowBufferSpec(
            unique_id=name, entry_size=t.buffer_page_size(), num_entries=entries, data_format_metadata=t.dtype
        )
        for name, t in rings
    ]
    tensor_parameters = [E.TensorParameter(unique_id=n, spec=t.spec) for n, t in tensors]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "add_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[E.producer_of(CB_INPUT_A, CB_INPUT_A), E.producer_of(CB_INPUT_B, CB_INPUT_B)],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="input_a", accessor_name="input_a"),
            E.TensorBinding(tensor_parameter_name="input_b", accessor_name="input_b"),
        ],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "add_compute.cpp"),
        hw_config=compute_hw_config(config),
        num_threads=C,
        dfb_bindings=[
            E.consumer_of(CB_INPUT_A, CB_INPUT_A),
            E.consumer_of(CB_INPUT_B, CB_INPUT_B),
            E.producer_of(CB_OUTPUT, CB_OUTPUT),
        ],
        compile_time_args={**plan, "block_tiles": int(block)},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "add_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of(CB_OUTPUT, CB_OUTPUT)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="add",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    dm_args = {"num_tiles": counts, "start_id": starts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_tiles": counts}),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {"input_a": 0, "input_b": 1, "output": 2}
