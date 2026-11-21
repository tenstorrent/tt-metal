# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for subtract, regime R0 `tile_stream_same_shape` (op_design.md).

Blocking model:
  * the output's tile pages, numbered flat (`tile_id`), are split into contiguous ranges over the whole
    grid (`ttnn.split_work_to_cores`, row-wise); a cluster's range is its block (`core_tiles`, runtime arg
    `num_tiles`, with `start_id`);
  * inside a cluster the thread plan THREAD_PLAN = (R, C, W) deals the range's tiles round-robin over the
    STRIDED rings: reader hart p reads tiles p, p+R, ..., NEO t computes t, t+C, ..., writer hart w writes
    w, w+W, ...;
  * each NEO makes ONE `sub` helper call over its share, taking `sync_tiles` tiles per DEST section and
    per ring handshake; every ring holds ENTRIES_PER_NEO = 2 * sync_tiles entries per NEO (double
    buffering).

Every knob is defined once here and passed down: THREAD_PLAN, SYNC_TILES (clamped by DEST capacity),
MIN_TILES_PER_CORE (grid engagement).
"""

from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# ---------------------------------------------------------------------------
# Knobs (single source of truth)
# ---------------------------------------------------------------------------

# Thread plan (R readers, C NEOs, W writers) per cluster. Data-movement bound (one FPU op per tile over
# tensors in DRAM): the target page's plan is (1, 1, 1). The kernels are written for any legal plan.
THREAD_PLAN = (1, 1, 1)

# Tiles per DEST section and per ring handshake inside the one helper call; clamped by DEST capacity.
SYNC_TILES = 4

# Ring depth per NEO, in handshake quanta: 2 = double buffering.
RING_DEPTH_QUANTA = 2

# Grid engagement: at least this many tiles per engaged cluster (1 = fill the grid first).
MIN_TILES_PER_CORE = 1

# The cluster model of the target page (.claude/references/targets/quasar.md); Python has no query for them.
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper
NEOS = 4


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in (1, 2, 4) and c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(
            f"thread plan (R, C, W) = {threads}: need C in {{1, 2, 4}}, R + W <= {FREE_DM_HARTS}, R | C, W | C"
        )


def dest_tiles(fp32_dest_acc_en: bool, dst_full_sync_en: bool) -> int:
    """DEST capacity per section, the table of kernel_lib's get_dest_limit() (dest_helpers.hpp)."""
    if dst_full_sync_en:
        return 8 if fp32_dest_acc_en else 16
    return 4 if fp32_dest_acc_en else 8


def compute_hw_config(config) -> "E.ComputeHardwareConfig":
    """Translate the resolved ttnn.ComputeConfigDescriptor into the Quasar ComputeHardwareConfig.

    fpu_math_fidelity is always set explicitly (ttsim aborts otherwise, quasar-library-gaps.md).
    """
    return E.ComputeHardwareConfig(
        fpu_math_fidelity=config.math_fidelity,
        sfpu_precision_mode=E.Precision.Approximate if config.math_approx_mode else E.Precision.Precise,
        enable_32_bit_dest=bool(config.fp32_dest_acc_en),
        double_buffer_dest=not bool(config.dst_full_sync_en),
    )


def tensor_tile_count(shape) -> int:
    """N * ceil(H/32) * ceil(W/32): per-image ceil, never floor(N*H/32)."""
    shape = list(shape)
    n = 1
    for d in shape[:-2]:
        n *= d
    h = shape[-2] if len(shape) >= 2 else 1
    w = shape[-1]
    return n * ((h + 31) // 32) * ((w + 31) // 32)


def split_tiles(num_tiles, grid_size, min_tiles_per_core=MIN_TILES_PER_CORE):
    """(core set, {core: tiles}, {core: first tile}) over the clusters that get work, row-major."""
    num_cores = grid_size.x * grid_size.y
    engaged = max(1, min(num_cores, -(-num_tiles // max(1, min_tiles_per_core))))
    # One path for every case: pick the `engaged` clusters row-wise (one placeholder unit each), then deal
    # the tiles over them in contiguous ranges, the first `extra` clusters taking one more (the same
    # assignment split_work_to_cores(grid, num_tiles) gives when every cluster is engaged).
    _, core_set, group_1, group_2, _, _ = ttnn.split_work_to_cores(grid_size, engaged, row_wise=True)
    cores = [c for g in (group_1, group_2) for c in ttnn.corerange_to_cores(g, None, True)]
    base, extra = divmod(num_tiles, engaged)
    counts, starts, start = {}, {}, 0
    for i, core in enumerate(cores):
        counts[core] = base + (1 if i < extra else 0)
        starts[core] = start
        start += counts[core]
    return core_set, counts, starts


def create_program_artifacts(a, b, out, config, *, threads=None, sync_tiles=None):
    """(ProgramSpec, ProgramRunArgs, tensor indices) for `out = a - b` (same shape, TILE, interleaved).

    `threads` / `sync_tiles` override THREAD_PLAN / SYNC_TILES (tests of non-default plans and quanta).
    """
    threads = tuple(threads) if threads is not None else THREAD_PLAN
    validate_thread_plan(threads)
    R, C, W = threads

    requested_sync = SYNC_TILES if sync_tiles is None else int(sync_tiles)
    sync = max(1, min(requested_sync, dest_tiles(bool(config.fp32_dest_acc_en), bool(config.dst_full_sync_en))))
    entries_per_neo = RING_DEPTH_QUANTA * sync

    num_tiles = out.buffer_num_pages()
    expected = tensor_tile_count(out.shape)
    if num_tiles != expected:
        raise RuntimeError(f"subtract: output has {num_tiles} tile pages, the shape gives {expected}")

    core_set, counts, starts = split_tiles(num_tiles, a.device().compute_with_storage_grid_size())

    # A STRIDED ring is sliced into one sub-ring per NEO, so its capacity is counted per NEO.
    dfbs = [
        E.DataflowBufferSpec(
            unique_id=name,
            entry_size=t.buffer_page_size(),
            num_entries=entries_per_neo * C,
            data_format_metadata=t.dtype,
        )
        for name, t in (("cb_minuend_tiles", a), ("cb_subtrahend_tiles", b), ("cb_difference_tiles", out))
    ]
    tensor_parameters = [
        E.TensorParameter(unique_id=n, spec=t.spec) for n, t in (("minuend", a), ("subtrahend", b), ("difference", out))
    ]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles", "start_id"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "subtract_reader.cpp"),
        hw_config=E.create_reader_datamovement_config(),
        num_threads=R,
        dfb_bindings=[
            E.producer_of("cb_minuend_tiles", "cb_minuend_tiles"),
            E.producer_of("cb_subtrahend_tiles", "cb_subtrahend_tiles"),
        ],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="minuend", accessor_name="minuend"),
            E.TensorBinding(tensor_parameter_name="subtrahend", accessor_name="subtrahend"),
        ],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "subtract_compute.cpp"),
        hw_config=compute_hw_config(config),
        num_threads=C,
        dfb_bindings=[
            E.consumer_of("cb_minuend_tiles", "cb_minuend_tiles"),
            E.consumer_of("cb_subtrahend_tiles", "cb_subtrahend_tiles"),
            E.producer_of("cb_difference_tiles", "cb_difference_tiles"),
        ],
        compile_time_args={**plan, "sync_tiles": sync},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_tiles"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "subtract_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of("cb_difference_tiles", "cb_difference_tiles")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="difference", accessor_name="difference")],
        compile_time_args=plan,
        runtime_arg_schema=dm_schema,
    )
    spec = E.ProgramSpec(
        name="subtract",
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
    return spec, run_args, {"minuend": 0, "subtrahend": 1, "difference": 2}
