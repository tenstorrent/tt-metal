# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for avg_pool2d, regime `col_split_stream` (op_design.md).

The global mean of one image [1, 1, H*W, C] (TILE, bfloat16) is a column reduction: every channel tile
column `ct` of the input (Ht = ceil(H*W / 32) tiles) reduces to row 0 of one output tile. The channel tile
columns are independent and are spread over the whole grid with `split_work_to_cores`; each engaged cluster
owns a contiguous range [ct_start, ct_start + core_ct_tiles) and processes it in blocks of `block_ct_tiles`
columns (Phase 0: one block, the whole assignment).

Per cluster (thread plan (R, C, W) = THREAD_PLAN = (1, 1, 1): the op is bound by data movement):

    reader   streams the block's tiles into `in` in the reduce helper's REDUCE_COL order
             (DEST chunk of `dest_chunk_tiles` columns -> position tile row -> column of the chunk), and zeroes
             rows [last_tile_valid_rows, 32) of each tile of the last position tile row in L1 (rule R1).
             Explicit sync, batched per chunk row (one reserve / barrier / push), user transaction id.
    writer   fills the scaler pair once ([1/(H*W) in all 32 position lanes, 1/(H*W) in the first
             `last_tile_valid_rows` lanes only]), then drains `out` to the output's tile row, implicit sync.

R1 (padding rows enter neither sum nor count) is the reader's zero fill, not the partial scaler. The partial
scaler stays (with zeroed padding it masks nothing valid) as the design's helper contract.
    compute  one reduce<SUM, REDUCE_COL> per block with the partial scaler on the last position row.

Every knob is defined once here: BLOCK_CT_TILES_CAP, INPUT_DEPTH_CHUNK_ROWS, OUTPUT_DEPTH_CHUNKS, THREAD_PLAN,
and `dest_chunk_tiles`, which is derived from the same ComputeHardwareConfig handed to the compute kernel.
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

TILE = 32

# Thread plan (R readers, C NEOs, W writers) per cluster. The op is data-movement bound (one FPU reduce_tile per
# 2 KB tile moved once). The kernels take the three counts as compile-time args.
THREAD_PLAN = (1, 1, 1)
FREE_DM_HARTS = 6  # target page, cluster model: 6 of 8 DM harts are free for user kernels
NEOS = 4

# Block knob: channel tiles per block. None = unbounded: a block is the cluster's whole assignment (Phase 0).
BLOCK_CT_TILES_CAP = None

# Buffer-depth knobs, in units of the DEST chunk (the reduce helper's REDUCE_COL streaming unit).
INPUT_DEPTH_CHUNK_ROWS = 2  # `in`: double buffer by chunk row -- the reader fills chunk row k+1 (one batch, one
                            # barrier) while compute unpacks chunk row k
OUTPUT_DEPTH_CHUNKS = 2     # `out`: compute packs chunk k+1 while the writer drains chunk k
SCALER_TILES = 2            # [full scaler, partial scaler]: ReducePartialScaler::with_partial()

# The DEST double-buffering (half sync) the compute kernel runs with; dest_chunk_tiles follows from it.
DOUBLE_BUFFER_DEST = True


def validate_thread_plan(threads):
    r, c, w = threads
    if not (1 <= c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need C <= {NEOS}, R + W <= {FREE_DM_HARTS}, R | C, W | C")


def dest_chunk_tiles_for(hw_config) -> int:
    """DEST tiles the REDUCE_COL helper accumulates at once: the rule of dest_helpers.hpp get_dest_limit()
    (half sync: 4 with a 32-bit DEST, 8 with 16-bit; full sync twice that), from the compute kernel's config."""
    base = 4 if hw_config.enable_32_bit_dest else 8
    return base * (1 if hw_config.double_buffer_dest else 2)


def compute_hw_config(compute_config) -> "E.ComputeHardwareConfig":
    """The compute kernel's hardware config from the caller's (normalized) ComputeConfigDescriptor.
    fpu_math_fidelity is always set explicitly."""
    return E.ComputeHardwareConfig(
        fpu_math_fidelity=compute_config.math_fidelity,
        sfpu_precision_mode=E.Precision.Approximate if compute_config.math_approx_mode else E.Precision.Precise,
        enable_32_bit_dest=bool(compute_config.fp32_dest_acc_en),
        double_buffer_dest=DOUBLE_BUFFER_DEST,
    )


def split_channel_tiles(tensor_ct_tiles, grid_size, block_ct_tiles_cap=BLOCK_CT_TILES_CAP):
    """(core set, {core: (ct_start, core_ct_tiles, block_ct_tiles)}) over the clusters that get work.

    The channel tile columns are dealt to the whole grid (row-major); each cluster owns a contiguous range.
    """
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, tensor_ct_tiles, row_wise=True
    )
    work, start = {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            block = per_core if block_ct_tiles_cap is None else min(per_core, block_ct_tiles_cap)
            work[core] = (start, per_core, block)
            start += per_core
    assert start == tensor_ct_tiles, (start, tensor_ct_tiles)
    return all_cores, work


def _dm_config(make, explicit_rings):
    """A data-movement hardware config with `explicit_rings` opted out of implicit sync (the kernel keeps their
    credits itself: reserve / push, wait / pop)."""
    hw = make()
    hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _scaler_bits(value: float, scaler_dtype) -> int:
    """The scaler as fp32 bits for the writer. A bfloat16 scaler ring is filled by prepare_reduce_scaler, which
    truncates (keeps the top 16 bits): round to the nearest bfloat16 here (ties to even), so the 1/(H*W) the
    reduce multiplies by is the nearest bfloat16, not the one below it (half the worst-case scale error)."""
    bits = _f32_bits(value)
    if scaler_dtype == ttnn.bfloat16:
        bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return bits


def create_program_artifacts(x, out, *, positions, compute_config, threads=THREAD_PLAN,
                             block_ct_tiles_cap=BLOCK_CT_TILES_CAP, grid_size=None):
    """ProgramSpec, ProgramRunArgs and tensor indices for one image's global mean.

    `positions` = H*W, the number of valid rows of the flattened input (the divisor; rule R1).
    `grid_size` overrides the queried compute grid (a test narrows it to force several DEST chunks per cluster).
    """
    validate_thread_plan(threads)
    R, C, W = threads
    tile_bytes = x.buffer_page_size()
    tensor_ct_tiles = x.padded_shape[-1] // TILE
    Ht = -(-positions // TILE)
    last_tile_valid_rows = positions - TILE * (Ht - 1)  # in [1, 32] by construction
    inv_hw_bits = _scaler_bits(1.0 / positions, x.dtype)  # the scaler ring has the input's format

    hw = compute_hw_config(compute_config)
    dest_chunk_tiles = dest_chunk_tiles_for(hw)

    if grid_size is None:
        grid_size = x.device().compute_with_storage_grid_size()
    core_set, work = split_channel_tiles(tensor_ct_tiles, grid_size, block_ct_tiles_cap)

    # A STRIDED ring is sliced into one sub-ring per NEO: capacity is counted per NEO (x C).
    def ring(name, entries_per_neo, dtype):
        return E.DataflowBufferSpec(
            unique_id=name, entry_size=tile_bytes, num_entries=entries_per_neo * C, data_format_metadata=dtype
        )

    in_entries_per_neo = INPUT_DEPTH_CHUNK_ROWS * dest_chunk_tiles
    dfbs = [
        ring("in", in_entries_per_neo, x.dtype),
        ring("scaler", SCALER_TILES, x.dtype),
        ring("out", OUTPUT_DEPTH_CHUNKS * dest_chunk_tiles, out.dtype),
    ]
    tensor_parameters = [E.TensorParameter(unique_id=n, spec=t.spec) for n, t in (("input", x), ("output", out))]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W, "dest_chunk_tiles": dest_chunk_tiles}

    def per_core(value_of):
        return {core: value_of(core) for core in work}

    ct_start = per_core(lambda c: work[c][0])
    core_ct_tiles = per_core(lambda c: work[c][1])
    block_ct_tiles = per_core(lambda c: work[c][2])
    uniform = lambda v: per_core(lambda c: v)  # noqa: E731

    # The reader assembles each `in` entry itself (a read, then the R1 padding-row zero fill on the last position
    # tile row): explicit sync, so that ring is opted out of implicit sync.
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "avg_pool2d_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, ["in"]),
        num_threads=R,
        dfb_bindings=[E.producer_of("in", "in")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="input", accessor_name="input")],
        compile_time_args={**plan, "in_entries_per_neo": in_entries_per_neo},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["ct_start", "core_ct_tiles", "block_ct_tiles", "Ht", "tensor_ct_tiles",
                               "last_tile_valid_rows"]
        ),
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "avg_pool2d_compute.cpp"),
        hw_config=hw,
        num_threads=C,
        dfb_bindings=[E.consumer_of("in", "in"), E.consumer_of("scaler", "scaler"), E.producer_of("out", "out")],
        compile_time_args=plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["core_ct_tiles", "block_ct_tiles", "Ht"]),
    )
    # The writer fills the scaler ring itself (explicit sync: reserve / push), so that ring is opted out of
    # implicit sync; the output ring keeps the default (implicit, one tagged write per entry).
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "avg_pool2d_writer.cpp"),
        hw_config=_dm_config(E.create_writer_datamovement_config, ["scaler"]),
        num_threads=W,
        dfb_bindings=[E.producer_of("scaler", "scaler"), E.consumer_of("out", "out")],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="output", accessor_name="output")],
        compile_time_args=plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(
            runtime_arg_names=["ct_start", "core_ct_tiles", "inv_hw_bits", "last_tile_valid_rows"]
        ),
    )
    spec = E.ProgramSpec(
        name="avg_pool2d",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values={
                "ct_start": ct_start, "core_ct_tiles": core_ct_tiles, "block_ct_tiles": block_ct_tiles,
                "Ht": uniform(Ht), "tensor_ct_tiles": uniform(tensor_ct_tiles),
                "last_tile_valid_rows": uniform(last_tile_valid_rows)}),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={
                "core_ct_tiles": core_ct_tiles, "block_ct_tiles": block_ct_tiles, "Ht": uniform(Ht)}),
            E.KernelRunArgs(kernel="writer", runtime_arg_values={
                "ct_start": ct_start, "core_ct_tiles": core_ct_tiles, "inv_hw_bits": uniform(inv_hw_bits),
                "last_tile_valid_rows": uniform(last_tile_valid_rows)}),
        ]
    )
    return spec, run_args, {"input": 0, "output": 1}
