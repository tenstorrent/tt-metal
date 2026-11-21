# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for rms_norm (regime `row_resident`, op_design.md).

Work distribution
    The input's tile-rows (N*C*ceil(H/32), contiguous in tile order) are grouped into units of
    ROWS_PER_BLOCK tile-rows. A cluster runs C_eff NEOs; units are dealt to the clusters in whole rounds of
    C_eff (split_work_to_cores over rounds), and inside a cluster unit k*C_eff + t goes to NEO t. Every NEO
    runs the same number of rounds; a NEO left without a unit in the last round recomputes NEO 0's unit of
    that round (the reader feeds it the same tiles, the writer stores the same bytes to the same pages), so
    at most C_eff - 1 units are recomputed in the whole op.

Thread plan
    THREAD_PLAN = (R, C, W) is the one statement of it. Per call C_eff = the largest c in {1, 2, 4}, c <= C,
    whose input ring fits the ring extent cap and that does not exceed the number of units; R_eff = min(R,
    C_eff), W_eff = min(W, C_eff). The counts reach the kernels as compile-time arguments.

Per NEO and unit (rows_per_block x Wt tiles, resident in `input_tiles`)
    sum_squares   sum_w x_w^2 accumulated in DEST, one tile per tile-row      -> row_stat
    reduce_row    SUM over the 32 columns                                    -> row_sum
    inv_rms       rsqrt(row_sum * (1/W) + epsilon)                            -> row_stat
    normalize     x * inv_rms (column broadcast) * gamma (row broadcast)      -> output_tiles

Rings (every capacity is per NEO x C_eff: a STRIDED ring is sliced per NEO, also the compute-owned ones)
    input_tiles   IN_DEPTH_ROWS * ROWS_PER_BLOCK * Wt   reader -> compute, implicit sync
    gamma_tiles   GAMMA_DEPTH_BLOCKS * gamma_block       reader -> compute, explicit sync (assembled entries)
    scaler        1                                      reader -> compute, explicit sync
    row_stat      ROWS_PER_BLOCK                         compute -> compute
    row_sum       ROWS_PER_BLOCK                         compute -> compute
    output_tiles  OUT_DEPTH_TILES                        compute -> writer, implicit sync
"""

from __future__ import annotations

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

TILE = 32

# ---------------------------------------------------------------------------
# Knobs (op_design.md, Blocking Model). Each is defined once; everything below derives from them.
# ---------------------------------------------------------------------------

THREAD_PLAN = (4, 4, 2)  # (R readers, C NEOs, W writers) per cluster: bound by compute, reads > writes
ROWS_PER_BLOCK = 1  # tile-rows one NEO holds and drives through one pass (`rows_per_block`)
IN_DEPTH_ROWS = 1  # whole units of input per NEO in `input_tiles` (the unit stays resident across both passes)
GAMMA_BLOCK_MAX_TILES = 16  # gamma tiles assembled per reader barrier: largest divisor of Wt <= this
GAMMA_DEPTH_BLOCKS = 2  # gamma blocks per NEO (one assembling while one is consumed)
OUT_DEPTH_TILES = 2  # output tiles per NEO (pack of tile w+1 overlaps the write of tile w)

# Ring formats fixed by a mechanism, each stated once (the ring size, the ring format and the kernel's entry
# geometry all derive from it).
GAMMA_FORMAT = ttnn.bfloat16  # 16-bit: Quasar's row broadcast refuses a 32-bit page (bcast.h)
SCALER_FORMAT = ttnn.bfloat16  # the reduce scaler (1.0, exact)

# Mechanism caps and the cluster model (target page). Python has no query for these.
DFB_EXTENT_CAP_BYTES = 1_048_560  # a ring's extent is a uint16 in 16-B units
L1_RESERVE_BYTES = 524_288  # left for L1-resident input/output/weight banks, which the query does not subtract
FREE_DM_HARTS = 6  # of 8: DM0 serves the ring interrupts, DM1 the remapper
NEOS = 4
NEO_COUNTS = (1, 2, 4)

# Test-only hook: NEO t multiplies its output by t + 1, so a test can see which NEO computed which tile-row.
_TEST_TAG_THREADS = False

# Names
DFB_INPUT = "input_tiles"
DFB_GAMMA = "gamma_tiles"
DFB_SCALER = "scaler"
DFB_ROW_STAT = "row_stat"
DFB_ROW_SUM = "row_sum"
DFB_OUTPUT = "output_tiles"
TP_INPUT = "input"
TP_WEIGHT = "weight"
TP_OUTPUT = "output"


def fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in NEO_COUNTS and c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(
            f"rms_norm: thread plan (R, C, W) = {threads}: need C in {NEO_COUNTS}, R + W <= {FREE_DM_HARTS}, R | C, W | C"
        )


def _geometry(input_tensor):
    """(Wt, num_tile_rows, num_units, tile_bytes)."""
    wt = int(input_tensor.padded_shape[-1]) // TILE
    num_tile_rows = input_tensor.buffer_num_pages() // wt
    num_units = -(-num_tile_rows // ROWS_PER_BLOCK)
    return wt, num_tile_rows, num_units, input_tensor.buffer_page_size()


def _stat_dtype(compute_config):
    """The per-row statistics follow DEST width (ledger audit 2)."""
    return ttnn.float32 if compute_config.fp32_dest_acc_en else ttnn.bfloat16


def _gamma_block_tiles(wt: int) -> int:
    """Largest divisor of Wt that is <= GAMMA_BLOCK_MAX_TILES (a block never straddles the ring end)."""
    for candidate in range(min(GAMMA_BLOCK_MAX_TILES, wt), 0, -1):
        if wt % candidate == 0:
            return candidate
    return 1


def _ring_bytes_per_neo(wt: int, tile_bytes: int, stat_bytes: int):
    """{ring: (entries per NEO, entry bytes)}."""
    gamma_block = _gamma_block_tiles(wt)
    return {
        DFB_INPUT: (IN_DEPTH_ROWS * ROWS_PER_BLOCK * wt, tile_bytes),
        DFB_GAMMA: (GAMMA_DEPTH_BLOCKS * gamma_block, ttnn.tile_size(GAMMA_FORMAT)),
        DFB_SCALER: (1, ttnn.tile_size(SCALER_FORMAT)),
        DFB_ROW_STAT: (ROWS_PER_BLOCK, stat_bytes),
        DFB_ROW_SUM: (ROWS_PER_BLOCK, stat_bytes),
        DFB_OUTPUT: (OUT_DEPTH_TILES, tile_bytes),
    }


def _l1_budget() -> int:
    return ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES


def _fits(c: int, rings: dict, budget: int) -> bool:
    total = 0
    for entries, entry_bytes in rings.values():
        ring = entries * c * entry_bytes
        if ring > DFB_EXTENT_CAP_BYTES:
            return False
        total += ring
    return total <= budget


def select_regime(input_tensor, compute_config, threads=None):
    """The regime and its thread plan for a call: {"regime", "C", "R", "W"}.

    `row_resident` when a unit fits one ring at one NEO; otherwise the deferred `row_streaming` regime would be
    needed, and the call is refused.
    """
    threads = THREAD_PLAN if threads is None else threads
    validate_thread_plan(threads)
    r, c_plan, w = threads
    wt, _, num_units, tile_bytes = _geometry(input_tensor)
    rings = _ring_bytes_per_neo(wt, tile_bytes, ttnn.tile_size(_stat_dtype(compute_config)))
    budget = _l1_budget()
    fitting = [c for c in NEO_COUNTS if c <= c_plan and _fits(c, rings, budget)]
    if not fitting:
        raise NotImplementedError(
            f"rms_norm: a unit of {ROWS_PER_BLOCK} tile-row(s) x {wt} tiles exceeds one ring at C=1 "
            "(regime row_streaming not built)"
        )
    c_eff = max([c for c in fitting if c <= num_units] or [min(fitting)])
    return {"regime": "row_resident", "C": c_eff, "R": min(r, c_eff), "W": min(w, c_eff)}


def split_units(num_units, grid_size, c):
    """(core set, {core: units}, {core: first unit}), dealing whole rounds of C units to the clusters."""
    num_rounds = -(-num_units // c)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid_size, num_rounds, row_wise=True)
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = min(per_core * c, num_units - start), start
            start += counts[core]
    return all_cores, counts, starts


def _dm_config(make, explicit_rings):
    hw = make()
    if explicit_rings:
        hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(disable_dfb_implicit_sync_for=explicit_rings)
    return hw


def create_program_artifacts(input_tensor, weight, output_tensor, *, epsilon, compute_config, threads=None):
    regime = select_regime(input_tensor, compute_config, threads)
    C, R, W = regime["C"], regime["R"], regime["W"]
    wt, num_tile_rows, num_units, tile_bytes = _geometry(input_tensor)
    stat_dtype = _stat_dtype(compute_config)
    rings = _ring_bytes_per_neo(wt, tile_bytes, ttnn.tile_size(stat_dtype))
    gamma_block = _gamma_block_tiles(wt)
    core_set, counts, starts = split_units(num_units, input_tensor.device().compute_with_storage_grid_size(), C)

    formats = {
        DFB_INPUT: input_tensor.dtype,
        DFB_GAMMA: GAMMA_FORMAT,
        DFB_SCALER: SCALER_FORMAT,
        DFB_ROW_STAT: stat_dtype,
        DFB_ROW_SUM: stat_dtype,
        DFB_OUTPUT: output_tensor.dtype,
    }
    dfbs = [
        E.DataflowBufferSpec(
            unique_id=name,
            entry_size=entry_bytes,
            num_entries=entries * C,
            data_format_metadata=formats[name],
        )
        for name, (entries, entry_bytes) in rings.items()
    ]
    tensor_parameters = [
        E.TensorParameter(unique_id=TP_INPUT, spec=input_tensor.spec),
        E.TensorParameter(unique_id=TP_WEIGHT, spec=weight.spec),
        E.TensorParameter(unique_id=TP_OUTPUT, spec=output_tensor.spec),
    ]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W}
    shape_args = {"Wt": wt, "rows_per_block": ROWS_PER_BLOCK}
    dm_schema = E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_units", "unit_start", "total_rows"])

    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        hw_config=_dm_config(E.create_reader_datamovement_config, [DFB_GAMMA, DFB_SCALER]),
        num_threads=R,
        dfb_bindings=[
            E.producer_of(DFB_INPUT, DFB_INPUT),
            E.producer_of(DFB_GAMMA, DFB_GAMMA),
            E.producer_of(DFB_SCALER, DFB_SCALER),
        ],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name=TP_INPUT, accessor_name=TP_INPUT),
            E.TensorBinding(tensor_parameter_name=TP_WEIGHT, accessor_name=TP_WEIGHT),
        ],
        compile_time_args={
            **plan,
            **shape_args,
            "gamma_block_tiles": gamma_block,
            # The gamma entry's geometry, from the one statement of its format: a weight page (one row of 32
            # gammas) and the bytes of one face of a gamma ring entry (a tile is 4 faces of 16 x 16).
            "gamma_page_bytes": weight.buffer_page_size(),
            "gamma_face_bytes": rings[DFB_GAMMA][1] // 4,
        },
        runtime_arg_schema=dm_schema,
    )

    unpack_modes = {}
    if compute_config.fp32_dest_acc_en:
        # A 32-bit ring under enable_32_bit_dest names its unpack mode; both are read by FPU consumers
        # (the reduce, the column broadcast) and by copy_tile, so through SrcA/SrcB.
        unpack_modes = {DFB_ROW_STAT: E.UnpackMode.UnpackToSrc, DFB_ROW_SUM: E.UnpackMode.UnpackToSrc}
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        hw_config=E.ComputeHardwareConfig(
            fpu_math_fidelity=compute_config.math_fidelity,
            sfpu_precision_mode=E.Precision.Approximate if compute_config.math_approx_mode else E.Precision.Precise,
            enable_32_bit_dest=bool(compute_config.fp32_dest_acc_en),
            unpack_modes=unpack_modes,
        ),
        num_threads=C,
        dfb_bindings=[
            E.consumer_of(DFB_INPUT, DFB_INPUT),
            E.consumer_of(DFB_GAMMA, DFB_GAMMA),
            E.consumer_of(DFB_SCALER, DFB_SCALER),
            E.producer_of(DFB_ROW_STAT, DFB_ROW_STAT),
            E.consumer_of(DFB_ROW_STAT, DFB_ROW_STAT),
            E.producer_of(DFB_ROW_SUM, DFB_ROW_SUM),
            E.consumer_of(DFB_ROW_SUM, DFB_ROW_SUM),
            E.producer_of(DFB_OUTPUT, DFB_OUTPUT),
        ],
        compile_time_args={**plan, **shape_args, "tag_threads": int(bool(_TEST_TAG_THREADS))},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_units", "inv_w_bits", "eps_bits"]),
    )

    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of(DFB_OUTPUT, DFB_OUTPUT)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name=TP_OUTPUT, accessor_name=TP_OUTPUT)],
        compile_time_args={**plan, **shape_args},
        runtime_arg_schema=dm_schema,
    )

    spec = E.ProgramSpec(
        name="rms_norm",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"], target_nodes=core_set)],
    )

    # The mean is over the W logical elements (rule R6), not 32 * Wt.
    inv_w_bits = fp32_bits(1.0 / float(int(input_tensor.shape[-1])))
    eps_bits = fp32_bits(epsilon)
    dm_args = {"num_units": counts, "unit_start": starts, "total_rows": {core: num_tile_rows for core in counts}}
    compute_args = {
        "num_units": counts,
        "inv_w_bits": {core: inv_w_bits for core in counts},
        "eps_bits": {core: eps_bits for core in counts},
    }
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values=dm_args),
            E.KernelRunArgs(kernel="compute", runtime_arg_values=compute_args),
            E.KernelRunArgs(kernel="writer", runtime_arg_values=dm_args),
        ]
    )
    return spec, run_args, {TP_INPUT: 0, TP_WEIGHT: 1, TP_OUTPUT: 2}
