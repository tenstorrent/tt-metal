# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""Metal 2.0 program artifacts for layer_norm (regime `row_resident`, op_design.md).

Work unit: a tile-row (Wt tiles: 32 rows of the input, the whole reduced axis). Rows are independent, so
they are spread over the grid: whole rounds of C rows go to the clusters (`split_rows`), and inside a
cluster NEO t owns local rows t, t+C, ... (`rounds` of them; a NEO left without a row in the last round
recomputes NEO 0's row of that round, and the writer stores the same bytes again). A NEO drives its rows
through the phase sequence in blocks of `block_tile_rows` rows; every ring that holds a block is sized
from that knob.

Thread plan: THREAD_PLAN = (R readers, C NEOs, W writers) per cluster, clamped per call by
`select_blocking()` to the work there is (fewer rows than clusters x C engages fewer NEOs per cluster)
and to what fits L1.

Every ring is STRIDED on both sides (the default), so its capacity is counted per NEO and allocated
`entries_per_neo * C`. The L1 ledger (l1_ledger.md) is the inventory and the footprint formula below.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
E = ttnn.experimental

# ---- knobs: each defined once, everything else derives from them ----
THREAD_PLAN = (4, 4, 2)          # (R, C, W): the plan of a compute-bound op with several inputs
L1_RESERVE_BYTES = 32768         # headroom kept out of the unreserved L1 per cluster
DFB_RING_CAP_BYTES = 1_048_560   # target page: largest ring extent on the compute side
INPUT_BLOCK_DEPTH = 1            # whole blocks of `cb_input_block` per NEO (a whole number: no resident window straddles the ring end)
OUTPUT_DEPTH = 2                 # pages of `cb_output_tiles` per NEO (packer <-> writer double buffer)
AFFINE_STREAM_DEPTH = 2          # pages of `cb_affine_stream` per NEO (reader <-> gamma/beta expansion)
# The cluster model of the target page; Python has no query for them.
NEOS = 4
FREE_DM_HARTS = 6
TILE = 32

# Ring names (the kernels' dfb:: names).
INPUT_BLOCK = "input_block"
AFFINE_STREAM = "affine_stream"
GAMMA_ROWS = "gamma_rows"
BETA_ROWS = "beta_rows"
CENTRED = "centred"
ROW_PARTIAL = "row_partial"
ROW_STAT = "row_stat"
OUTPUT_TILES = "output_tiles"
MEAN_SCALER = "mean_scaler"
VAR_SCALER = "var_scaler"
EPSILON = "epsilon"
READER_EXPLICIT_RINGS = [MEAN_SCALER, VAR_SCALER, EPSILON]  # written by the hart itself
# Rings in the intermediate format that the compute kernel reads (each needs an unpack mode under 32-bit DEST).
COMPUTE_READ_RINGS = [GAMMA_ROWS, BETA_ROWS, CENTRED, ROW_PARTIAL, ROW_STAT, EPSILON, MEAN_SCALER, VAR_SCALER]

TENSOR_INDICES = {"x": 0, "weight": 1, "bias": 2, "out": 3}


@dataclass(frozen=True)
class ComputeConfig:
    math_fidelity: object
    math_approx_mode: bool
    fp32_dest_acc_en: bool


def resolve_compute_config(config) -> ComputeConfig:
    """Rule R4: a ttnn.ComputeConfigDescriptor or tt-metal's device compute kernel config; the three fields
    are read as attributes, and the fields a descriptor lacks are ignored."""
    return ComputeConfig(
        math_fidelity=config.math_fidelity,
        math_approx_mode=bool(config.math_approx_mode),
        fp32_dest_acc_en=bool(config.fp32_dest_acc_en),
    )


def validate_thread_plan(threads):
    r, c, w = threads
    if not (c in (1, 2, 4) and c <= NEOS and r >= 1 and w >= 1 and r + w <= FREE_DM_HARTS and c % r == 0 and c % w == 0):
        raise ValueError(f"thread plan (R, C, W) = {threads}: need C in (1, 2, 4), R + W <= {FREE_DM_HARTS}, R | C, W | C")


# ---------------------------------------------------------------------------
# Page sizes and the footprint (l1_ledger.md, Footprint)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Pages:
    t_in: int    # input / output page
    t_aff: int   # weight / bias page
    t_acc: int   # intermediate page (Float32 under fp32 accumulation, else bfloat16); also the reduce
                 # scalers' and epsilon's page: the operands of one FPU chain share one ALU format


def acc_dtype(fp32_dest_acc_en: bool):
    return ttnn.float32 if fp32_dest_acc_en else ttnn.bfloat16


def pages_of(x, weight, fp32_dest_acc_en: bool) -> Pages:
    return Pages(
        t_in=x.buffer_page_size(),
        t_aff=weight.buffer_page_size(),
        t_acc=ttnn.tile_size(acc_dtype(fp32_dest_acc_en)),
    )


def ring_entries_per_neo(bh: int, Wt: int) -> dict:
    """Entries per NEO of every ring, from the block knobs (the ledger's capacity column)."""
    return {
        INPUT_BLOCK: INPUT_BLOCK_DEPTH * bh * Wt,
        AFFINE_STREAM: AFFINE_STREAM_DEPTH,
        GAMMA_ROWS: Wt,
        BETA_ROWS: Wt,
        CENTRED: bh * Wt,
        ROW_PARTIAL: bh,  # three FIFO uses per block: 2·Σx, then Σc², then rstd (disjoint lifetimes)
        ROW_STAT: bh,
        OUTPUT_TILES: OUTPUT_DEPTH,
        MEAN_SCALER: 1,
        VAR_SCALER: 1,
        EPSILON: 1,
    }


def ring_page_bytes(p: Pages) -> dict:
    return {
        INPUT_BLOCK: p.t_in,
        AFFINE_STREAM: p.t_aff,
        GAMMA_ROWS: p.t_acc,
        BETA_ROWS: p.t_acc,
        CENTRED: p.t_acc,
        ROW_PARTIAL: p.t_acc,
        ROW_STAT: p.t_acc,
        OUTPUT_TILES: p.t_in,
        MEAN_SCALER: p.t_acc,
        VAR_SCALER: p.t_acc,
        EPSILON: p.t_acc,
    }


def l1_budget() -> int:
    return ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES


def fits(c: int, bh: int, Wt: int, p: Pages) -> bool:
    """c NEOs per cluster, bh tile-rows per block: the whole cluster footprint fits the L1 budget, and every
    ring fits the ring-extent cap."""
    entries, page = ring_entries_per_neo(bh, Wt), ring_page_bytes(p)
    per_neo = sum(entries[name] * page[name] for name in entries)
    if c * per_neo > l1_budget():
        return False
    return all(c * entries[name] * page[name] <= DFB_RING_CAP_BYTES for name in entries)


def fits_resident(x, weight, fp32_dest_acc_en: bool) -> bool:
    """The `row_resident` regime's predicate: one NEO, one tile-row per block."""
    Wt = -(-x.padded_shape[-1] // TILE)
    return fits(1, 1, Wt, pages_of(x, weight, fp32_dest_acc_en))


# ---------------------------------------------------------------------------
# Work distribution
# ---------------------------------------------------------------------------


def split_rows(num_rows, grid_size, C):
    """(core set, {core: tile-rows}, {core: first tile-row}), dealing whole rounds of C rows to the clusters,
    so that only the last round of the last cluster can be short."""
    num_rounds = -(-num_rows // C)
    _, all_cores, group_1, group_2, per_core_1, per_core_2 = ttnn.split_work_to_cores(
        grid_size, num_rounds, row_wise=True
    )
    counts, starts, start = {}, {}, 0
    for group, per_core in ((group_1, per_core_1), (group_2, per_core_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, None, True):
            counts[core], starts[core] = min(per_core * C, num_rows - start), start
            start += counts[core]
    return all_cores, counts, starts


@dataclass(frozen=True)
class Blocking:
    reader_threads: int
    compute_threads: int
    writer_threads: int
    block_tile_rows: int
    Wt: int
    num_tile_rows: int
    core_set: object
    counts: dict
    starts: dict

    @property
    def rounds_max(self) -> int:
        return max(-(-n // self.compute_threads) for n in self.counts.values())

    @property
    def blocks_per_neo_max(self) -> int:
        return -(-self.rounds_max // self.block_tile_rows)


def select_blocking(x, weight, fp32_dest_acc_en, *, thread_plan=THREAD_PLAN, max_clusters=None,
                    max_block_tile_rows=None) -> Blocking:
    """The thread plan and the block knobs for one call (op_design.md, Thread plan selection).

    C: the smallest of 1, 2, 4 (at most the plan's C) with no more rounds than clusters, else the plan's C;
    halved while the cluster footprint does not fit. R and W: the plan's, clamped to C. The block: a NEO's
    whole share of rows when it fits, else the largest number of rows that fits.
    `max_clusters` and `max_block_tile_rows` are test overrides (they pin a regime on a small shape).
    """
    validate_thread_plan(thread_plan)
    R_plan, C_plan, W_plan = thread_plan
    Wt = -(-x.padded_shape[-1] // TILE)
    num_tile_rows = x.buffer_num_pages() // Wt
    p = pages_of(x, weight, fp32_dest_acc_en)

    grid = x.device().compute_with_storage_grid_size()
    if max_clusters is not None:
        grid = ttnn.CoreCoord(min(max_clusters, grid.x), 1)
    num_clusters = grid.x * grid.y

    candidates = [c for c in (1, 2, 4) if c <= C_plan]
    C = next((c for c in candidates if -(-num_tile_rows // c) <= num_clusters), C_plan)
    while C > 1 and not fits(C, 1, Wt, p):
        C //= 2
    R, W = min(R_plan, C), min(W_plan, C)

    core_set, counts, starts = split_rows(num_tile_rows, grid, C)
    rounds_max = max(-(-n // C) for n in counts.values())
    bh = rounds_max if max_block_tile_rows is None else min(rounds_max, max_block_tile_rows)
    while bh > 1 and not fits(C, bh, Wt, p):
        bh -= 1
    return Blocking(R, C, W, bh, Wt, num_tile_rows, core_set, counts, starts)


# ---------------------------------------------------------------------------
# The program
# ---------------------------------------------------------------------------


def fp32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def epsilon_word(epsilon: float, fp32_dest_acc_en: bool) -> int:
    """The first 32-bit word of the epsilon tile (element [0, 0], all a scalar broadcast reads), in the
    intermediate format: float32 bits, or bfloat16 (round to nearest even) in both halves."""
    bits = fp32_bits(epsilon)
    if fp32_dest_acc_en:
        return bits
    bf16 = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) & 0xFFFF
    return (bf16 << 16) | bf16


def create_program_artifacts(x, weight, bias, out, *, epsilon: float, config: ComputeConfig,
                             thread_plan=THREAD_PLAN, max_clusters=None, max_block_tile_rows=None,
                             blocking: Blocking | None = None):
    if blocking is None:
        blocking = select_blocking(x, weight, config.fp32_dest_acc_en, thread_plan=thread_plan,
                                   max_clusters=max_clusters, max_block_tile_rows=max_block_tile_rows)
    R, C, W = blocking.reader_threads, blocking.compute_threads, blocking.writer_threads
    bh, Wt = blocking.block_tile_rows, blocking.Wt
    acc = acc_dtype(config.fp32_dest_acc_en)
    p = pages_of(x, weight, config.fp32_dest_acc_en)
    entries, page = ring_entries_per_neo(bh, Wt), ring_page_bytes(p)
    fmt = {
        INPUT_BLOCK: x.dtype, AFFINE_STREAM: weight.dtype, GAMMA_ROWS: acc, BETA_ROWS: acc,
        CENTRED: acc, ROW_PARTIAL: acc, ROW_STAT: acc, OUTPUT_TILES: out.dtype,
        MEAN_SCALER: acc, VAR_SCALER: acc, EPSILON: acc,
    }
    dfbs = [
        E.DataflowBufferSpec(unique_id=name, entry_size=page[name], num_entries=entries[name] * C,
                             data_format_metadata=fmt[name])
        for name in entries
    ]
    tensor_parameters = [
        E.TensorParameter(unique_id=n, spec=t.spec) for n, t in (("x", x), ("weight", weight), ("bias", bias), ("out", out))
    ]
    W_logical = x.shape[-1]
    plan = {"reader_threads": R, "compute_threads": C, "writer_threads": W, "Wt": Wt}

    reader_hw = E.create_reader_datamovement_config()
    reader_hw.config_2xx = E.DataMovementHardwareConfig.DataMovement2XXConfig(
        disable_dfb_implicit_sync_for=READER_EXPLICIT_RINGS
    )
    reader = E.KernelSpec(
        unique_id="reader",
        source=str(KERNEL_DIR / "layer_norm_reader.cpp"),
        hw_config=reader_hw,
        num_threads=R,
        dfb_bindings=[
            E.producer_of(INPUT_BLOCK, INPUT_BLOCK),
            E.producer_of(AFFINE_STREAM, AFFINE_STREAM),
            E.producer_of(MEAN_SCALER, MEAN_SCALER),
            E.producer_of(VAR_SCALER, VAR_SCALER),
            E.producer_of(EPSILON, EPSILON),
        ],
        tensor_bindings=[
            E.TensorBinding(tensor_parameter_name="x", accessor_name="x"),
            E.TensorBinding(tensor_parameter_name="weight", accessor_name="weight"),
            E.TensorBinding(tensor_parameter_name="bias", accessor_name="bias"),
        ],
        compile_time_args={
            **plan,
            # The row-sum pass accumulates x + x, so the mean's scaler is 1 / (2W); the variance's is 1 / W.
            "mean_scaler_bits": fp32_bits(1.0 / (2.0 * W_logical)),
            "var_scaler_bits": fp32_bits(1.0 / W_logical),
        },
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_rows", "row_start", "epsilon_word"]),
    )
    compute = E.KernelSpec(
        unique_id="compute",
        source=str(KERNEL_DIR / "layer_norm_compute.cpp"),
        hw_config=E.ComputeHardwareConfig(
            fpu_math_fidelity=config.math_fidelity,  # set explicitly
            sfpu_precision_mode=E.Precision.Approximate if config.math_approx_mode else E.Precision.Precise,
            enable_32_bit_dest=config.fp32_dest_acc_en,
            # A Float32 ring the compute kernel reads needs an explicit unpack mode under 32-bit DEST; the
            # default unpack (through the source registers) is the one this kernel's FPU ops take.
            unpack_modes={name: E.UnpackMode.UnpackToSrc for name in COMPUTE_READ_RINGS if fmt[name] == ttnn.float32},
        ),
        num_threads=C,
        dfb_bindings=[
            E.consumer_of(INPUT_BLOCK, INPUT_BLOCK),
            E.consumer_of(AFFINE_STREAM, AFFINE_STREAM),
            E.producer_of(GAMMA_ROWS, GAMMA_ROWS),
            E.consumer_of(GAMMA_ROWS, GAMMA_ROWS),
            E.producer_of(BETA_ROWS, BETA_ROWS),
            E.consumer_of(BETA_ROWS, BETA_ROWS),
            E.producer_of(CENTRED, CENTRED),
            E.consumer_of(CENTRED, CENTRED),
            E.producer_of(ROW_PARTIAL, ROW_PARTIAL),
            E.consumer_of(ROW_PARTIAL, ROW_PARTIAL),
            E.producer_of(ROW_STAT, ROW_STAT),
            E.consumer_of(ROW_STAT, ROW_STAT),
            E.producer_of(OUTPUT_TILES, OUTPUT_TILES),
            E.consumer_of(MEAN_SCALER, MEAN_SCALER),
            E.consumer_of(VAR_SCALER, VAR_SCALER),
            E.consumer_of(EPSILON, EPSILON),
        ],
        compile_time_args={**plan, "block_tile_rows": bh},
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_rows"]),
    )
    writer = E.KernelSpec(
        unique_id="writer",
        source=str(KERNEL_DIR / "layer_norm_writer.cpp"),
        hw_config=E.create_writer_datamovement_config(),
        num_threads=W,
        dfb_bindings=[E.consumer_of(OUTPUT_TILES, OUTPUT_TILES)],
        tensor_bindings=[E.TensorBinding(tensor_parameter_name="out", accessor_name="out")],
        compile_time_args=plan,
        runtime_arg_schema=E.KernelSpec.RuntimeArgSchema(runtime_arg_names=["num_rows", "row_start"]),
    )
    spec = E.ProgramSpec(
        name="layer_norm",
        kernels=[reader, compute, writer],
        dataflow_buffers=dfbs,
        tensor_parameters=tensor_parameters,
        work_units=[E.WorkUnitSpec(name="main", kernels=["reader", "compute", "writer"],
                                   target_nodes=blocking.core_set)],
    )
    counts, starts = blocking.counts, blocking.starts
    eps = {core: epsilon_word(epsilon, config.fp32_dest_acc_en) for core in counts}
    run_args = E.ProgramRunArgs(
        kernel_run_args=[
            E.KernelRunArgs(kernel="reader", runtime_arg_values={"num_rows": counts, "row_start": starts,
                                                                 "epsilon_word": eps}),
            E.KernelRunArgs(kernel="compute", runtime_arg_values={"num_rows": counts}),
            E.KernelRunArgs(kernel="writer", runtime_arg_values={"num_rows": counts, "row_start": starts}),
        ]
    )
    return spec, run_args, dict(TENSOR_INDICES)
