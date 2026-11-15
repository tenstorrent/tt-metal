# SPDX-FileCopyrightText: (c) 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for gated_delta_net (regime R-STEP, op_design.md).

Work unit: one (sequence, value head) pair. Pairs are split across the grid with
ttnn.split_work_to_cores(row_wise=True); each core walks a contiguous pair range, one pair
state resident at a time, token rows in blocks of one tile row, one token per recurrence step.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE = 32
FP32_TILE_BYTES = 4096
BF16_TILE_BYTES = 2048

# ---- Blocking-model knobs (op_design.md, Blocking Model) ----
PAIRS_PER_BLOCK = 1  # pairs per work-split unit
TOKEN_TILE_ROWS_PER_BLOCK = 1  # prep / norm block along t (tile rows); only 1 is built
V_BLOCK_TILES = None  # state column block; None = Vt (R-VSPLIT deferred)
MAX_CORES = None  # optional grid cap; None = full compute grid

# ---- Buffer-depth knobs ----
WINDOW_DEPTH = 2  # channel groups of tap tiles in flight
TOK_COEF_DEPTH = 2  # tokens of coefficient pages in flight
ROW_DEPTH = 2  # tile rows of gate / z / output tiles in flight
STATE_IN_DEPTH = 1  # pair states (S0) in flight
STATE_OUT_DEPTH = 1  # finished pair states awaiting the writer
CONV_ACC_DEPTH = 1  # partial tap sum: compute-to-compute, one compute thread, so depth 1

HISTORY = 3  # spec constant: conv_state rows per sequence
CONV_TAPS = HISTORY + 1  # history rows + current token, not a knob
TOK_COEF_PAGES = 3  # alpha (all rows), beta at row p, alpha at row p
L1_BUDGET = 1460288  # usable L1 per Blackhole worker core (risks.md R11)

# ---- CB indices ----
CB_STATE_IN = 0
CB_STATE_IN_MM = 1
CB_STATE = 2
CB_STATE_MM = 3
CB_STATE_OUT = 4
CB_CONV_W = 5
CB_WINDOW = 6
CB_Y_QK = 7
CB_V = 8
CB_SQ = 9
CB_RINV = 10
CB_KQ_ROWS = 11
CB_KNT = 12
CB_D = 13
CB_GATE_IN = 14
CB_GATE_W = 15
CB_GATES = 16
CB_TOK_COEF = 17
CB_READS = 18
CB_U = 19
CB_O_ACC = 20
CB_Z = 21
CB_NORM_W = 22
CB_OUT_ROWS = 23
CB_REDUCE_SCALER = 24
CB_U_F32 = 25
CB_CONV_ACC = 26
CB_SCRATCH = 27
CB_SQ_O = 28
CB_RINV_O = 29
NUM_CB_SLOTS = 64


def _kernel_defines():
    """CB indices and spec constants for all three kernels (as GDN_<name>): defined here once, never restated."""
    names = [n for n in globals() if n.startswith("CB_")] + ["HISTORY", "CONV_TAPS", "TOK_COEF_PAGES"]
    return [(f"GDN_{n}", str(globals()[n])) for n in names]


# Pages per CB; each entry is (index list sharing one allocation, pages, page bytes, unpack-to-dest).
def _cb_plan(Kt, Vt):
    n_state = Kt * Vt
    trb = TOKEN_TILE_ROWS_PER_BLOCK
    return [
        ((CB_STATE_IN, CB_STATE_IN_MM), n_state * STATE_IN_DEPTH),
        ((CB_STATE, CB_STATE_MM), n_state),
        ((CB_STATE_OUT,), n_state * STATE_OUT_DEPTH),
        ((CB_CONV_W,), CONV_TAPS * (2 * Kt + Vt)),
        ((CB_WINDOW,), CONV_TAPS * max(Kt, Vt) * WINDOW_DEPTH),
        ((CB_CONV_ACC,), CONV_ACC_DEPTH),
        ((CB_Y_QK,), 2 * Kt * trb),
        ((CB_V,), Vt * trb),
        # L2-norm and RMS-norm squares / rsqrt columns get separate buffers: they are pushed in
        # different counts and read by index, so a shared FIFO would leave a phase's block wrapped.
        ((CB_SQ,), Kt * trb),
        ((CB_RINV,), 2 * trb),
        ((CB_SQ_O,), Vt * trb),
        ((CB_RINV_O,), trb),
        ((CB_KQ_ROWS,), 2 * Kt * trb),
        ((CB_KNT,), Kt * trb),
        ((CB_D,), trb),
        ((CB_GATE_IN,), 2 * ROW_DEPTH),
        ((CB_GATE_W,), 2),
        ((CB_GATES,), 2 * trb),
        ((CB_TOK_COEF,), TOK_COEF_PAGES * TOK_COEF_DEPTH),
        ((CB_READS,), 2 * Vt),
        ((CB_U, CB_U_F32), Vt),
        ((CB_O_ACC,), Vt * trb),
        ((CB_Z,), Vt * ROW_DEPTH),
        ((CB_NORM_W,), Vt),
        ((CB_OUT_ROWS,), Vt * ROW_DEPTH),
        ((CB_SCRATCH,), 1),
    ]


# Indices read by copy_tile / SFPU at full fp32 (UnpackToDestFp32); the rest feed srcA/srcB or dataflow.
U2D_CBS = (
    CB_STATE_IN,
    CB_STATE,
    CB_CONV_W,
    CB_WINDOW,
    CB_CONV_ACC,
    CB_Y_QK,
    CB_V,
    CB_SQ,
    CB_RINV,
    CB_D,
    CB_GATE_IN,
    CB_GATE_W,
    CB_TOK_COEF,
    CB_READS,
    CB_U_F32,
    CB_O_ACC,
    CB_Z,
    CB_NORM_W,
)


def l1_footprint_bytes(Kt, Vt):
    return sum(pages for _, pages in _cb_plan(Kt, Vt)) * FP32_TILE_BYTES + BF16_TILE_BYTES


def _float_bits(value):
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


def _largest_divisor_at_most(n, cap):
    return max(d for d in range(1, min(n, cap) + 1) if n % d == 0)


def create_program_descriptor(inputs, outputs, *, epsilon, compute_kernel_config):
    mixed_qkv, z, b, a, conv_state, recurrent_state, conv_weight, A_log, dt_bias, norm_weight = inputs
    output, conv_state_out, recurrent_state_out = outputs

    assert TOKEN_TILE_ROWS_PER_BLOCK == 1, "only one tile row per prep block is built"
    batch, value_heads, key_dim, value_dim = (int(d) for d in recurrent_state.shape)
    rows, channels = int(mixed_qkv.shape[1]), int(mixed_qkv.shape[2])
    tokens = rows // batch
    Kt, Vt = key_dim // TILE, value_dim // TILE
    v_block_tiles = Vt if V_BLOCK_TILES is None else V_BLOCK_TILES
    assert v_block_tiles == Vt, "R-VSPLIT (V_BLOCK_TILES < Vt) is not built"
    key_heads = (channels - value_heads * value_dim) // (2 * key_dim)
    Ct = channels // TILE
    Gt = (value_heads + TILE - 1) // TILE

    # ---- work split: pairs in blocks of PAIRS_PER_BLOCK ----
    device = mixed_qkv.device()
    grid = device.compute_with_storage_grid_size()
    if MAX_CORES is not None:
        grid = ttnn.CoreCoord(grid.x, max(1, min(grid.y, MAX_CORES // grid.x)))
    num_pairs = batch * value_heads
    num_blocks = (num_pairs + PAIRS_PER_BLOCK - 1) // PAIRS_PER_BLOCK
    (_, all_cores, group_1, group_2, blocks_1, blocks_2) = ttnn.split_work_to_cores(grid, num_blocks, row_wise=True)

    assignment = []  # (core, first_pair, pair_count)
    next_block = 0
    for group, per_core in ((group_1, blocks_1), (group_2, blocks_2)):
        if per_core == 0:
            continue
        for core in ttnn.corerange_to_cores(group, row_wise=True):
            first_pair = next_block * PAIRS_PER_BLOCK
            last_pair = min(num_pairs, (next_block + per_core) * PAIRS_PER_BLOCK)
            assignment.append((core, first_pair, last_pair - first_pair))
            next_block += per_core

    # ---- circular buffers ----
    cbs = []
    for indices, pages in _cb_plan(Kt, Vt):
        formats = []
        for idx in indices:
            formats.append(
                ttnn.CBFormatDescriptor(buffer_index=idx, data_format=ttnn.float32, page_size=FP32_TILE_BYTES)
            )
        cbs.append(
            ttnn.CBDescriptor(total_size=pages * FP32_TILE_BYTES, core_ranges=all_cores, format_descriptors=formats)
        )
    cbs.append(
        ttnn.CBDescriptor(
            total_size=BF16_TILE_BYTES,
            core_ranges=all_cores,
            format_descriptors=[
                ttnn.CBFormatDescriptor(
                    buffer_index=CB_REDUCE_SCALER, data_format=ttnn.bfloat16, page_size=BF16_TILE_BYTES
                )
            ],
        )
    )

    # ---- reader ----
    reader_tensors = (
        mixed_qkv,
        z,
        b,
        a,
        conv_state,
        recurrent_state,
        conv_weight,
        A_log,
        dt_bias,
        norm_weight,
        conv_state_out,
    )
    reader_ct = [Kt, Vt, value_heads, key_heads, Ct, Gt]
    for t in reader_tensors:
        reader_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    reader_addrs = [t.buffer_address() for t in reader_tensors]

    # ---- writer ----
    writer_ct = [Kt, Vt, value_heads]
    for t in (output, recurrent_state_out):
        writer_ct.extend(ttnn.TensorAccessorArgs(t).get_compile_time_args())
    writer_addrs = [output.buffer_address(), recurrent_state_out.buffer_address()]

    # ---- compute ----
    compute_ct = [
        Kt,
        Vt,
        value_heads,
        _largest_divisor_at_most(Vt, 4),  # read-matmul subblock width (fp32 half-sync DEST: 4 tiles)
        _float_bits(value_dim**0.5),
        _float_bits(key_dim**-0.5),
    ]

    reader_rt, writer_rt, compute_rt = ttnn.RuntimeArgs(), ttnn.RuntimeArgs(), ttnn.RuntimeArgs()
    for core, first_pair, count in assignment:
        reader_rt[core.x][core.y] = reader_addrs + [first_pair, count, tokens]
        writer_rt[core.x][core.y] = writer_addrs + [first_pair, count, tokens]
        compute_rt[core.x][core.y] = [first_pair, count, tokens, _float_bits(value_dim * epsilon)]

    modes = list(getattr(compute_kernel_config, "unpack_to_dest_mode", []) or [])
    if len(modes) < NUM_CB_SLOTS:
        modes = modes + [ttnn.UnpackToDestMode.Default] * (NUM_CB_SLOTS - len(modes))
    for idx in U2D_CBS:
        modes[idx] = ttnn.UnpackToDestMode.UnpackToDestFp32
    compute_kernel_config.unpack_to_dest_mode = modes

    defines = _kernel_defines()
    kernels = [
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gated_delta_net_reader.cpp"),
            core_ranges=all_cores,
            compile_time_args=reader_ct,
            runtime_args=reader_rt,
            defines=defines,
            config=ttnn.ReaderConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gated_delta_net_writer.cpp"),
            core_ranges=all_cores,
            compile_time_args=writer_ct,
            runtime_args=writer_rt,
            defines=defines,
            config=ttnn.WriterConfigDescriptor(),
        ),
        ttnn.KernelDescriptor(
            kernel_source=str(KERNEL_DIR / "gated_delta_net_compute.cpp"),
            core_ranges=all_cores,
            compile_time_args=compute_ct,
            defines=defines + [("INP_FLOAT32", "1")],
            runtime_args=compute_rt,
            config=compute_kernel_config,
        ),
    ]
    return ttnn.ProgramDescriptor(kernels=kernels, semaphores=[], cbs=cbs)
