# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm program descriptor.

Parameterized row-parallel streaming reduction (see kernels/*.cpp). Every block
factor and buffer depth is a knob derived from one source of truth:

  * W_BLOCK_TILES  — W-tiles streamed per reduce chunk (phase-1: 1).
  * ROW_BLOCK_TILES — tile-rows per outer pass (phase-1: 1).

CB page counts and loop trip counts are computed FROM those knobs and the input
shape — never sized to Wt / W / sequence length, so per-core L1 stays bounded
for arbitrarily wide W. Phase-1 grid = single core owning all tile-rows; going
multi-core is a runtime-arg change (start_tile_row / num_tile_rows) with no
loop-nest change.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# --- Blocking-model knobs (tunable from day 1; single source of truth) ---
W_BLOCK_TILES = 1  # W-tiles streamed per reduce chunk
ROW_BLOCK_TILES = 1  # tile-rows processed per outer pass


def _f32_bits(value: float) -> int:
    return struct.unpack("I", struct.pack("f", float(value)))[0]


def _elt(tensor) -> int:
    """Per-element byte size, tolerant of block formats.

    bfloat8_b / bfloat4_b have no well-defined per-element size (16 values share
    one exponent) and `element_size()` raises for them. They are TILE-only, so
    the byte size only ever feeds ROW_MAJOR-stick page math that is never
    allocated for a block-format tensor — return a harmless stand-in of 1.
    """
    try:
        return tensor.element_size()
    except (ValueError, RuntimeError):
        return 1


def create_program_descriptor(
    input_tensor: ttnn.Tensor,
    output_tensor: ttnn.Tensor,
    *,
    gamma: ttnn.Tensor = None,
    epsilon: float = 1e-6,
    compute_kernel_config: ttnn.ComputeConfigDescriptor = None,
) -> ttnn.ProgramDescriptor:
    shape = list(input_tensor.shape)
    origin_H = int(shape[-2])
    origin_W = int(shape[-1])
    leading = 1
    for d in shape[:-2]:
        leading *= int(d)

    is_row_major = input_tensor.layout == ttnn.ROW_MAJOR_LAYOUT
    has_gamma = gamma is not None
    gamma_is_row_major = has_gamma and gamma.layout == ttnn.ROW_MAJOR_LAYOUT

    # --- Derived geometry (all from the knobs + shape) ---
    Wt = math.ceil(origin_W / TILE_DIM)
    tiles_per_image = math.ceil(origin_H / TILE_DIM)
    total_tile_rows = leading * tiles_per_image
    # Phase-1: W_BLOCK_TILES == 1 divides every Wt. (A knob > 1 that does not
    # divide Wt would need partial-last-W-block handling — a later refinement.)
    assert Wt % W_BLOCK_TILES == 0, "W_BLOCK_TILES must divide Wt (phase-1 uses 1)"
    num_w_blocks = Wt // W_BLOCK_TILES

    partial_w = origin_W % TILE_DIM
    has_partial_w = partial_w != 0
    scaler_bits = _f32_bits(1.0 / float(origin_W))
    eps_bits = _f32_bits(epsilon)

    input_elt = _elt(input_tensor)
    output_elt = _elt(output_tensor)
    gamma_elt = _elt(gamma) if has_gamma else input_elt

    in_dtype = input_tensor.dtype
    out_dtype = output_tensor.dtype
    gamma_dtype = gamma.dtype if has_gamma else in_dtype

    # Intermediate (accumulator/scratch) CB format. fp32 input keeps fp32
    # intermediates (fp32 dest-acc path); bf16 keeps bf16; bf8b uses bf16 — a
    # block-float accumulator/scratch would be far too lossy for Sum(x^2) and the
    # x*(1/rms) scratch. This is identity for fp32/bf16 (byte-identical to prior
    # passing cells) and only lifts bf8b's intermediates off the block format.
    interm_dtype = ttnn.bfloat16 if in_dtype == ttnn.bfloat8_b else in_dtype

    # NOTE (R2, null result): forcing cb_sumsq to Float32 when fp32_dest_acc_en=True
    # (numeric-formats-metal §4) was measured on the wide-W bf16 loose cases and
    # REVERTED — it is a net regression, not a win. The bf16 accumulator has a
    # cliff (fine <=W=16384, catastrophic at 32768: rel-RMS 0.40); the fp32
    # accumulator instead grows SMOOTHLY with W (the residual ReduceTile
    # matmul-with-ones bias R1 fixed for fp32 via AccumulateViaAdd but left on the
    # bf16 path), so fp32-sumsq pushes W=16384 from 0.037 pass -> 0.044 fail while
    # W=32768 (0.092) still fails. No single-core config passes both wide cases;
    # the real fix is the R1-analog AccumulateViaAdd fp32-raw-accumulator datapath
    # for bf16 — a follow-up refinement (see op_requirements Refinement 2a). No
    # cartesian SUPPORTED cell (W<=8192) needs it, so cb_sumsq stays interm_dtype.

    # Gamma tiles: TILE gamma is a raw tile copy (reader writes the on-disk bytes,
    # so the CB MUST carry gamma_dtype); RM gamma is tilized by compute (which
    # converts format), so pack it at the intermediate precision (== in_dtype for
    # fp32/bf16 — unchanged; bf16 for bf8b input).
    gamma_tiles_dtype = gamma_dtype if (has_gamma and not gamma_is_row_major) else interm_dtype

    in_tile = ttnn.tile_size(in_dtype)
    out_tile = ttnn.tile_size(out_dtype)
    interm_tile = ttnn.tile_size(interm_dtype)
    gamma_tiles_tile = ttnn.tile_size(gamma_tiles_dtype)
    scaler_tile = ttnn.tile_size(ttnn.bfloat16)

    wblock_cols = W_BLOCK_TILES * TILE_DIM
    in_rm_page = wblock_cols * input_elt  # one W-block-wide stick slice
    out_rm_page = out_tile  # untilize emits tile-sized pages
    gamma_rm_page = wblock_cols * gamma_elt

    # --- Grid: phase-1 single core owns all tile-rows ---
    core = ttnn.CoreCoord(0, 0)
    core_grid = ttnn.CoreRangeSet([ttnn.CoreRange(core, core)])
    start_tile_row = 0
    num_tile_rows = total_tile_rows

    # --- CB indices (semantic) ---
    CB_INPUT_RM = 0
    CB_INPUT_TILES = 1
    CB_SCALER = 2
    CB_GAMMA_RM = 3
    CB_GAMMA_TILES = 4
    CB_OUTPUT_TILES = 16
    CB_OUTPUT_RM = 17
    CB_XSQ = 24
    CB_SUMSQ = 25
    CB_NORM = 26

    # --- Buffer-depth knobs (data-movement<->compute overlap, not reuse) ---
    STICK_BLOCK = TILE_DIM  # one tile-row height of sticks
    double = 2

    def cb(index, dtype, page_size, num_pages):
        return ttnn.CBDescriptor(
            total_size=num_pages * page_size,
            core_ranges=core_grid,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=dtype, page_size=page_size)],
        )

    cbs = [
        # streamed input tiles (reader in TILE regime / tilize in RM regime)
        cb(CB_INPUT_TILES, in_dtype, in_tile, double * W_BLOCK_TILES),
        # reduce scaler (bf16): full [+ partial]
        cb(CB_SCALER, ttnn.bfloat16, scaler_tile, 2 if has_partial_w else 1),
        # normalized output tiles (mul -> writer in TILE / -> untilize in RM)
        cb(CB_OUTPUT_TILES, out_dtype, out_tile, double * W_BLOCK_TILES),
        # x^2 scratch (pass 1) — compute->compute, single-depth full block
        cb(CB_XSQ, interm_dtype, interm_tile, W_BLOCK_TILES),
        # Sum(x^2)/W accumulator -> 1/rms (held across pass 2)
        cb(CB_SUMSQ, interm_dtype, interm_tile, max(double * ROW_BLOCK_TILES, 2)),
    ]
    if is_row_major:
        # RM input sticks -> tilize; RM output sticks <- untilize
        cbs.append(cb(CB_INPUT_RM, in_dtype, in_rm_page, double * STICK_BLOCK))
        cbs.append(cb(CB_OUTPUT_RM, out_dtype, out_rm_page, double * W_BLOCK_TILES))
    if has_gamma:
        # cb_gamma_rm only exists on the RM-gamma leg (reader sticks -> compute
        # tilize). TILE gamma skips it: the reader writes tiles straight into
        # cb_gamma_tiles (single producer per build — the two legs are separate
        # compiled programs, so the one-producer rule holds per build).
        if gamma_is_row_major:
            cbs.append(cb(CB_GAMMA_RM, gamma_dtype, gamma_rm_page, double))
        cbs.append(cb(CB_GAMMA_TILES, gamma_tiles_dtype, gamma_tiles_tile, double * W_BLOCK_TILES))
        # normalize scratch (x*(1/rms)) before gamma mul — compute->compute
        cbs.append(cb(CB_NORM, interm_dtype, interm_tile, W_BLOCK_TILES))

    # ================= Reader =================
    reader_ct_args = [
        1 if is_row_major else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        partial_w if has_partial_w else TILE_DIM,
        scaler_bits,
        origin_W,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        input_elt,
        gamma_elt,
        # RM gamma -> stick-read + compute tilize; TILE gamma -> read tiles direct.
        1 if gamma_is_row_major else 0,
    ]
    reader_ct_args.extend(ttnn.TensorAccessorArgs(input_tensor).get_compile_time_args())
    reader_ct_args.extend(
        ttnn.TensorAccessorArgs(gamma).get_compile_time_args()
        if has_gamma
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )
    reader_rt_args = ttnn.RuntimeArgs()
    reader_rt_args[core.x][core.y] = [
        input_tensor.buffer_address(),
        gamma.buffer_address() if has_gamma else 0,
        start_tile_row,
        num_tile_rows,
    ]
    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_reader.cpp"),
        core_ranges=core_grid,
        compile_time_args=reader_ct_args,
        runtime_args=reader_rt_args,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ================= Writer =================
    writer_ct_args = [
        1 if is_row_major else 0,
        origin_W,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        output_elt,
    ]
    writer_ct_args.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())
    writer_rt_args = ttnn.RuntimeArgs()
    writer_rt_args[core.x][core.y] = [
        output_tensor.buffer_address(),
        start_tile_row,
        num_tile_rows,
    ]
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_writer.cpp"),
        core_ranges=core_grid,
        compile_time_args=writer_ct_args,
        runtime_args=writer_rt_args,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ================= Compute =================
    compute_ct_args = [
        1 if is_row_major else 0,
        1 if has_gamma else 0,
        1 if has_partial_w else 0,
        origin_H,
        tiles_per_image,
        Wt,
        W_BLOCK_TILES,
        num_w_blocks,
        # 1/W as float bits: the mean scaler. For the fp32 tile-aligned reduce path
        # (AccumulateViaAdd, SUM has no scaler tile) this is applied as the last-chunk
        # post_reduce_op; the ReduceTile path applies it via the bf16 scaler CB instead.
        scaler_bits,
        # IS_FP32 selects the fp32-accurate reduce datapath (AccumulateViaAdd) — see
        # the compute kernel. Keeps bf16/bf8b on the unchanged ReduceTile path.
        1 if in_dtype == ttnn.float32 else 0,
        # GAMMA_IS_ROW_MAJOR: RM gamma is tilized by compute; TILE gamma arrives
        # already tiled from the reader (skip the gamma-tilize step).
        1 if gamma_is_row_major else 0,
    ]
    compute_rt_args = ttnn.RuntimeArgs()
    compute_rt_args[core.x][core.y] = [num_tile_rows, start_tile_row, eps_bits]
    # compute_kernel_config is threaded through as-is (math_fidelity /
    # fp32_dest_acc_en / math_approx_mode / dst_full_sync_en all honored by the
    # kernel's helpers). No CB is tagged UnpackToDestFp32: every fp32 intermediate
    # here (cb_xsq, cb_sumsq) feeds an FPU op — the reduce, or the AccumulateViaAdd
    # add_tiles fold — and UnpackToDestFp32 is exclusive with any FPU consumer
    # (numeric-formats-metal §1.5). So no CB qualifies; tagging would break math.
    compute_config = compute_kernel_config if compute_kernel_config is not None else ttnn.ComputeConfigDescriptor()
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "rms_norm_compute.cpp"),
        core_ranges=core_grid,
        compile_time_args=compute_ct_args,
        runtime_args=compute_rt_args,
        config=compute_config,
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
