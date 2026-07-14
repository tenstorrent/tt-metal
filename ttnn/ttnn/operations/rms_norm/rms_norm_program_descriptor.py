# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""rms_norm program descriptor.

Parameterized row-parallel streaming reduction (see kernels/*.cpp). Every block
factor and buffer depth is a knob derived from one source of truth:

  * W_BLOCK_TARGET  — desired W-tiles per reduce chunk (Refinement 3 co-tune).
  * ROW_BLOCK_TILES — tile-rows per outer pass (phase-1: 1).

CB page counts and loop trip counts are computed FROM those knobs and the input
shape — never sized to Wt / W / sequence length, so per-core L1 stays bounded
for arbitrarily wide W. Phase-1 grid = single core owning all tile-rows; going
multi-core is a runtime-arg change (start_tile_row / num_tile_rows) with no
loop-nest change.

Refinement 3 (Data-movement co-tune, PERF). The real bottleneck (found by
on-device measurement, not the design's assumption) was NOT NoC bandwidth but
per-tile SYNCHRONIZATION: with W_BLOCK_TILES=1 the reader/writer ping-pong the
input/output CBs one tile at a time (reserve/read/barrier/push per tile), and the
compute runs each helper on a single tile — so the per-tile CB handshake + barrier
+ per-helper init/reconfig overhead dominates. (A DM-payload ablation — stubbing
the NoC reads while KEEPING the per-tile CB signaling — moved device-ns by 0%,
i.e. reads are hidden; the cost is the signaling GRANULARITY, not the transfer.)
The co-tune raises the block to W_BLOCK_TILES tiles, which compounds three levers:
  * compute_block_size — each square/reduce/mul/tilize/untilize helper runs on
    W_BLOCK_TILES tiles per call, amortizing init/reconfig/pipeline fill-drain.
  * double_buffer (reader + writer) — issue a whole block of async reads/writes
    then ONE barrier, and coarsen the reader->compute and compute->writer CB
    handshake W_BLOCK_TILES-fold (this is the dominant win).
  * transfer size (RM regime) — a W_BLOCK_TILES-wide stick slice is one big read
    instead of W_BLOCK_TILES narrow ones.
Measured (median device-ns, W_BLOCK_TARGET 1->8, WH B0, 1 core): TILE bf16
(1,1,32,8192) 283.5us -> 88.1us = 3.22x; RM bf16 same shape 6.43ms -> 0.85ms =
7.5x. Every guard-set path (TILE/RM x gamma/no_gamma x bf16/fp32) improved
2.0x-7.7x, none regressed. reader_placement (row_wise) is deferred to Refinement 4
(needs the multi-core reader). See changelog for the full before/after table.
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"

TILE_DIM = 32

# --- Blocking-model knobs (single source of truth) ---
# W_BLOCK_TARGET is the DESIRED reduce-chunk width in W-tiles. The effective
# W_BLOCK_TILES is derived per invocation (below) as the largest divisor of Wt
# that is <= this target, so every W-block is uniformly W_BLOCK_TILES tiles (no
# partial last W-block) in BOTH layout regimes and BOTH passes — the templated
# tilize/untilize<W_BLOCK_TILES> and the reader/writer block loops all stay
# uniform, and Wt % W_BLOCK_TILES == 0 holds by construction. Wide-W perf targets
# (Wt=128/256) get the full target; prime/awkward Wt degrade to a smaller divisor
# (small shapes where per-helper overhead is not the bottleneck anyway).
W_BLOCK_TARGET = 8  # phase-1 was 1; Refinement 3 co-tuned to 8 (measured sweet spot)
ROW_BLOCK_TILES = 1  # tile-rows per outer pass; see the assert in the body —
# NOT yet threaded into the compute row-loop (raising it needs a multi-row reduce
# + per-row cb_sumsq expansion). Left at 1 and guarded so it is not a silent
# half-wired knob; a follow-up refinement threads it through.


def _largest_divisor_leq(n: int, cap: int) -> int:
    """Largest divisor of n that is <= cap (>= 1). Keeps W-blocks uniform."""
    k = max(1, min(cap, n))
    while k > 1 and n % k != 0:
        k -= 1
    return k


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
    # Refinement 3 co-tune: pick the largest divisor of Wt that is <= the target
    # so every W-block is uniformly W_BLOCK_TILES tiles (no partial last W-block).
    # This is the single source of truth for the block factor; every dependent
    # (CB page counts, loop trip counts, num_w_blocks) derives from it below.
    W_BLOCK_TILES = _largest_divisor_leq(Wt, W_BLOCK_TARGET)
    assert Wt % W_BLOCK_TILES == 0, "W_BLOCK_TILES must divide Wt (holds by construction)"
    num_w_blocks = Wt // W_BLOCK_TILES
    # ROW_BLOCK_TILES>1 would need a multi-row reduce + per-row cb_sumsq expansion
    # in the compute row-loop, which is not implemented; guard so it is explicit.
    assert ROW_BLOCK_TILES == 1, "ROW_BLOCK_TILES>1 not yet threaded into the compute row-loop"

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

    # REDUCE DATAPATH selector (Refinement 1 for fp32; Refinement 2a extends it to
    # bf16). The tile-aligned float Σx² reduce runs on ReduceAlgorithm::AccumulateViaAdd:
    # the accumulator holds the RAW element-wise Σx² tile (folded per W-block with
    # add_tiles) and is reduced ONCE on the last block — removing the per-block
    # reduced-partial reload of ReduceTile whose truncation undercounts mean(x²) ∝ W.
    #   * fp32 (R1): the accumulator was already fp32 (interm_dtype), fixed the
    #     ∝W scale bias.
    #   * bf16 (R2a): the accumulator was bf16 and hit a catastrophic cliff at very
    #     wide W (W=32768: rel-RMS 0.40). Extending the AccumulateViaAdd datapath
    #     alone is NOT enough — the RAW running sum must not truncate, so the
    #     accumulator CB is forced to fp32 here (the reduce helper natively folds a
    #     bf16 input CB into an fp32 accumulator: reconfig_data_format_srcb/srca
    #     around the acc-add, per reduce_helpers_compute.inl).
    # bf8b stays on ReduceTile (already passes there, R2) and the non-tile-aligned
    # partial path stays on ReduceTile (AccumulateViaAdd cross-call cannot express
    # the masked partial tile). This is the R2 null-result's real fix: R2 measured
    # that merely forcing cb_sumsq fp32 on the *ReduceTile* path was a net regression
    # (removed the cliff but exposed the smooth ∝W bias); the fix is the fp32
    # accumulator ON the AccumulateViaAdd datapath, which has no ∝W bias.
    use_acc_via_add = in_dtype in (ttnn.float32, ttnn.bfloat16) and not has_partial_w
    # Accumulator CB format: fp32 whenever AccumulateViaAdd is used (raw Σx² must
    # not truncate); otherwise interm_dtype (unchanged ReduceTile path). cb_xsq
    # stays interm_dtype — it holds individual x² values (small; bf16 is fine) and
    # the helper handles the mixed bf16-input / fp32-accumulator fold.
    sumsq_dtype = ttnn.float32 if use_acc_via_add else interm_dtype

    # Gamma tiles: TILE gamma is a raw tile copy (reader writes the on-disk bytes,
    # so the CB MUST carry gamma_dtype); RM gamma is tilized by compute (which
    # converts format), so pack it at the intermediate precision (== in_dtype for
    # fp32/bf16 — unchanged; bf16 for bf8b input).
    gamma_tiles_dtype = gamma_dtype if (has_gamma and not gamma_is_row_major) else interm_dtype

    in_tile = ttnn.tile_size(in_dtype)
    out_tile = ttnn.tile_size(out_dtype)
    interm_tile = ttnn.tile_size(interm_dtype)
    sumsq_tile = ttnn.tile_size(sumsq_dtype)
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
        # Sum(x^2)/W accumulator -> 1/rms (held across pass 2). fp32 on the
        # AccumulateViaAdd path (raw Σx² must not truncate); interm_dtype otherwise.
        cb(CB_SUMSQ, sumsq_dtype, sumsq_tile, max(double * ROW_BLOCK_TILES, 2)),
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
        # 1/W as float bits: the mean scaler. For the tile-aligned AccumulateViaAdd
        # reduce path (SUM has no scaler tile) this is applied as the last-chunk
        # post_reduce_op; the ReduceTile path applies it via the bf16 scaler CB instead.
        scaler_bits,
        # USE_ACC_VIA_ADD selects the accurate AccumulateViaAdd reduce datapath
        # (R1 for fp32, R2a extends it to bf16 with an fp32 accumulator CB). Already
        # folds in !has_partial_w. bf8b and the non-tile-aligned partial path keep
        # the unchanged ReduceTile path.
        1 if use_acc_via_add else 0,
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
