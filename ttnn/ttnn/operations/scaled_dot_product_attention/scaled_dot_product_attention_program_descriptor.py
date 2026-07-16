# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for scaled_dot_product_attention (FlashAttention-2).

Blocking model (see op_design.md, binding):
  * Split B·H·q-chunks (all independent) across the grid — no cross-core combine.
  * Each core loops its assigned work units; per work unit it streams all KV
    chunks once and folds them into a running (m, l, O) online-softmax recurrence.
  * Every CB page count and kernel loop bound derives from three block knobs
    (Sq_chunk_t, Skv_chunk_t, KV_DEPTH) + Dt — no CB grows with S_q or S_kv.
"""

import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# ---- L1 budget (bytes) for the streaming working set. Conservative. ----
L1_BUDGET = 1_400_000

# ---- Compute-block factor targets (single source of truth; R3a) ----
# The block factor is the COARSEST chunk that fits L1 (design "coarsest that
# fits"). R3a raises the target from the phase-0 value 4 to 8 as the compute-side
# perf lever on the flagged compute-bound shape: a coarser chunk (a) amortizes the
# ~10 sequential per-chunk online-softmax helper phases (each pays a fixed
# reconfig/init/fill-drain) over 2x the tiles by halving n_kv_chunks, and (b) grows
# the QKᵀ matmul out_subblock_w toward the fp32_dest_acc_en=False DEST budget (8
# bf16 tiles) since out_subblock_w = decomp_n(Skv_chunk_t, dest_limit). `_fit_l1`
# shrinks from these targets under L1 pressure and `_chunk_size` caps per axis
# (and enforces the straddle-safe remainder), so raising the target is safe for
# every shape/dtype — the coarsest that fits is chosen, never larger than L1 or the
# axis tile-count allows.
SQ_CHUNK_TARGET = 8
SKV_CHUNK_TARGET = 8
KV_DEPTH_DEFAULT = 2
OUT_DEPTH_DEFAULT = 2

# ---- CB indices (semantic names; slots are just buffer indices) ----
CB_Q_IN = 0
CB_K_IN = 1
CB_V_IN = 2
CB_MASK_IN = 3
CB_SCALER = 4
CB_SCALE = 5
CB_M_NEW = 6  # scratch: new running-max before overwriting cb_row_max
CB_SUM_CHUNK = 7  # scratch: per-chunk row-sum before folding into l
CB_KV_MASK = 8  # last-KV-tile softmax padding mask (0 valid cols, -inf pad cols)
CB_OUT = 16
CB_Q_SCALED = 24
CB_SCORES = 25
CB_EXP = 26
CB_ROW_MAX = 27
CB_ROW_SUM = 28
CB_PV = 29
CB_OUT_ACCUM = 30
CB_CORR = 31


def _ceil_div(a, b):
    return (a + b - 1) // b


def _chunk_size(axis_t, target):
    """Largest chunk <= target that avoids an unsafe partial last chunk.

    Selection order:
      1. Largest EXACT DIVISOR of axis_t in (1, target]. A whole chunk on every
         step means NO partial last chunk, so the cb_scores / cb_exp ring never
         carries a fractional offset across work units. This is the common case for
         every real shape (tile-counts are almost always composite: powers of two,
         296 = 8·37, …) and is what the flagged perf shape rides (Skv_t=296 -> 8).
      2. Prime tile-count > target (e.g. Skv_t=101): no divisor in (1, target], so a
         partial last chunk is unavoidable if we want to beat the 1-tile collapse.
         Fall back to R1b's coarse chunk with a straddle-safe remainder (`rem | chunk`,
         i.e. `2*rem <= chunk`, which keeps the within-work-unit reduce/exp read
         window in-bounds): Skv_t=101 -> 5 (rem 1). This branch is reachable only by
         shapes with no golden cell AND a single work unit per core (verified), so the
         cross-work-unit ring carry below never triggers for them.

    Why divisors are PREFERRED (R3a): R1b used a coarse chunk + partial for every
    non-divisible tile-count and guarded only the *within*-work-unit ring straddle
    (`rem | chunk`). But a partial last chunk pushes `sq_valid*rem` tiles — a
    FRACTION of the `Sq_chunk_t*Skv_chunk_t`-page ring — so when a core runs >1 work
    unit (total_work > grid), the cb_scores/cb_exp read+write pointers start the next
    work unit mid-ring and the reduce's linear window straddles the wrap -> garbage
    (pcc≈0, rms=inf). R3a's coarser target (8) made the L1 shrink land previously-
    clean shapes (e.g. Skv_t=128, which 4 divides) on a partial chunk (6), exposing
    that latent cross-work-unit carry. Preferring a divisor removes partials for
    every shape that has one, so the ring realigns to slot 0 after each work unit.
    """
    hi = min(axis_t, target)
    # 1. largest exact divisor <= target (no partial chunk)
    for c in range(hi, 1, -1):
        if axis_t % c == 0:
            return c
    # 2. prime tile-count > target: R1b straddle-safe coarse + partial (avoids 1)
    for c in range(hi, 1, -1):
        rem = axis_t % c
        if c % rem == 0:  # rem != 0 here (no divisor found above)
            return c
    return 1


def _working_set_bytes(sq_chunk_t, skv_chunk_t, dt, kv_depth, out_depth, has_mask, in_bytes, out_bytes, interm_bytes):
    """Per-core L1 working set. R2: input/output tile bytes follow the tensor dtype
    (fp32 doubles vs bf16, bf8b halves) and the softmax intermediates
    (cb_scores/cb_exp) follow `interm_bytes` (fp32 under fp32-DEST accumulation)."""
    bf16 = ttnn.tile_size(ttnn.bfloat16)
    fp32 = ttnn.tile_size(ttnn.float32)
    total = 0
    total += sq_chunk_t * dt * in_bytes  # q_in
    total += skv_chunk_t * dt * kv_depth * in_bytes  # k_in
    total += skv_chunk_t * dt * kv_depth * in_bytes  # v_in
    if has_mask:
        total += sq_chunk_t * skv_chunk_t * kv_depth * in_bytes  # mask_in
    total += 2 * bf16  # scaler + scale (always bf16, reader-filled)
    total += sq_chunk_t * dt * bf16  # q_scaled (bf16 regardless of input dtype)
    total += sq_chunk_t * skv_chunk_t * interm_bytes  # scores
    total += sq_chunk_t * skv_chunk_t * interm_bytes  # exp
    total += sq_chunk_t * dt * out_depth * out_bytes  # out
    total += (sq_chunk_t * 5) * fp32  # row_max, row_sum, corr, m_new, sum_chunk
    total += (sq_chunk_t * dt * 2) * fp32  # pv, out_accum
    return total


def _fit_l1(sq_t, skv_t, dt, has_mask, in_bytes, out_bytes, interm_bytes):
    """Compute the three block knobs once; shrink Skv_chunk_t -> KV_DEPTH -> Sq_chunk_t
    until the working set fits L1."""
    sq_target = SQ_CHUNK_TARGET
    skv_target = SKV_CHUNK_TARGET
    kv_depth = KV_DEPTH_DEFAULT
    out_depth = OUT_DEPTH_DEFAULT
    while True:
        sq_chunk_t = _chunk_size(sq_t, sq_target)
        skv_chunk_t = _chunk_size(skv_t, skv_target)
        need = _working_set_bytes(
            sq_chunk_t, skv_chunk_t, dt, kv_depth, out_depth, has_mask, in_bytes, out_bytes, interm_bytes
        )
        if need <= L1_BUDGET:
            break
        if skv_target > 1:
            skv_target -= 1
        elif kv_depth > 1:
            kv_depth = 1
        elif sq_target > 1:
            sq_target -= 1
        else:
            break
    return sq_chunk_t, skv_chunk_t, kv_depth, out_depth


def _f32_bits(x):
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def _cb(index, page_size, num_pages, data_format, core_grid):
    return ttnn.CBDescriptor(
        total_size=num_pages * page_size,
        core_ranges=core_grid,
        format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=data_format, page_size=page_size)],
    )


def create_program_descriptor(
    query, key, value, output_tensor, *, attn_mask=None, scale=1.0, compute_kernel_config=None
):
    q_shape = list(query.shape)
    k_shape = list(key.shape)

    B, H, S_q, D = q_shape
    H_kv = k_shape[1]
    S_kv = k_shape[-2]

    Sq_t = _ceil_div(S_q, TILE_DIM)
    Skv_t = _ceil_div(S_kv, TILE_DIM)
    Dt = _ceil_div(D, TILE_DIM)

    has_mask = attn_mask is not None
    mask_H = attn_mask.shape[1] if has_mask else 0

    # R1 (h_non_aligned): valid columns in the last S_kv tile (0 => aligned).
    # When non-zero, the last KV chunk's boundary tile carries an additive -inf
    # mask on its padding columns so they fall out of the softmax reduce.
    skv_partial = S_kv % TILE_DIM
    has_kv_pad = skv_partial != 0

    # R2: precision surface. Input/output CB tile bytes follow the tensor dtype;
    # the softmax intermediates (cb_scores/cb_exp) are fp32 under fp32-DEST
    # accumulation (the precision lever that pushes adversarial distributions back
    # under tolerance — verification_report.md), else bf16 so the perf-flagged
    # bf16 + 16-bit-DEST regime stays byte-identical. Never bf8b: block-float
    # intermediates would compound quantization through the online softmax.
    fp32_dest = bool(getattr(compute_kernel_config, "fp32_dest_acc_en", True))
    in_bytes = ttnn.tile_size(query.dtype)
    out_bytes = ttnn.tile_size(output_tensor.dtype)
    interm_format = ttnn.float32 if fp32_dest else ttnn.bfloat16
    interm_bytes = ttnn.tile_size(interm_format)

    Sq_chunk_t, Skv_chunk_t, KV_DEPTH, OUT_DEPTH = _fit_l1(Sq_t, Skv_t, Dt, has_mask, in_bytes, out_bytes, interm_bytes)

    n_q_chunks = _ceil_div(Sq_t, Sq_chunk_t)
    n_kv_chunks = _ceil_div(Skv_t, Skv_chunk_t)
    total_work = B * H * n_q_chunks

    # DEST tile budget: fp32 accumulation halves the 8-tile bf16 budget. The matmul
    # N-subblock decomposition (out_subblock_w <= dest_limit) is derived on-device
    # (R1b) so the partial last chunk's runtime N re-derives cleanly — single source
    # of truth in the compute kernel's decomp_n().
    dest_limit = 4 if fp32_dest else 8

    # ---- Grid + work distribution ----
    device = query.device()
    grid = device.compute_with_storage_grid_size()
    num_cores, all_cores, group1, group2, per_core_1, per_core_2 = ttnn.split_work_to_cores(grid, total_work, True)

    g1_cores = ttnn.corerange_to_cores(group1, None, True) if group1.num_cores() > 0 else []
    g2_cores = ttnn.corerange_to_cores(group2, None, True) if group2.num_cores() > 0 else []
    ordered = list(g1_cores) + list(g2_cores)
    counts = [per_core_1] * len(g1_cores) + [per_core_2] * len(g2_cores)

    bf16 = ttnn.tile_size(ttnn.bfloat16)
    fp32 = ttnn.tile_size(ttnn.float32)
    in_page = query.buffer_page_size()
    out_page = output_tensor.buffer_page_size()

    # ---- Circular buffers ----
    cbs = [
        _cb(CB_Q_IN, in_page, Sq_chunk_t * Dt, query.dtype, all_cores),
        _cb(CB_K_IN, in_page, Skv_chunk_t * Dt * KV_DEPTH, query.dtype, all_cores),
        _cb(CB_V_IN, in_page, Skv_chunk_t * Dt * KV_DEPTH, query.dtype, all_cores),
        _cb(CB_SCALER, bf16, 1, ttnn.bfloat16, all_cores),
        _cb(CB_SCALE, bf16, 1, ttnn.bfloat16, all_cores),
        _cb(CB_M_NEW, fp32, Sq_chunk_t, ttnn.float32, all_cores),
        _cb(CB_SUM_CHUNK, fp32, Sq_chunk_t, ttnn.float32, all_cores),
        _cb(CB_OUT, out_page, Sq_chunk_t * Dt * OUT_DEPTH, output_tensor.dtype, all_cores),
        _cb(CB_Q_SCALED, bf16, Sq_chunk_t * Dt, ttnn.bfloat16, all_cores),
        # cb_scores / cb_exp: fp32 under fp32-DEST accumulation (softmax precision),
        # bf16 otherwise. Consumed by FPU ops (add/reduce/sub/matmul) so they are
        # NOT UnpackToDestFp32-tagged — the L1 format alone lifts the intermediate
        # from bf16 (7-bit) to fp32 (unpacks to TF32, 10-bit) through the pipeline.
        _cb(CB_SCORES, interm_bytes, Sq_chunk_t * Skv_chunk_t, interm_format, all_cores),
        _cb(CB_EXP, interm_bytes, Sq_chunk_t * Skv_chunk_t, interm_format, all_cores),
        _cb(CB_ROW_MAX, fp32, Sq_chunk_t, ttnn.float32, all_cores),
        _cb(CB_ROW_SUM, fp32, Sq_chunk_t, ttnn.float32, all_cores),
        _cb(CB_PV, fp32, Sq_chunk_t * Dt, ttnn.float32, all_cores),
        _cb(CB_OUT_ACCUM, fp32, Sq_chunk_t * Dt, ttnn.float32, all_cores),
        _cb(CB_CORR, fp32, Sq_chunk_t, ttnn.float32, all_cores),
    ]
    if has_mask:
        cbs.append(_cb(CB_MASK_IN, in_page, Sq_chunk_t * Skv_chunk_t * KV_DEPTH, query.dtype, all_cores))
    if has_kv_pad:
        # One score-block worth of mask tiles (0 everywhere except the last S_kv
        # column tiles). Streamed once per work unit on the last KV chunk.
        cbs.append(_cb(CB_KV_MASK, bf16, Sq_chunk_t * Skv_chunk_t, ttnn.bfloat16, all_cores))

    # ---- Reader kernel ----
    reader_ct = [
        B,
        H,
        H_kv,
        Sq_t,
        Skv_t,
        Dt,
        Sq_chunk_t,
        Skv_chunk_t,
        n_q_chunks,
        n_kv_chunks,
        mask_H,
        1 if has_mask else 0,
        _f32_bits(scale),
        skv_partial,
    ]
    reader_ct += ttnn.TensorAccessorArgs(query).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(key).get_compile_time_args()
    reader_ct += ttnn.TensorAccessorArgs(value).get_compile_time_args()
    reader_ct += (
        ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()
    start = 0
    for core, cnt in zip(ordered, counts):
        reader_rt[core.x][core.y] = [
            query.buffer_address(),
            key.buffer_address(),
            value.buffer_address(),
            attn_mask.buffer_address() if has_mask else 0,
            start,
            cnt,
        ]
        writer_rt[core.x][core.y] = [output_tensor.buffer_address(), start, cnt]
        # start_wu lets compute decode qc per work unit -> sq_valid (partial q-chunk).
        compute_rt[core.x][core.y] = [cnt, start]
        start += cnt

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )

    # ---- Writer kernel ----
    writer_ct = [B, H, Sq_t, Dt, Sq_chunk_t, n_q_chunks]
    writer_ct += ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args()
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )

    # ---- Compute kernel ----
    # R1b: matmul subblocks are derived on-device from the per-chunk runtime tile
    # counts (sq_valid = M, skv_valid = QKᵀ N / PV K), so only the block knobs +
    # axis tile-counts + dest_limit are threaded (no host subblock CT args).
    compute_ct = [
        Dt,
        Sq_chunk_t,
        Skv_chunk_t,
        n_kv_chunks,
        1 if has_mask else 0,
        skv_partial,
        Sq_t,
        n_q_chunks,
        Skv_t,
        dest_limit,
    ]
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        config=ttnn.ComputeConfigDescriptor(
            # R2: full compute-config surface threaded from the caller's config. The
            # defaults reproduce default_compute_kernel_config() (HiFi4 + fp32 DEST +
            # no approx + half-sync) so callers passing nothing see identical results.
            math_fidelity=getattr(compute_kernel_config, "math_fidelity", ttnn.MathFidelity.HiFi4),
            fp32_dest_acc_en=fp32_dest,
            math_approx_mode=bool(getattr(compute_kernel_config, "math_approx_mode", False)),
            dst_full_sync_en=bool(getattr(compute_kernel_config, "dst_full_sync_en", False)),
        ),
    )

    descriptor = ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
    ordered_inputs = [query, key, value] + ([attn_mask] if has_mask else [])
    return descriptor, ordered_inputs
