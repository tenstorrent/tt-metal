# SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
# SPDX-License-Identifier: Apache-2.0

"""ProgramDescriptor for scaled_dot_product_attention (Flash Attention).

Work unit = one (batch b, head h, Q-block i) triple producing one
q_chunk_t x D_t output block. q_chunk_t == k_chunk_t == 1 tile: each work
unit reads one tile-row of Q (D_t tiles), iterates over S_kv_t KV blocks
(one tile-row of K/V each), and runs the online-softmax recurrence in the
compute kernel. The full S_q x S_kv score matrix is never materialized —
the score / prob CBs are a single tile.

Distribution: total_units = B * H * S_q_t spread contiguously across the
compute grid; each core gets [start_unit, num_units).
"""

import math
import struct
from pathlib import Path

import ttnn

KERNEL_DIR = Path(__file__).parent / "kernels"
TILE_DIM = 32

# CB indices (must match the kernels).
CB_Q_IN = 0
CB_K_IN = 1
CB_V_IN = 2
CB_MASK_IN = 3
CB_KV_PAD_MASK = 6  # on-device -inf column mask for the last KV tile (S_kv non-aligned)
CB_CAUSAL_MASK = 7  # on-device triangular bias (causal only); reader-filled once
CB_SCALE = 8
CB_SCALER_MAX = 9
CB_SCALER_SUM = 15
CB_MAX = 10
CB_MAX_PREV = 11
CB_CORR = 12
CB_L = 13
CB_L_BLOCK = 14
CB_M_BLK = 23
CB_QK = 24
CB_P = 25
CB_O_ACC = 26
CB_PV = 27
CB_O_TMP = 28
CB_OUT = 16


def _f32_bits(x: float) -> int:
    return struct.unpack("I", struct.pack("f", float(x)))[0]


def create_program_descriptor(
    query: ttnn.Tensor,
    key: ttnn.Tensor,
    value: ttnn.Tensor,
    attn_mask,
    output_tensor: ttnn.Tensor,
    *,
    scale: float,
    is_causal: bool = False,
    compute_kernel_config=None,
) -> ttnn.ProgramDescriptor:
    device = query.device()

    # --- Resolve compute config ------------------------------------------------
    # When no compute_kernel_config is passed we synthesize defaults below.
    # fp32 DEST accumulation is always on by default (the online-softmax
    # recurrence needs it). math_fidelity is dtype-aware (see the else branch):
    # bf16/bf8b keep Phase-0 HiFi2 — so the pre-existing bf16 path is byte-
    # identical to Phase 0 — while fp32 defaults to HiFi4. A caller passing a
    # compute_kernel_config gets exactly what they specify.
    if compute_kernel_config is not None:
        math_fidelity = compute_kernel_config.math_fidelity
        fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en
        math_approx_mode = compute_kernel_config.math_approx_mode
        dst_full_sync_en = compute_kernel_config.dst_full_sync_en
    else:
        fp32_dest_acc_en = True
        math_approx_mode = False
        dst_full_sync_en = False
        # Default math_fidelity is dtype-aware:
        #   * bf16 / bf8b -> HiFi2 (Phase-0 default; HiFi4 is known-bad with
        #     bf16 + fp32_dest_acc on Wormhole B0, and block-float is
        #     lower-precision regardless).
        #   * fp32 -> HiFi4. fp32 matmul operands are truncated to TF32 in
        #     srcA/srcB (the unavoidable FP32->TF32 drop), so at HiFi2 fp32 only
        #     reaches ~bf16 precision and misses fp32's tight RMS target. HiFi4's
        #     multi-pass matmul recovers the lost mantissa bits. This is a NEW
        #     default for a newly-supported dtype — it does not change the
        #     pre-existing bf16 behavior. A caller can still override via
        #     compute_kernel_config.
        math_fidelity = ttnn.MathFidelity.HiFi4 if query.dtype == ttnn.float32 else ttnn.MathFidelity.HiFi2

    b, h, s_q, d = (int(x) for x in query.shape)
    h_kv = int(key.shape[1])  # K/V heads (GQA/MQA: < h; MHA: == h)
    s_kv = int(key.shape[-2])
    mask_h = int(attn_mask.shape[1]) if attn_mask is not None else 1
    has_mask = attn_mask is not None

    D_t = (d + TILE_DIM - 1) // TILE_DIM
    S_q_t = (s_q + TILE_DIM - 1) // TILE_DIM
    S_kv_t = (s_kv + TILE_DIM - 1) // TILE_DIM

    # Non-tile-aligned handling (Refinement 4). The last tile along a partial
    # dimension carries padding from from_torch's tilization (NOT guaranteed
    # zero). Two independent edges:
    #   * d_valid  = D % 32  — partial last D tile. QK^T contracts over D, so
    #     Q/K's padded D columns must be ZEROED in the reader (0*x = 0); else
    #     garbage padding pollutes the dot product.
    #   * kv_valid = S_kv % 32 — partial last KV tile. The padded key columns of
    #     the score block would otherwise enter the softmax (score ~ 0, not
    #     -inf) and inflate the running max/sum. An on-device -inf column mask
    #     is added to the score on the last KV block (j == S_kv_t-1).
    # S_q % 32 (partial query rows) needs nothing: rows are independent in every
    # tile op and the padded output rows are dropped on readback.
    d_valid = d % TILE_DIM
    kv_valid = s_kv % TILE_DIM
    # bf16=2B, fp32=4B per element (bf8b non-aligned is an op EXCLUSION, never
    # reaches here). Used by the reader's column/row zeroing of the last tile.
    elem_bytes = 4 if query.dtype == ttnn.float32 else 2

    # --- Two-pass softmax gate (Refinement 6) -----------------------------
    # The online-softmax recurrence accumulates SFPU-exp rounding across the KV
    # blocks (error ~ sqrt(num_kv_blocks); see changelog R6). At fp32 the tight
    # golden rms target (0.02) is breached only at the longest context
    # (S_kv = 8192 → 256 blocks: device rms 0.0284). For that regime we switch
    # to a NON-online two-pass softmax: pass 1 finds the global per-row max (no
    # exp), pass 2 recomputes the scores and evaluates exp ONCE per element with
    # the final max, accumulating l/O by plain addition (no per-block
    # correction). Host simulation (calibrated to the device's 0.0284) puts the
    # two-pass rms at ~0.004 — clears 0.02 with >4x margin.
    #
    # Gated narrowly so the binding online-softmax path is byte-identical for
    # every other cell:
    #   * dtype == fp32      — bf16/bf8b have looser targets the online path
    #                          already meets; never switch them.
    #   * S_kv_t > 128       — only S_kv > 4096 (i.e. S=8192, 256 blocks) breaches
    #                          the target. S=4096 (128 blocks, rms 0.0151) passes
    #                          online, so it is left on the online path.
    #   * not is_causal      — causal processes ~half the KV blocks, so its
    #                          effective accumulation stays under target even at
    #                          S=8192; keep its (block-skipping) online path.
    #   * not has_mask       — the failing cells are MHA/no-mask; custom-mask
    #                          long-context fp32 already passes online. Keeping
    #                          masks on the online path avoids re-applying them
    #                          in both passes for no benefit.
    # Two-pass re-reads K from DRAM (pass 1 + pass 2) — per-core memory stays
    # O(1) (no score materialization), so no L1/OOM interaction with Refinement 5.
    two_pass = query.dtype == ttnn.float32 and S_kv_t > 128 and not is_causal and not has_mask

    total_units = b * h * S_q_t

    # --- Work distribution over the compute grid ---
    grid = device.compute_with_storage_grid_size()
    max_cores = grid.x * grid.y
    num_cores = min(total_units, max_cores)
    cores = ttnn.grid_to_cores(num_cores, grid.x, grid.y, True)

    base = total_units // num_cores
    rem = total_units % num_cores

    all_cores = ttnn.CoreRangeSet([ttnn.CoreRange(c, c) for c in cores])

    # --- Circular-buffer data formats (derived from dtype + compute config) ---
    # ttnn dtypes double as CB data formats here. Three roles:
    #   * input-side  (Q/K/V/mask) → the input tensor dtype (bf16/fp32/bf8b).
    #   * output-side (cb_out)     → the output tensor dtype.
    #   * intermediate / accumulator (running m_i, l_i, O_i, their scratch, AND
    #     the score/prob blocks) → fp32 when fp32_dest_acc_en, else input dtype.
    # The online-softmax recurrence keeps its running accumulators in fp32:
    # packing them back to a lower-precision format between KV blocks compounds
    # rounding across the KV loop (error grows with S / num_kv_blocks) — see
    # op_design.md Key Risks ("Numerical exactness requires fp32 DEST
    # accumulation"). Keeping the score/prob blocks (cb_qk/cb_p) at fp32 too is
    # the score-path precision lever folded in from the verifier's deferred
    # observation #2 (lifts the sign-biased / low-variance canaries). This
    # replaces the Phase-0 hard-coded f32 accumulator formats: when a caller
    # turns fp32_dest_acc_en off, the intermediates follow the input dtype.
    #
    # No CB is tagged UnpackToDestFp32: every intermediate/accumulator CB feeds
    # at least one FPU op (matmul / reduce / FPU binary), and an
    # UnpackToDestFp32-tagged CB cannot participate in any FPU op (it bypasses
    # srcA/srcB). The fp32 storage already gives the precision win; the FPU
    # inputs land in TF32 regardless, which is the unavoidable srcA/srcB drop.
    input_fmt = query.dtype
    out_fmt = output_tensor.dtype
    if fp32_dest_acc_en:
        accum_fmt = ttnn.float32
    elif input_fmt == ttnn.bfloat8_b:
        # Block-float (16 values share an exponent) is unusable for the online-
        # softmax running stats: cb_max / cb_l hold one valid value per row and
        # cb_qk / cb_p hold raw scores / probabilities — bf8b storage collapses
        # them (PCC -> 0). Floor bf8b intermediates to bf16 when fp32 DEST acc is
        # off. (With the default fp32_dest_acc_en=True this branch is dead.)
        accum_fmt = ttnn.bfloat16
    else:
        accum_fmt = input_fmt

    def cb(index, num_pages, fmt=input_fmt):
        page = ttnn.tile_size(fmt)
        return ttnn.CBDescriptor(
            total_size=num_pages * page,
            core_ranges=all_cores,
            format_descriptors=[ttnn.CBFormatDescriptor(buffer_index=index, data_format=fmt, page_size=page)],
        )

    # --- CB inventory as (index, num_pages, fmt) specs ---------------------
    # The seven D_t-scaling CBs (cb_q/k/v_in, cb_o_acc, cb_pv, cb_o_tmp, cb_out)
    # dominate the per-core L1 footprint: each is 2*D_t pages and, on the fp32
    # path, 4 B/elem (4096 B/tile). At D=1024 (D_t=32) the double-buffered fp32
    # set totals ~1.82 MB > the 1.5 MB L1 budget, so the 4 `Q1x1x128x1024` fp32
    # golden cells throw `program.cpp:1450` (statically-allocated CBs beyond max
    # L1). bf16/bf8b (half/quarter the bytes) and small-D fp32 fit double-
    # buffered. See Refinement 5 / changelog.
    cb_specs = [
        (CB_Q_IN, 2 * D_t, input_fmt),  # Q block, held across KV loop (double-buffered)
        (CB_K_IN, 2 * D_t, input_fmt),  # K block, streamed
        (CB_V_IN, 2 * D_t, input_fmt),  # V block, streamed
        # Reduce scalers are bf16-packed by prepare_reduce_scaler — always bf16.
        (CB_SCALE, 1, ttnn.bfloat16),
        (CB_SCALER_MAX, 1, ttnn.bfloat16),
        (CB_SCALER_SUM, 1, ttnn.bfloat16),
        (CB_MAX, 2, accum_fmt),  # running max m_i (persists across KV loop)
        (CB_MAX_PREV, 2, accum_fmt),
        (CB_CORR, 2, accum_fmt),
        (CB_L, 2, accum_fmt),  # running sum l_i (persists)
        (CB_L_BLOCK, 2, accum_fmt),
        (CB_M_BLK, 2, accum_fmt),
        (CB_QK, 2, accum_fmt),  # score block S = Q.Kᵀ (precision lever)
        (CB_P, 2, accum_fmt),  # prob block P = exp(S - m)
        (CB_O_ACC, 2 * D_t, accum_fmt),  # running output O_i (persists)
        (CB_PV, 2 * D_t, accum_fmt),
        (CB_O_TMP, 2 * D_t, accum_fmt),
        (CB_OUT, 2 * D_t, out_fmt),
    ]
    if has_mask:
        cb_specs.append((CB_MASK_IN, 2, input_fmt))  # mask block: input dtype (see note above)
    if kv_valid != 0:
        # On-device -inf column mask for the partial last KV tile. element
        # (r,c) = 0 if c < kv_valid else -inf (same for every row). Reader fills
        # it once; compute adds it to the score block on the last KV block
        # (j == S_kv_t-1), composing additively with the custom / causal masks.
        # bf16 is exact for {0,-inf} and mirrors cb_scale / cb_causal_mask.
        cb_specs.append((CB_KV_PAD_MASK, 1, ttnn.bfloat16))
    if is_causal:
        # On-device triangular bias. q_chunk_t == k_chunk_t == 1 and causal
        # requires S_q == S_kv, so the diagonal-straddling block is always
        # j == qi and its per-element mask (element (r,c) = 0 if c <= r else
        # -inf) is the SAME constant tile for every work unit. The reader
        # generates it once into this CB; the compute adds it to the score
        # block only on the diagonal KV block. bf16 is exact for {0, -inf} and
        # mirrors the cb_scale format (a bf16 second operand to an fp32 score
        # block — same mixed-format add as the scale mul).
        cb_specs.append((CB_CAUSAL_MASK, 1, ttnn.bfloat16))

    # --- L1 budget fit: single-buffer D_t-scaling CBs on the large-D fp32 path
    # (Refinement 5). Demoting a CB from 2*D_t -> 1*D_t pages halves its
    # footprint. We demote in a SAFE priority order — least pipelining cost
    # first — only as far as needed to fit, so shapes that already fit (all
    # bf16/bf8b, small-D fp32) are left byte-identical to Refinement 4:
    #   1. cb_o_acc / cb_pv / cb_o_tmp — compute->compute (intra-compute) CBs.
    #      Producer and consumer are the same compute thread running
    #      sequentially, so the "2 pages for pipelining" is fictitious (see
    #      /memory-budget-metal §4.2): the consumer pops the block before the
    #      next producer reserves it, so 1*D_t suffices with ZERO pipelining
    #      loss. cb_o_tmp is pure scratch for the corr*O_i block-bcast.
    #   2. cb_out — compute->writer. Single-buffering serializes one handoff
    #      per work unit (modest, per-work-unit cost), not in the KV hot loop.
    #   3. cb_q_in — reader->compute, held across the KV loop. Single-buffering
    #      forgoes prefetching the NEXT work unit's Q during the current KV loop
    #      (modest). Demoted last; for the supported shapes it is never reached.
    # cb_k_in / cb_v_in are NEVER demoted: they stream per KV block in the hot
    # loop, where double-buffering is real reader/compute pipelining.
    L1_BUDGET = 1499136  # bytes; matches the program.cpp static-CB ceiling
    SAFETY_MARGIN = 32 * 1024
    SINGLE_BUFFER_PRIORITY = [CB_O_ACC, CB_PV, CB_O_TMP, CB_OUT, CB_Q_IN]
    demoted = set()

    def _footprint():
        total = 0
        for idx, num_pages, fmt in cb_specs:
            pages = D_t if idx in demoted else num_pages
            total += pages * ttnn.tile_size(fmt)
        return total

    for idx in SINGLE_BUFFER_PRIORITY:
        if _footprint() <= L1_BUDGET - SAFETY_MARGIN:
            break
        demoted.add(idx)
    # If still over budget after exhausting the priority list, the allocator
    # raises program.cpp:1450 — the loud OOM signal is intentional (no silent
    # EXCLUSION / shape-size bucketing per the Refinement 5 contract).

    cbs = [cb(idx, (D_t if idx in demoted else num_pages), fmt) for idx, num_pages, fmt in cb_specs]

    # --- Reader CT args ---
    # h is H_q (Q/output heads); h_kv is the K/V head count. For GQA/MQA the
    # reader remaps each Q head to KV head h_q // (H_q / h_kv). MHA → h_kv == h.
    reader_ct = [
        D_t,
        S_q_t,
        S_kv_t,
        h,
        mask_h,
        1 if has_mask else 0,
        _f32_bits(scale),
        h_kv,
        1 if is_causal else 0,
        d_valid,  # partial last-D-tile valid columns (0 = D tile-aligned)
        kv_valid,  # partial last-KV-tile valid columns (0 = S_kv tile-aligned)
        elem_bytes,  # element size for the reader's last-tile zeroing (2=bf16, 4=fp32)
        1 if two_pass else 0,  # Refinement 6: stream K twice for the fp32 two-pass path
    ]
    reader_ct.extend(ttnn.TensorAccessorArgs(query).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(key).get_compile_time_args())
    reader_ct.extend(ttnn.TensorAccessorArgs(value).get_compile_time_args())
    reader_ct.extend(
        ttnn.TensorAccessorArgs(attn_mask).get_compile_time_args()
        if has_mask
        else ttnn.TensorAccessorArgs().get_compile_time_args()
    )

    # --- Writer CT args ---
    writer_ct = [D_t, S_q_t, h]
    writer_ct.extend(ttnn.TensorAccessorArgs(output_tensor).get_compile_time_args())

    # --- Compute CT args ---
    # is_causal + S_q_t let the compute kernel decode qi = (start_unit+u) % S_q_t
    # and (a) cap the KV loop at j <= qi (skip whole-future blocks) and (b) add
    # the on-device triangular bias on the diagonal block j == qi.
    compute_ct = [D_t, S_kv_t, 1 if has_mask else 0, 1 if is_causal else 0, S_q_t, kv_valid, 1 if two_pass else 0]

    q_addr = query.buffer_address()
    k_addr = key.buffer_address()
    v_addr = value.buffer_address()
    mask_addr = attn_mask.buffer_address() if has_mask else 0
    out_addr = output_tensor.buffer_address()

    reader_rt = ttnn.RuntimeArgs()
    writer_rt = ttnn.RuntimeArgs()
    compute_rt = ttnn.RuntimeArgs()

    start = 0
    for i, c in enumerate(cores):
        n = base + (1 if i < rem else 0)
        reader_rt[c.x][c.y] = [q_addr, k_addr, v_addr, mask_addr, start, n]
        writer_rt[c.x][c.y] = [out_addr, start, n]
        compute_rt[c.x][c.y] = [n, start]
        start += n

    reader_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_reader.cpp"),
        core_ranges=all_cores,
        compile_time_args=reader_ct,
        runtime_args=reader_rt,
        config=ttnn.ReaderConfigDescriptor(),
    )
    writer_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_writer.cpp"),
        core_ranges=all_cores,
        compile_time_args=writer_ct,
        runtime_args=writer_rt,
        config=ttnn.WriterConfigDescriptor(),
    )
    compute_kernel = ttnn.KernelDescriptor(
        kernel_source=str(KERNEL_DIR / "scaled_dot_product_attention_compute.cpp"),
        core_ranges=all_cores,
        compile_time_args=compute_ct,
        runtime_args=compute_rt,
        # Caller-controlled via compute_kernel_config (resolved above). Defaults
        # reproduce Phase 0: HiFi2 + fp32 DEST accumulation for the
        # online-softmax recurrence.
        config=ttnn.ComputeConfigDescriptor(
            math_fidelity=math_fidelity,
            fp32_dest_acc_en=fp32_dest_acc_en,
            math_approx_mode=math_approx_mode,
            dst_full_sync_en=dst_full_sync_en,
        ),
    )

    return ttnn.ProgramDescriptor(
        kernels=[reader_kernel, writer_kernel, compute_kernel],
        semaphores=[],
        cbs=cbs,
    )
