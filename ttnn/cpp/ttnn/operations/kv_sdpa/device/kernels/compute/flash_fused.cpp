// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Specialized single-head flash SDPA for the small-query MQA shape (Sq == 1 tile chunk, non-causal,
// no mask/sink/chunking). TWO-SOURCE variant: the resident prefix K/V and the new suffix K/V may use
// DIFFERENT tile heights (e.g. a 32x32 bf8 prefix + a 16x32 bf8 suffix). Because a K tile produces one
// [Sq_h x 32] score tile regardless of its own height, both phases write the same cb_qk_im and share
// the running online-softmax state (max/sum/output are keyed on the Q rows, which are uniform). We
// therefore run the flash loop over the prefix chunks (prefix geometry) and then the suffix chunks
// (suffix geometry), carrying the ping-pong max/sum/out state across both. All block/subblock sizing
// is a runtime arg so the two phases can differ. This inlines the online-softmax of sdpa_inner_loop
// (Sq_chunk_t == 1) rather than calling sdpa_standard, which reads a single K/V source.
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

// Accumulate one flash K/V chunk into the running (max/sum/out) state. Reads a full chunk from
// k_src/v_src (their own tile geometry), computes QK^T -> online-softmax exp -> QK@V, and combines
// with the previous running state. The prev/cur ping-pong CB aliases are updated in place.
template <
    uint32_t cb_qk_im,
    uint32_t cb_id_scale,
    uint32_t scale_fp32,
    uint32_t DHt,
    bool use_provided_mask,
    uint32_t cb_mask_in>
inline void flash_accumulate_chunk(
    uint32_t cb_q_in,
    uint32_t k_src,
    uint32_t v_src,
    uint32_t Sk_chunk_t,
    uint32_t qk_subblock_w,
    uint32_t out_subblock_w,
    uint32_t cb_exp_max_diff,
    uint32_t& prev_max,
    uint32_t& cur_max,
    uint32_t& prev_sum,
    uint32_t& cur_sum,
    uint32_t& prev_out,
    uint32_t& cur_out,
    uint32_t processed_k_chunks) {
    constexpr uint32_t Sq_chunk_t = 1;
    // QK^T block params: M=Sq(1), N=Sk_chunk, K=DHt. The reader transposes only the K TILE GRID
    // ([Sk_chunk, DHt] -> [DHt, Sk_chunk]); matmul's transpose flag is still required to transpose
    // each tile's contents. Without it, score column c does not correspond to K row c, so a partial
    // additive mask lands on different keys even though whole-tile masking appears correct.
    const uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_subblock_w;
    // QK@V block params: M=Sq(1), N=vDHt(DHt), K=Sk_chunk. One K-block (in0_block_w == Sk_chunk).
    const uint32_t out_in1_num_subblocks = DHt / out_subblock_w;

    /* QK = Q_CHUNK @ K_CHUNK^T -> cb_qk_im [Sq_chunk_t x Sk_chunk_t] */
    // Prefix K is commonly 32x32 while Q and suffix K are 16x32. Their data formats can be identical,
    // so a format-only reconfig leaves SrcA's tile descriptor at the geometry programmed by mm_init.
    // Explicitly update dimensions/stride as well (the gap documented in matmul.h #46769).
    reconfig_data_format<false /*to_from_int8*/, true /*is_tile_dim_reconfig_en*/>(k_src, cb_q_in);
    pack_reconfig_data_format(cb_qk_im);
    matmul_blocks(
        cb_q_in,
        k_src,
        cb_qk_im,
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        /*num_blocks=*/1,
        /*in0_num_subblocks=*/1,
        qk_in1_num_subblocks,
        /*in0_block_w=*/DHt,
        /*subblock_h=*/1,
        qk_subblock_w,
        /*transpose=*/true);

    /* QK += MASK (additive, over this chunk's column-tiles). Applied to the RAW scores, before the
       row-max, so the scale folds in via sub_exp below exactly as in the general SDPA -- that ordering
       is what makes the two numerically comparable. add_block_inplace's pop/reserve/push cycle on
       cb_qk_im re-arms the produced state for reduce_c below; it is only safe because cb_qk_im is
       single-buffered (see the program factory), otherwise the masked scores would land in the other
       buffer and the reduce would read the unmasked ones. */
    if constexpr (use_provided_mask) {
        add_block_inplace(cb_qk_im, cb_mask_in, Sq_chunk_t * Sk_chunk_t);
    }

    /* cur_max = (processed>0) ? max(prev_max, rowmax(QK)) : rowmax(QK) */
    reconfig_data_format(cb_qk_im, cb_id_scale);
    reduce_c<PoolType::MAX, ReduceDim::REDUCE_ROW, cb_qk_im, cb_id_scale, Sq_chunk_t>(
        cur_max, prev_max, Sk_chunk_t, processed_k_chunks > 0);

    /* QK = exp((QK - cur_max) * scale) in place; cur_sum = partial rowsum(QK).
       A tiny 16x32 score tile must traverse its two side-by-side faces with VectorMode::R; RC assumes
       four faces and writes through a nonexistent second face-row. */
    sub_exp_block_bcast_cols_inplace<cb_qk_im, Sq_chunk_t, scale_fp32, true, true, QK_TILE_VECTOR_MODE>(
        cur_max, cur_sum, Sk_chunk_t);

    /* OUT_IM = QK @ V_CHUNK */
    // QK is always Q-height x 32, while prefix/suffix V can switch 32x32 -> 16x32 at the phase
    // boundary. Reprogram both unpack descriptors even when their data formats are unchanged.
    reconfig_data_format<false /*to_from_int8*/, true /*is_tile_dim_reconfig_en*/>(v_src, cb_qk_im);
    pack_reconfig_data_format(cur_out);
    matmul_blocks(
        cb_qk_im,
        v_src,
        cur_out,
        Sq_chunk_t,
        DHt,
        Sk_chunk_t,
        /*num_blocks=*/1,
        /*in0_num_subblocks=*/1,
        out_in1_num_subblocks,
        /*in0_block_w=*/Sk_chunk_t,
        /*subblock_h=*/1,
        out_subblock_w,
        /*transpose=*/false);
    CircularBuffer(cb_qk_im).pop_front(Sq_chunk_t * Sk_chunk_t);
    reconfig_data_format(prev_max, cur_max);

    if (processed_k_chunks > 0) {
        /* cb_exp_max_diff = exp((prev_max - cur_max) * scale) */
        sub_exp_block<scale_fp32>(prev_max, cur_max, cb_exp_max_diff, Sq_chunk_t);
        CircularBuffer(prev_max).pop_front(Sq_chunk_t);
        /* prev_sum *= exp_max_diff ; cur_sum += prev_sum */
        mul_tiles_bcast_cols_inplace(prev_sum, cb_exp_max_diff, Sq_chunk_t);
        add_block_inplace(cur_sum, prev_sum, Sq_chunk_t);
        /* cur_out += prev_out * exp_max_diff (L1 accumulate) */
        mul_block_bcast_cols<Sq_chunk_t, DHt, false, true>(prev_out, cb_exp_max_diff, cur_out);
    }

    std::swap(prev_sum, cur_sum);
    std::swap(prev_out, cur_out);
    std::swap(prev_max, cur_max);
}

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(0);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(1);
    // Prefix (resident past K/V) phase geometry.
    constexpr uint32_t prefix_num_chunks = get_compile_time_arg_val(2);
    constexpr uint32_t prefix_Sk_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t prefix_qk_subblock_w = get_compile_time_arg_val(4);
    constexpr uint32_t prefix_out_subblock_w = get_compile_time_arg_val(5);
    // Suffix (new K/V) phase geometry.
    constexpr uint32_t suffix_num_chunks = get_compile_time_arg_val(6);
    constexpr uint32_t suffix_Sk_chunk_t = get_compile_time_arg_val(7);
    constexpr uint32_t suffix_qk_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t suffix_out_subblock_w = get_compile_time_arg_val(9);
    // Split-KV role, compile-time per core set. suffix_num_chunks (arg 6) is already the PER-CORE
    // count -- only the reducer is given the suffix -- so the loops above stay fully specialized.
    // Dense additive mask over the folded KV (index 10 -- keep is_reducer/num_children AFTER it, the
    // factory push_backs them in that order).
    constexpr bool use_provided_mask = get_compile_time_arg_val(10) == 1;
    constexpr bool is_reducer = get_compile_time_arg_val(11) == 1;
    constexpr uint32_t num_children = get_compile_time_arg_val(12);

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;      // suffix K
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;      // suffix V
    constexpr uint32_t cb_k_prefix = tt::CBIndex::c_8;  // prefix K (own tile geometry)
    constexpr uint32_t cb_v_prefix = tt::CBIndex::c_9;  // prefix V
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;   // one chunk's mask column-tiles
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    // Split-KV reduction CBs (see the program factory). Unused when kv_splits == 1.
    constexpr uint32_t cb_l_in = tt::CBIndex::c_11;                 // child sum
    constexpr uint32_t cb_m_in = tt::CBIndex::c_12;                 // child max
    constexpr uint32_t cb_out_o = tt::CBIndex::c_13;                // child out
    constexpr uint32_t cb_prev_sum_2 = tt::CBIndex::c_14;           // child sum, staged for correction
    constexpr uint32_t cb_out_accumulate_im_2 = tt::CBIndex::c_15;  // child out, staged for rescale
    constexpr uint32_t cb_exp_max_diff_2 = tt::CBIndex::c_17;       // exp(child_max - cur_max)
    constexpr uint32_t cb_partial_out = tt::CBIndex::c_18;          // this worker's (l, m, o) to send
    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;

    constexpr uint32_t Sq_chunk_t = 1;
    constexpr uint32_t vDHt = DHt;

    mm_init(cb_q_in, cb_k_in, cb_out);

    // Wait for this core's Q head once (used as in0 in every QK matmul; never popped mid-loop).
    CircularBuffer(cb_q_in).wait_front(Sq_chunk_t * DHt);

    // Ping-pong running-state aliases (mutated by the accumulate helper's std::swap).
    uint32_t prev_max = cb_max_A, cur_max = cb_max_B;
    uint32_t prev_sum = cb_sum_A, cur_sum = cb_sum_B;
    uint32_t prev_out = cb_out_im_A, cur_out = cb_out_im_B;

    uint32_t processed = 0;
    // Phase 1: resident prefix K/V (its own tile geometry, e.g. 32x32). Under split-KV this is only
    // THIS core's slice of the prefix -- the reader is given the matching tile offset.
    for (uint32_t c = 0; c < prefix_num_chunks; ++c) {
        flash_accumulate_chunk<cb_qk_im, cb_identity_scale_in, scale_fp32, DHt, use_provided_mask, cb_mask_in>(
            cb_q_in,
            cb_k_prefix,
            cb_v_prefix,
            prefix_Sk_chunk_t,
            prefix_qk_subblock_w,
            prefix_out_subblock_w,
            cb_exp_max_diff,
            prev_max,
            cur_max,
            prev_sum,
            cur_sum,
            prev_out,
            cur_out,
            processed);
        processed++;
    }
    // Phase 2: new suffix K/V (model tiny tile, e.g. 16x32). Only the reducer owns the suffix.
    for (uint32_t c = 0; c < suffix_num_chunks; ++c) {
        flash_accumulate_chunk<cb_qk_im, cb_identity_scale_in, scale_fp32, DHt, use_provided_mask, cb_mask_in>(
            cb_q_in,
            cb_k_in,
            cb_v_in,
            suffix_Sk_chunk_t,
            suffix_qk_subblock_w,
            suffix_out_subblock_w,
            cb_exp_max_diff,
            prev_max,
            cur_max,
            prev_sum,
            cur_sum,
            prev_out,
            cur_out,
            processed);
        processed++;
    }

    /* Collapse the partial row-sum into a true column vector. Every core does this BEFORE the
       cross-split merge, because correction_block treats sum/max as column vectors. */
    matmul_reduce<Sq_chunk_t>(cb_col_identity, prev_sum);

    if (!is_reducer) {
        /* Worker: stage (l, m, o) for the writer to push to this head's reducer. No normalization --
           the reducer owns the final divide. Order must match the writer's read order. */
        move_block<false>(prev_sum, cb_partial_out, Sq_chunk_t);
        move_block<false>(prev_max, cb_partial_out, Sq_chunk_t);
        move_block<false>(prev_out, cb_partial_out, Sq_chunk_t * vDHt);
        CircularBuffer(cb_q_in).pop_front(Sq_chunk_t * DHt);
        return;
    }

    /* Reducer: merge each child's partial state into the running one. The writer feeds children in
       one at a time (the 1-deep cb_l_in/cb_m_in/cb_out_o backpressure serializes it), and each merge
       is the standard online-softmax combine:
         cur_max = max(prev_max, child_max)
         cur_sum = prev_sum*exp(prev_max-cur_max) + child_sum*exp(child_max-cur_max)
         prev_out = prev_out*exp(prev_max-cur_max) + child_out*exp(child_max-cur_max)   */
    for (uint32_t child = 0; child < num_children; ++child) {
        move_block<true>(cb_l_in, cb_prev_sum_2, Sq_chunk_t);
        correction_block<scale_fp32>(
            cb_m_in,
            cb_prev_sum_2,
            cur_max,
            prev_max,
            cur_sum,
            prev_sum,
            cb_exp_max_diff,
            cb_exp_max_diff_2,
            Sq_chunk_t);
        move_block<true>(cb_out_o, cb_out_accumulate_im_2, Sq_chunk_t * vDHt);
        mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(prev_out, cb_exp_max_diff);
        mul_block_bcast_cols_inplace<Sq_chunk_t, vDHt>(cb_out_accumulate_im_2, cb_exp_max_diff_2);
        add_block_inplace<true>(prev_out, cb_out_accumulate_im_2, Sq_chunk_t * vDHt);
        CircularBuffer(prev_max).pop_front(Sq_chunk_t);
        CircularBuffer(cb_m_in).pop_front(Sq_chunk_t);
        move_block<true>(cur_max, prev_max, Sq_chunk_t);
        move_block<true>(cur_sum, prev_sum, Sq_chunk_t);
    }

    /* Reciprocal and output normalization. */
    recip_block_inplace(prev_sum, Sq_chunk_t);
    pack_reconfig_data_format(cb_out);
    mul_block_bcast_cols<Sq_chunk_t, vDHt, false, false>(prev_out, prev_sum, cb_out);

    CircularBuffer(cb_q_in).pop_front(Sq_chunk_t * DHt);
}
