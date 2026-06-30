// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Specialized single-head flash SDPA for the small-query MQA shape (Sq == 1 tile chunk, non-causal,
// no mask/sink/chunking). Reuses the production fused online-softmax routine (sdpa_standard) from the
// transformer SDPA compute_common.hpp so the attention math matches production speed by construction;
// this op owns only the dataflow (one core per Q head; the single KV head is read per head).
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"

void kernel_main() {
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t Skt = get_compile_time_arg_val(2);           // total K tiles (Kt)
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(3);  // Skt / Sk_chunk_t
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(4);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(5);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(6);

    constexpr uint32_t Sq_chunk_t = 1;
    constexpr uint32_t vDHt = DHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_7;
    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;
    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;  // unused

    // Block params for Sq_chunk_t == 1.
    constexpr uint32_t qk_in0_block_w = DHt;
    constexpr uint32_t qk_subblock_h = 1;
    constexpr uint32_t qk_in0_num_subblocks = 1;
    constexpr uint32_t qk_in1_num_subblocks = Sk_chunk_t / qk_subblock_w;
    constexpr uint32_t qk_num_blocks = DHt / qk_in0_block_w;
    constexpr uint32_t out_in0_block_w = Sk_chunk_t;
    constexpr uint32_t out_subblock_h = 1;
    constexpr uint32_t out_in0_num_subblocks = 1;
    constexpr uint32_t out_in1_num_subblocks = vDHt / out_subblock_w;
    constexpr uint32_t out_num_blocks = Sk_chunk_t / out_in0_block_w;

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    mm_init(cb_q_in, cb_k_in, cb_out);
    LightweightMaskContext lw_mask;  // unused (non-causal, no provided mask)

    sdpa_standard<
        cb_qk_im,
        cb_identity_scale_in,
        cb_attention_sink,
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        vDHt,
        /*use_attention_sink=*/false,
        /*is_causal=*/false,
        /*use_provided_mask=*/false,
        /*use_padded_mask=*/false,
        /*is_chunked=*/false,
        scale_fp32,
        /*sliding_window_size=*/0,
        /*lightweight_mask_enabled=*/false>(
        Skt,
        qk_in0_block_w,
        qk_subblock_w,
        qk_subblock_h,
        qk_in0_num_subblocks,
        qk_in1_num_subblocks,
        qk_num_blocks,
        out_in0_block_w,
        out_subblock_w,
        out_subblock_h,
        out_in0_num_subblocks,
        out_in1_num_subblocks,
        out_num_blocks,
        0,  // iter_q_start
        1,  // iter_q_end (one q chunk for this head)
        1,  // q_num_chunks
        0,  // local_q_start
        0,  // chunked_q_chunk_offset
        k_num_chunks,
        q_chunk_tiles,
        k_chunk_tiles,
        v_chunk_tiles,
        qk_chunk_tiles,
        out_chunk_tiles,
        cb_q_in,
        cb_k_in,
        cb_v_in,
        cb_mask_in,
        cb_col_identity,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        cb_exp_max_diff,
        cb_out,
        lw_mask);
}
