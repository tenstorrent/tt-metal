// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "compute_common.hpp"
#include "compute_streaming.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NQH = get_compile_time_arg_val(1);
    constexpr uint32_t NKH = get_compile_time_arg_val(2);
    constexpr uint32_t Skt = get_compile_time_arg_val(3);
    constexpr uint32_t DHt = get_compile_time_arg_val(4);
    constexpr uint32_t vDHt = get_compile_time_arg_val(5);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(7);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(8);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(9);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(10);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(12);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(13);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(14);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(15);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(18);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(21);

    constexpr uint32_t num_cores = get_compile_time_arg_val(22);

    constexpr bool is_causal = get_compile_time_arg_val(23) == 1;
    constexpr bool use_provided_mask = get_compile_time_arg_val(24) == 1;
    constexpr bool use_padded_mask = get_compile_time_arg_val(25) == 1;
    constexpr bool is_chunked = get_compile_time_arg_val(26) == 1;
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(27);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(28);
    constexpr bool use_attention_sink = get_compile_time_arg_val(29) == 1;
    constexpr bool use_streaming_compute = get_compile_time_arg_val(30) == 1;
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(31);
    constexpr bool uniform_dataformat = get_compile_time_arg_val(32) == 1;
    constexpr uint32_t k_partial_col = get_compile_time_arg_val(33);
    // Zigzag remap flag drives the external remap_q_index call on the flat B*NQH*q_num_chunks range.
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(34) == 1;

    const uint32_t core_id = get_arg_val<uint32_t>(0);
    const uint32_t num_phases = get_arg_val<uint32_t>(1);
    const uint32_t use_chunk_start_idx_tensor = get_arg_val<uint32_t>(2);
    uint32_t chunked_q_chunk_offset_phase_1 = get_arg_val<uint32_t>(3);
    uint32_t chunked_q_chunk_offset_phase_2 = 0;
    if (num_phases == 2) {
        chunked_q_chunk_offset_phase_2 = get_arg_val<uint32_t>(4);
    }

    // Global Q scheduling args follow phase_2 slot.
    const uint32_t global_q_start = get_arg_val<uint32_t>(5);
    const uint32_t global_q_count = get_arg_val<uint32_t>(6);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_attention_sink = tt::CBIndex::c_4;
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

    constexpr uint32_t cb_chunk_start_idx = tt::CBIndex::c_8;
    uint32_t chunked_q_chunk_offset = 0;
    mm_init(cb_q_in, cb_k_in, cb_out);

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            cb_wait_front(cb_chunk_start_idx, 1);
            uint32_t chunk_start_idx = ckernel::read_tile_value(cb_chunk_start_idx, 0, 0);
            cb_pop_front(cb_chunk_start_idx, 1);
            const uint32_t q_chunk_size = Sq_chunk_t * TILE_HEIGHT;
            chunked_q_chunk_offset_phase_1 = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunked_q_chunk_offset_phase_2 = chunked_q_chunk_offset_phase_1;
            }
        }
    }

    if constexpr (use_streaming_compute) {
        // Streaming SDPA v2: direct cb_qkt_im writes via cb_push_back_hold_wr_ptr.
        // No row buffers needed. c_4 used only as 1-tile recip scratch for normalization.
        constexpr uint32_t cb_recip_scratch = tt::CBIndex::c_4;

        // Wait once for identity scale; v2 removes per-call waits inside reduce_c_row_group
        cb_wait_front(cb_identity_scale_in, 1);

        // Lightweight-mask context: writer pre-generates [neginf(0), causal_diag?, k_partial?] tiles
        // permanently fronted. Causal uses neginf + diag (2 tiles). Non-causal partial-tile K
        // uses neginf + partial (2 tiles); the per-row stamp masks the partial col + trailing neginf.
        LightweightMaskContext lw_mask;
        if constexpr (is_causal) {
            lw_mask.is_causal = true;
            lw_mask.neginf_tile_idx = 0;
            lw_mask.causal_diag_tile_idx = 1;
            cb_wait_front(cb_mask_in, 2);
        } else if constexpr (k_partial_col > 0) {
            lw_mask.global_n_partial_col = k_partial_col;
            lw_mask.global_n_partial_tile_idx = 1;  // [neginf(0), partial(1)]
            // global_n_padded_tiles = Sk_chunk_t - valid_tiles_in_last_chunk
            constexpr uint32_t last_chunk_first_tile =
                (valid_Skt > Sk_chunk_t) ? ((valid_Skt - 1) / Sk_chunk_t) * Sk_chunk_t : 0u;
            constexpr uint32_t valid_tiles_in_last_chunk = valid_Skt - last_chunk_first_tile;
            lw_mask.global_n_padded_tiles = Sk_chunk_t - valid_tiles_in_last_chunk;
            cb_wait_front(cb_mask_in, 2);
        }

        // Global Q scheduling: sdpa_standard_v2 walks the per-core flat range over
        // B*NQH*q_num_chunks chunks; the modulo inside its inner loop extracts the per-head q_chunk
        // from each flat index. num_phases==1 is pinned for streaming, so the chunked offset comes
        // from phase 1.
        sdpa_standard_v2<
            Sq_chunk_t,
            Sk_chunk_t,
            valid_Skt,
            DHt,
            vDHt,
            scale_fp32,
            qk_subblock_h,
            qk_subblock_w,
            out_subblock_h,
            out_subblock_w,
            use_padded_mask,
            cb_q_in,
            cb_k_in,
            cb_v_in,
            cb_qk_im,
            cb_identity_scale_in,
            cb_exp_max_diff,
            cb_col_identity,
            cb_recip_scratch,
            cb_out,  // normalized output goes directly to output CB
            cb_mask_in,
            uniform_dataformat,
            is_causal>(
            global_q_count,
            k_num_chunks,
            cb_out_im_A,
            cb_out_im_B,
            cb_max_A,
            cb_max_B,
            cb_sum_A,
            cb_sum_B,
            global_q_start,
            chunked_q_chunk_offset_phase_1,
            lw_mask,
            q_num_chunks,
            use_zigzag_balancing);
    } else {
        // Standard SDPA path (causal, masked, chunked, etc.)
        constexpr bool use_lightweight_causal_mask = is_causal && !use_provided_mask && (sliding_window_size == 0);

        LightweightMaskContext lw_mask;
        if constexpr (use_lightweight_causal_mask) {
            lw_mask.is_causal = true;
            lw_mask.neginf_tile_idx = 0;
            lw_mask.causal_diag_tile_idx = 1;
            cb_wait_front(cb_mask_in, 2);
        }

        for (uint32_t phase = 0; phase < num_phases; ++phase) {
            if (phase == 0) {
                chunked_q_chunk_offset = chunked_q_chunk_offset_phase_1;
            } else {
                chunked_q_chunk_offset = chunked_q_chunk_offset_phase_2;
            }

            // Global Q scheduling: sdpa_standard walks the per-core flat range over
            // B*NQH*q_num_chunks chunks; the modulo inside its inner loop extracts the per-head
            // q_chunk from each flat index.
            sdpa_standard<
                cb_qk_im,
                cb_identity_scale_in,
                cb_attention_sink,
                Sq_chunk_t,
                Sk_chunk_t,
                DHt,
                vDHt,
                use_attention_sink,
                is_causal,
                use_provided_mask,
                use_padded_mask,
                is_chunked,
                scale_fp32,
                sliding_window_size,
                use_lightweight_causal_mask>(
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
                /*iter_q_start=*/0,
                /*iter_q_end=*/global_q_count,
                q_num_chunks,
                /*local_q_start=*/global_q_start,
                chunked_q_chunk_offset,
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
                lw_mask,
                use_zigzag_balancing);
        }
    }
}
