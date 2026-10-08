// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "sdpa_block_ops.hpp"
#include "compute_streaming.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/windowed_mode.hpp"

void kernel_main() {
    // CT args 0, 7, 10-12 and 15-16 are the legacy loop's blocking; the slots are kept, as is CT arg 24
    // (use_streaming_compute), because models/demos/wormhole/bge_m3 compiles this kernel with hand-numbered args.
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t q_num_chunks = get_compile_time_arg_val(4);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t k_num_chunks = get_compile_time_arg_val(6);

    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(8);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(13);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(14);

    constexpr bool is_causal = get_compile_time_arg_val(17) == 1;
    constexpr bool use_provided_mask = get_compile_time_arg_val(18) == 1;
    constexpr bool use_padded_mask = get_compile_time_arg_val(19) == 1;
    constexpr bool is_chunked = get_compile_time_arg_val(20) == 1;
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(21);
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(22);
    constexpr bool use_attention_sink = get_compile_time_arg_val(23) == 1;
    static_assert(get_compile_time_arg_val(24) == 1, "SDPA has only its streaming compute path");
    constexpr uint32_t valid_Skt = get_compile_time_arg_val(25);
    constexpr uint32_t k_partial_col = get_compile_time_arg_val(26);
    // Zigzag remap flag drives the external remap_q_index call on the flat B*NQH*q_num_chunks range.
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(27) == 1;
    // Windowed K-range narrowing: per-Q-chunk [k_lo, k_hi) arrives from the reader over a ctrl CB.
    // Compute is mode-agnostic: windowed causal lives entirely in that range and the generated mask.
    constexpr auto windowed_mode = static_cast<WindowedMode>(get_compile_time_arg_val(28));
    constexpr bool use_windowed_narrowing = is_windowed_mode(windowed_mode);

    // Runtime args 0 (number of Q phases, always 1) and 3 (second phase's chunk offset) are unused slots.
    const uint32_t use_chunk_start_idx_tensor = get_arg_val<uint32_t>(1);
    uint32_t chunked_q_chunk_offset = get_arg_val<uint32_t>(2);
    const uint32_t global_q_start = get_arg_val<uint32_t>(4);
    const uint32_t global_q_count = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_arg_offset = 29;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 6);
    constexpr uint32_t cb_chunk_start_idx = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t cb_recip_scratch = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 9);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(cb_arg_offset + 10);
    constexpr uint32_t cb_out_im_A = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_out_im_B = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_max_B = get_compile_time_arg_val(cb_arg_offset + 14);
    constexpr uint32_t cb_sum_A = get_compile_time_arg_val(cb_arg_offset + 15);
    constexpr uint32_t cb_sum_B = get_compile_time_arg_val(cb_arg_offset + 16);
    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(cb_arg_offset + 17);
    constexpr uint32_t cb_windowed_k_range = get_compile_time_arg_val(cb_arg_offset + 18);
    CircularBuffer cb_chunk_start_idx_obj(cb_chunk_start_idx);
    CircularBuffer cb_identity_scale_in_obj(cb_identity_scale_in);
    CircularBuffer cb_mask_in_obj(cb_mask_in);
    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_out);
    matmul_init(cb_q_in, cb_k_in);

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            cb_chunk_start_idx_obj.wait_front(1);
            uint32_t chunk_start_idx = ckernel::read_tile_value(cb_chunk_start_idx, 0, 0);
            cb_chunk_start_idx_obj.pop_front(1);
            const uint32_t q_chunk_size = Sq_chunk_t * TILE_HEIGHT;
            chunked_q_chunk_offset = chunk_start_idx / q_chunk_size;
        }
    }

    // Streaming SDPA v2: direct cb_qkt_im writes via cb_push_back_hold_wr_ptr.
    // No row buffers needed; a dedicated 1-tile CB is used as recip scratch.

    // Wait once for identity scale; v2 removes per-call waits inside reduce_c_row_group
    cb_identity_scale_in_obj.wait_front(1);

    // Lightweight-mask context: writer pre-generates either [neginf, causal_diag, partial?]
    // or, for sliding, [neginf, trailing_primary, leading_prev, leading_current, trailing_next, partial?].
    // primary_diag_tile_idx is the per-layout tile used for the row-local diagonal stamp.
    LightweightMaskContext lw_mask;
    uint32_t lw_mask_tile_count = 1;
    lw_mask.neginf_tile_idx = 0;
    lw_mask.is_causal = is_causal;
    if constexpr (sliding_window_size > 0) {
        lw_mask.primary_diag_tile_idx = 1;
        lw_mask.sliding_leading_prev_tile_idx = 2;
        lw_mask.sliding_leading_tile_idx = 3;
        lw_mask.sliding_trailing_next_tile_idx = 4;
        lw_mask_tile_count = 5;
    } else if constexpr (is_causal) {
        lw_mask.causal_diag_tile_idx = lw_mask_tile_count++;
        lw_mask.primary_diag_tile_idx = lw_mask.causal_diag_tile_idx;
    }
    if constexpr (k_partial_col > 0) {
        lw_mask.global_n_partial_col = k_partial_col;
        lw_mask.global_n_partial_tile_idx = lw_mask_tile_count++;
        // global_n_padded_tiles = Sk_chunk_t - valid_tiles_in_last_chunk
        constexpr uint32_t last_chunk_first_tile =
            (valid_Skt > Sk_chunk_t) ? ((valid_Skt - 1) / Sk_chunk_t) * Sk_chunk_t : 0u;
        constexpr uint32_t valid_tiles_in_last_chunk = valid_Skt - last_chunk_first_tile;
        lw_mask.global_n_padded_tiles = Sk_chunk_t - valid_tiles_in_last_chunk;
    }
    // A user-provided dense mask is streamed per-chunk by the reader and consumed inside the
    // inner loop — it does not use the writer-generated lightweight palette, so skip this wait.
    if constexpr ((is_causal || sliding_window_size > 0 || k_partial_col > 0) && !use_provided_mask) {
        cb_mask_in_obj.wait_front(lw_mask_tile_count);
    }

    // Global Q scheduling: sdpa_standard_v2 walks the per-core flat range over
    // B*NQH*q_num_chunks chunks; the modulo inside its inner loop extracts the per-head q_chunk
    // from each flat index.
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
        sliding_window_size,
        is_causal,
        use_attention_sink,
        cb_attention_sink,
        use_provided_mask,
        use_windowed_narrowing,
        cb_windowed_k_range>(
        global_q_count,
        k_num_chunks,
        cb_out_im_A,
        cb_out_im_B,
        cb_max_A,
        cb_max_B,
        cb_sum_A,
        cb_sum_B,
        global_q_start,
        chunked_q_chunk_offset,
        lw_mask,
        q_num_chunks,
        use_zigzag_balancing);
}
