// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 fork of operations/transformer/sdpa/device/kernels/compute/sdpa.cpp. The main-tree
// original still serves RingDistributedSDPADeviceOperation; this fork is bound by
// the quasar SDPAProgramFactory (ttnn::prim::qsr).

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "compute_common.hpp"
#include "compute_streaming.hpp"

void kernel_main() {
    [[maybe_unused]] constexpr auto B = get_arg(args::B);
    [[maybe_unused]] constexpr auto NQH = get_arg(args::NQH);
    [[maybe_unused]] constexpr auto NKH = get_arg(args::NKH);
    constexpr auto Skt = get_arg(args::Skt);
    constexpr auto DHt = get_arg(args::DHt);
    constexpr auto vDHt = get_arg(args::vDHt);
    constexpr auto Sq_chunk_t = get_arg(args::Sq_chunk_t);
    constexpr auto q_num_chunks = get_arg(args::q_num_chunks);
    constexpr auto Sk_chunk_t = get_arg(args::Sk_chunk_t);
    constexpr auto k_num_chunks = get_arg(args::k_num_chunks);

    constexpr auto qk_in0_block_w = get_arg(args::qk_in0_block_w);
    constexpr auto qk_subblock_w = get_arg(args::qk_subblock_w);
    constexpr auto qk_subblock_h = get_arg(args::qk_subblock_h);
    constexpr auto qk_in0_num_subblocks = get_arg(args::qk_in0_num_subblocks);
    constexpr auto qk_in1_num_subblocks = get_arg(args::qk_in1_num_subblocks);
    constexpr auto qk_num_blocks = get_arg(args::qk_num_blocks);
    constexpr auto out_in0_block_w = get_arg(args::out_in0_block_w);
    constexpr auto out_subblock_w = get_arg(args::out_subblock_w);
    constexpr auto out_subblock_h = get_arg(args::out_subblock_h);
    constexpr auto out_in0_num_subblocks = get_arg(args::out_in0_num_subblocks);
    constexpr auto out_in1_num_subblocks = get_arg(args::out_in1_num_subblocks);
    constexpr auto out_num_blocks = get_arg(args::out_num_blocks);

    [[maybe_unused]] constexpr auto num_cores = get_arg(args::num_cores);

    constexpr bool is_causal = get_arg(args::is_causal) == 1;
    constexpr bool use_provided_mask = get_arg(args::use_provided_mask) == 1;
    constexpr bool use_padded_mask = get_arg(args::use_padded_mask) == 1;
    constexpr bool is_chunked = get_arg(args::is_chunked) == 1;
    constexpr uint32_t scale_fp32 = get_arg(args::scale_fp32);
    constexpr uint32_t sliding_window_size = get_arg(args::sliding_window_size);
    constexpr bool use_attention_sink = get_arg(args::use_attention_sink) == 1;
    constexpr bool use_streaming_compute = get_arg(args::use_streaming_compute) == 1;
    constexpr uint32_t valid_Skt = get_arg(args::valid_Skt);
    constexpr uint32_t k_partial_col = get_arg(args::k_partial_col);
    // Zigzag remap flag drives the external remap_q_index call on the flat B*NQH*q_num_chunks range.
    constexpr bool use_zigzag_balancing = get_arg(args::use_zigzag_balancing) == 1;
    // Windowed K-range narrowing: per-Q-chunk [k_lo, k_hi) arrives from the reader over a ctrl CB.
    constexpr bool use_windowed_narrowing = get_arg(args::use_windowed_narrowing) == 1;

    const uint32_t core_id = get_arg(args::core_id);
    const uint32_t num_phases = get_arg(args::num_phases);
    const uint32_t use_chunk_start_idx_tensor = get_arg(args::use_chunk_start_idx_tensor);
    uint32_t chunked_q_chunk_offset_phase_1 = get_arg(args::chunked_q_chunk_offset_phase_1);
    uint32_t chunked_q_chunk_offset_phase_2 = get_arg(args::chunked_q_chunk_offset_phase_2);

    // Global Q scheduling args.
    const uint32_t global_q_start = get_arg(args::global_q_start);
    const uint32_t global_q_count = get_arg(args::global_q_count);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // Named DFB handles. Conditionally-bound buffers alias to a bound placeholder (q_in) where the host
    // does not bind them; those paths never touch the buffer, so the placeholder is inert.
    constexpr auto dfb_q_in = dfb::q_in;
    constexpr auto dfb_k_in = dfb::k_in;
    constexpr auto dfb_v_in = dfb::v_in;
    constexpr auto dfb_identity_scale_in = dfb::identity_scale_in;
    constexpr auto dfb_col_identity = dfb::col_identity;
    constexpr auto dfb_out = dfb::out;
    constexpr auto dfb_qk_im = dfb::qk_im;
    constexpr auto dfb_out_im_A = dfb::out_im_A;
    constexpr auto dfb_out_im_B = dfb::out_im_B;
    constexpr auto dfb_max_A = dfb::max_A;
    constexpr auto dfb_max_B = dfb::max_B;
    constexpr auto dfb_sum_A = dfb::sum_A;
    constexpr auto dfb_sum_B = dfb::sum_B;
    constexpr auto dfb_exp_max_diff = dfb::exp_max_diff;
#ifdef HAS_MASK
    constexpr auto dfb_mask_in = dfb::mask_in;
#else
    constexpr auto dfb_mask_in = dfb::q_in;  // placeholder; mask path inactive
#endif
#ifdef USE_ATTENTION_SINK
    constexpr auto dfb_attention_sink = dfb::attention_sink;
#else
    constexpr auto dfb_attention_sink = dfb::q_in;  // placeholder; sink path inactive
#endif
#ifdef FLEXIBLE_CHUNKED
    constexpr auto dfb_chunk_start_idx = dfb::chunk_start_idx_compute;
#else
    constexpr auto dfb_chunk_start_idx = dfb::q_in;  // placeholder; chunk-start tensor path inactive
#endif
#ifdef USE_STREAMING_COMPUTE
    constexpr auto dfb_recip_scratch = dfb::recip_scratch;
#else
    constexpr auto dfb_recip_scratch = dfb::q_in;  // placeholder; streaming path inactive
#endif
#ifdef USE_WINDOWED_NARROWING
    constexpr auto dfb_windowed_k_range = dfb::windowed_k_range;
#else
    constexpr auto dfb_windowed_k_range = dfb::q_in;  // placeholder; narrowing path inactive
#endif

    uint32_t chunked_q_chunk_offset = 0;
    DataflowBuffer dfb_chunk_start_idx_obj(dfb_chunk_start_idx);
    DataflowBuffer dfb_identity_scale_in_obj(dfb_identity_scale_in);
    DataflowBuffer dfb_mask_in_obj(dfb_mask_in);
    compute_kernel_hw_startup<SrcOrder::Reverse>(dfb_q_in, dfb_k_in, dfb_out);
    matmul_init(dfb_q_in, dfb_k_in);

    if constexpr (is_chunked) {
        if (use_chunk_start_idx_tensor != 0) {
            dfb_chunk_start_idx_obj.wait_front(1);
            uint32_t chunk_start_idx = ckernel::read_tile_value(dfb_chunk_start_idx, 0, 0);
            dfb_chunk_start_idx_obj.pop_front(1);
            const uint32_t q_chunk_size = Sq_chunk_t * TILE_HEIGHT;
            chunked_q_chunk_offset_phase_1 = chunk_start_idx / q_chunk_size;
            if (num_phases == 2) {
                chunked_q_chunk_offset_phase_2 = chunked_q_chunk_offset_phase_1;
            }
        }
    }

    if constexpr (use_streaming_compute) {
        // Streaming SDPA v2: direct cb_qkt_im writes via cb_push_back_hold_wr_ptr.
        // No row buffers needed; a dedicated 1-tile CB is used as recip scratch.

        // Wait once for identity scale; v2 removes per-call waits inside reduce_c_row_group
        dfb_identity_scale_in_obj.wait_front(1);

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
            dfb_mask_in_obj.wait_front(lw_mask_tile_count);
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
            dfb_q_in,
            dfb_k_in,
            dfb_v_in,
            dfb_qk_im,
            dfb_identity_scale_in,
            dfb_exp_max_diff,
            dfb_col_identity,
            dfb_recip_scratch,
            dfb_out,  // normalized output goes directly to output CB
            dfb_mask_in,
            sliding_window_size,
            is_causal,
            use_attention_sink,
            dfb_attention_sink,
            use_provided_mask,
            use_windowed_narrowing,
            dfb_windowed_k_range>(
            global_q_count,
            k_num_chunks,
            dfb_out_im_A,
            dfb_out_im_B,
            dfb_max_A,
            dfb_max_B,
            dfb_sum_A,
            dfb_sum_B,
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
            lw_mask.primary_diag_tile_idx = lw_mask.causal_diag_tile_idx;
            dfb_mask_in_obj.wait_front(2);
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
                dfb_qk_im,
                dfb_identity_scale_in,
                dfb_attention_sink,
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
                use_lightweight_causal_mask,
                use_windowed_narrowing,
                dfb_windowed_k_range>(
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
                dfb_q_in,
                dfb_k_in,
                dfb_v_in,
                dfb_mask_in,
                dfb_col_identity,
                dfb_out_im_A,
                dfb_out_im_B,
                dfb_max_A,
                dfb_max_B,
                dfb_sum_A,
                dfb_sum_B,
                dfb_exp_max_diff,
                dfb_out,
                lw_mask,
                use_zigzag_balancing);
        }
    }
}
