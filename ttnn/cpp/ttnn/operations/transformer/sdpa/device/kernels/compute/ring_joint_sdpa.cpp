// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include <tt-metalium/constants.hpp>
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t DHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(5);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(7);
    constexpr uint32_t logical_n = get_compile_time_arg_val(8);
    constexpr uint32_t logical_nt = get_compile_time_arg_val(9);
    constexpr uint32_t Lt = get_compile_time_arg_val(10);
    constexpr uint32_t L = get_compile_time_arg_val(11);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t ring_size = get_compile_time_arg_val(17);

    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(18);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(19);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(20);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(21);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(22);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(23);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(24);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(26);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(27);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(28);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(29);

    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(30);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(31) == 1;
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(32);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(33);
    constexpr bool uniform_dataformat = get_compile_time_arg_val(34) == 1;

    // Lightweight mask: all mask tiles live in cb_mask_in (c_3).
    // Layout: [neginf(0)] [global_n_partial?(1)] [joint_l_partial?(1 or 2)]
    // Only needed when any K/joint dimension has padding that doesn't fill a chunk.
    constexpr bool local_n_has_padding = local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool joint_has_padding = L > 0 && L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool needs_lightweight_mask = local_n_has_padding || global_n_has_padding || joint_has_padding;

    constexpr uint32_t neginf_tile_idx = 0;
    constexpr uint32_t global_n_partial_tile_idx = (global_n_partial_col > 0) ? 1 : 0;
    constexpr uint32_t joint_l_partial_tile_idx =
        (joint_l_partial_col > 0) ? (1 + (global_n_partial_col > 0 ? 1 : 0)) : 0;
    constexpr uint32_t total_mask_tiles = 1 + (global_n_partial_col > 0 ? 1 : 0) + (joint_l_partial_col > 0 ? 1 : 0);

    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_per_core = global_q_end - global_q_start;

    RingSDPAOpIndexer fused_op_indexer = RingSDPAOpIndexer(argidx);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * DHt;

    constexpr uint32_t cb_q_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_k_in = tt::CBIndex::c_1;
    constexpr uint32_t cb_v_in = tt::CBIndex::c_2;
    constexpr uint32_t cb_mask_in = tt::CBIndex::c_3;
    constexpr uint32_t cb_scale_in = tt::CBIndex::c_4;
    constexpr uint32_t cb_identity_scale_in = tt::CBIndex::c_5;
    constexpr uint32_t cb_col_identity = tt::CBIndex::c_8;
    constexpr uint32_t cb_max_in = tt::CBIndex::c_6;  // deferred norm: running max
    constexpr uint32_t cb_lse_in = tt::CBIndex::c_6;  // eager norm: LSE
    constexpr uint32_t cb_prev_out = tt::CBIndex::c_7;
    constexpr uint32_t cb_qk_im = tt::CBIndex::c_24;
    constexpr uint32_t cb_out_im_A = tt::CBIndex::c_25;
    constexpr uint32_t cb_out_im_B = tt::CBIndex::c_26;
    constexpr uint32_t cb_max_A = tt::CBIndex::c_27;
    constexpr uint32_t cb_max_B = tt::CBIndex::c_28;
    constexpr uint32_t cb_sum_A = tt::CBIndex::c_29;
    constexpr uint32_t cb_sum_B = tt::CBIndex::c_30;
    constexpr uint32_t cb_exp_max_diff = tt::CBIndex::c_31;

    constexpr uint32_t cb_out = tt::CBIndex::c_16;
    constexpr uint32_t cb_max_out = tt::CBIndex::c_17;  // deferred norm: running max
    constexpr uint32_t cb_lse_out = tt::CBIndex::c_17;  // eager norm: LSE

    // Streaming compute uses c_9 as 1-tile recip scratch for normalize_row_streaming.
    // (c_4 is used by cb_scale_in in ring joint SDPA, unlike regular SDPA.)
    constexpr uint32_t cb_recip_scratch = tt::CBIndex::c_9;

    // Deferred norm: sum save/restore CBs for multi Q-chunk DRAM round-trip.
    constexpr uint32_t cb_sum_out = tt::CBIndex::c_10;
    constexpr uint32_t cb_sum_in = tt::CBIndex::c_11;

    mm_init(cb_q_in, cb_k_in, cb_qk_im);

    // Wait once for identity scale; streaming v2 removes per-call waits inside reduce_c_row_group.
    cb_wait_front(cb_identity_scale_in, 1);

    // Wait for all lightweight mask tiles once before the ring loop.
    // Writer generates them once and they stay permanently fronted.
    if constexpr (needs_lightweight_mask) {
        cb_wait_front(cb_mask_in, total_mask_tiles);
    }

    // Precompute padded tile counts that are constant across ring iterations
    constexpr uint32_t local_n_padded_tiles =
        (local_padded_Nt % Sk_chunk_t != 0) ? (Sk_chunk_t - (local_padded_Nt % Sk_chunk_t)) : 0;
    constexpr uint32_t joint_n_padded_tiles = (Lt % Sk_chunk_t != 0) ? (Sk_chunk_t - (Lt % Sk_chunk_t)) : 0;

    RingAccumulatorState acc_state = {
        {cb_sum_A, cb_max_A, cb_out_im_A},  // prev
        {cb_sum_B, cb_max_B, cb_out_im_B},  // cur
    };

    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_indexer.seq, local_padded_Nt, logical_n / tt::constants::TILE_HEIGHT, L);

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        // DeviceZoneScopedN("RING ITER");
        uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;

        // First, find out if this ring iter processes any KV chunks.
        const uint32_t ring_iter_kv_start_tile = ring_id * local_padded_Nt;
        const uint32_t ring_iter_kv_end_tile = ring_iter_kv_start_tile + num_local_k_chunks * Sk_chunk_t;
        const uint32_t global_n_tile_id = logical_n / tt::constants::TILE_HEIGHT;
        const bool ring_iter_processes_KV_chunks = ring_iter_kv_start_tile <= global_n_tile_id;
        const bool ring_iter_does_work = ring_iter_processes_KV_chunks || (do_joint_kv && L != 0);

        if (!ring_iter_does_work) {
            continue;
        }

        const int32_t global_n_within_ring_iter = logical_n - ring_id * local_padded_N;
        // Note the > and <=. This means there is real length of logical_n within this ring iter.
        const bool global_n_is_within_ring_iter =
            global_n_within_ring_iter > 0 && global_n_within_ring_iter <= (int32_t)local_padded_N;
        const bool global_n_needs_masking = global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_global_n_mask = global_n_is_within_ring_iter && global_n_needs_masking;
        const uint32_t global_n_mask_chunk_id = global_n_within_ring_iter / (Sk_chunk_t * tt::constants::TILE_HEIGHT);

        // LOCAL N MASK
        const bool local_n_needs_masking = local_padded_Nt % Sk_chunk_t != 0;
        const uint32_t local_n_mask_chunk_id = local_padded_Nt / Sk_chunk_t;

        // JOINT L MASK
        const bool joint_n_needs_masking = L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;
        const uint32_t joint_n_mask_chunk_id = L / (Sk_chunk_t * tt::constants::TILE_HEIGHT);

        // Build lightweight mask context for this ring iteration
        LightweightMaskContext lw_mask;
        lw_mask.enabled = needs_lightweight_mask;
        lw_mask.neginf_tile_idx = neginf_tile_idx;
        lw_mask.local_n_padded_tiles = local_n_padded_tiles;
        lw_mask.joint_n_padded_tiles = joint_n_padded_tiles;
        lw_mask.global_n_partial_col = global_n_partial_col;
        lw_mask.joint_l_partial_col = joint_l_partial_col;
        lw_mask.global_n_partial_tile_idx = global_n_partial_tile_idx;
        lw_mask.joint_l_partial_tile_idx = joint_l_partial_tile_idx;
        if (ring_iter_needs_global_n_mask) {
            const uint32_t unpadded_in_chunk = global_n_within_ring_iter % (Sk_chunk_t * tt::constants::TILE_HEIGHT);
            const uint32_t valid_tiles =
                (unpadded_in_chunk + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
            lw_mask.global_n_padded_tiles = Sk_chunk_t - valid_tiles;
        }

        if constexpr (use_streaming_compute) {
            const bool is_last_ring_iter = (ring_iter == last_active_ring_iter);
            sdpa_ring_v2<
                Sq_chunk_t,
                Sk_chunk_t,
                0,  // Skt — not used for ring
                DHt,
                DHt,  // vDHt = DHt for ring
                scale_fp32,
                qk_subblock_h,
                cb_q_in,
                cb_k_in,
                cb_v_in,
                cb_qk_im,
                cb_identity_scale_in,
                cb_exp_max_diff,
                cb_col_identity,
                cb_recip_scratch,
                cb_mask_in,
                cb_scale_in,
                cb_max_in,
                cb_max_out,
                cb_prev_out,
                cb_out,
                uniform_dataformat,
                cb_out,  // cb_normalized_out — output goes directly to cb_out
                cb_sum_out,
                cb_sum_in>(
                global_q_start,
                global_q_end,
                num_kv_chunks,
                ring_iter,
                ring_id,
                num_local_k_chunks,
                local_padded_Nt,
                logical_nt,
                ring_iter_needs_global_n_mask,
                ring_iter_needs_joint_n_mask,
                local_n_needs_masking,
                global_n_mask_chunk_id,
                local_n_mask_chunk_id,
                joint_n_mask_chunk_id,
                acc_state,
                is_last_ring_iter,
                q_per_core,
                lw_mask);
        } else {
            sdpa_ring<cb_qk_im, cb_identity_scale_in, cb_scale_in, Sq_chunk_t, Sk_chunk_t, DHt, scale_fp32>(
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
                global_q_start,
                global_q_end,
                0,
                num_kv_chunks,
                q_chunk_tiles,
                k_chunk_tiles,
                qk_chunk_tiles,
                out_chunk_tiles,
                ring_iter,
                ring_id,
                num_local_k_chunks,
                local_padded_Nt,
                logical_nt,
                ring_iter_needs_global_n_mask,
                ring_iter_needs_joint_n_mask,
                local_n_needs_masking,
                global_n_mask_chunk_id,
                local_n_mask_chunk_id,
                joint_n_mask_chunk_id,
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
                cb_lse_in,
                cb_lse_out,
                cb_prev_out,
                cb_out,
                lw_mask);
        }
    }
}
