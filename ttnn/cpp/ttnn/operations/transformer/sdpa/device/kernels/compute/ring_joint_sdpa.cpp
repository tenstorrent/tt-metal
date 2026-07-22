// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include <tt-metalium/constants.hpp>
#include "compute_common.hpp"
#include "compute_streaming.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"

template <bool kv_pad_rotation_enabled>
constexpr void assert_kv_pad_rotation_streaming_only() {
    static_assert(
        !kv_pad_rotation_enabled,
        "kv_actual_isl requires the ring-joint streaming compute path; the compute_common.hpp path selected by "
        "fp32_dest_acc_en=true is not supported.");
}

void kernel_main() {
    constexpr uint32_t B = get_compile_time_arg_val(0);
    constexpr uint32_t NH = get_compile_time_arg_val(1);
    constexpr uint32_t NHK = get_compile_time_arg_val(2);
    constexpr uint32_t DHt = get_compile_time_arg_val(3);
    constexpr uint32_t vDHt = get_compile_time_arg_val(4);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(5);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(6);
    constexpr uint32_t q_local_padded_Nt [[maybe_unused]] = get_compile_time_arg_val(7);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(8);
    constexpr uint32_t padded_Nt = get_compile_time_arg_val(9);
    constexpr uint32_t logical_n_compile = get_compile_time_arg_val(10);
    constexpr uint32_t logical_nt_compile [[maybe_unused]] = get_compile_time_arg_val(11);
    constexpr uint32_t Lt = get_compile_time_arg_val(12);
    constexpr uint32_t L = get_compile_time_arg_val(13);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t num_joint_q_chunks = get_compile_time_arg_val(15);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(16);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(17);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(18);
    constexpr uint32_t ring_size = get_compile_time_arg_val(19);
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(20);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(21);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(22);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(23);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(24);
    constexpr uint32_t qk_num_blocks = get_compile_time_arg_val(25);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(26);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(27);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(28);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(29);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(30);
    constexpr uint32_t out_num_blocks = get_compile_time_arg_val(31);

    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(32);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(33) == 1;
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(34);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(35);
    constexpr bool is_causal = get_compile_time_arg_val(36) == 1;
    constexpr bool is_balanced = get_compile_time_arg_val(37) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(38) == 1;
    constexpr bool chunked_enabled = get_compile_time_arg_val(39) == 1;
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(40);
    constexpr bool kv_pad_rotation_enabled = get_compile_time_arg_val(41) == 1;
    // Slots 42-47 are retained for compile-time arg index stability; live KV-pad Q mapping
    // and active-ring masks are runtime args below.
    constexpr uint32_t kv_pad_q_pre_wrap_start_tile_compile [[maybe_unused]] = get_compile_time_arg_val(42);
    constexpr uint32_t kv_pad_q_pre_wrap_tile_count_compile [[maybe_unused]] = get_compile_time_arg_val(43);
    constexpr uint32_t kv_pad_q_post_wrap_start_tile_compile [[maybe_unused]] = get_compile_time_arg_val(44);
    constexpr uint32_t kv_pad_q_valid_tile_count_compile [[maybe_unused]] = get_compile_time_arg_val(45);
    constexpr uint32_t active_ring_iter_mask_compile [[maybe_unused]] = get_compile_time_arg_val(46);
    constexpr uint32_t last_active_ring_iter_compile [[maybe_unused]] = get_compile_time_arg_val(47);
    constexpr bool v_shares_k_buffer = get_compile_time_arg_val(48) == 1;
    constexpr uint32_t v_cb_physical_width_t = v_shares_k_buffer ? DHt : vDHt;
    // Sparse-frames extension (SR windowed pattern). All three set together at the host or all
    // zero (feature disabled). Slots placed after the CB block (base = cb_arg_offset + 23 = 72).
    // With `sparse_frames_enabled=1`, the kernel maps each Q chunk to a single frame via integer
    // division (host requires frame_seqlen_tiles % Sq_chunk_t == 0 and % Sk_chunk_t == 0, so no
    // chunk straddles a frame boundary; chunk sizes may be smaller than the frame to fit L1) and
    // drains K/V chunks whose (q_frame, k_frame) pair is disallowed by the packed frame_allow
    // bitmap in runtime args 11..(11+31).
    constexpr bool sparse_frames_enabled = get_compile_time_arg_val(72) == 1;
    constexpr uint32_t frame_seqlen_tiles = get_compile_time_arg_val(73);
    constexpr uint32_t num_frames_padded_compile = get_compile_time_arg_val(74);
    // In-place latent-V (single-tile Q): read V straight from K^T instead of materializing it.
    // Shared with the program factory and reader via kt_inplace_v_enabled().
    constexpr bool kt_inplace_v = kt_inplace_v_enabled(v_shares_k_buffer, Sq_chunk_t);
    // Diagonal-mask tile slot is shared by the kernel's is_causal path and the chunked-prefill
    // path. kernel_is_causal is masked off by the program factory when chunked is on, so only
    // one of the two paths drives the stamp per program — but they share the CB slot layout.
    constexpr bool diag_tile_enabled = is_causal || chunked_enabled;

    // Lightweight mask: all mask tiles live in cb_mask_in.
    // Layout: [neginf(0)] [causal_diag?(1)] [global_n_partial?] [joint_l_partial?]
    constexpr bool local_n_has_padding = kv_local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n_compile % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool joint_has_padding = L > 0 && L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    constexpr bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) || diag_tile_enabled;

    constexpr uint32_t neginf_tile_idx = 0;
    constexpr uint32_t causal_diag_tile_idx = diag_tile_enabled ? 1 : 0;
    constexpr uint32_t base_partial_offset = 1 + (diag_tile_enabled ? 1 : 0);
    constexpr uint32_t global_n_partial_tile_idx = (global_n_partial_col > 0) ? base_partial_offset : 0;
    constexpr uint32_t joint_l_partial_tile_idx =
        (joint_l_partial_col > 0) ? (base_partial_offset + (global_n_partial_col > 0 ? 1 : 0)) : 0;
    constexpr uint32_t total_mask_tiles =
        1 + (diag_tile_enabled ? 1 : 0) + (global_n_partial_col > 0 ? 1 : 0) + (joint_l_partial_col > 0 ? 1 : 0);

    constexpr uint32_t q_start_idx_t =
        chunked_enabled && !kv_pad_rotation_enabled ? logical_nt_compile - q_local_padded_Nt * ring_size : 0;

    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_per_core = global_q_end - global_q_start;

    const uint32_t ring_size_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t forward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t backward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    const uint32_t kv_pad_q_pre_wrap_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t kv_pad_q_pre_wrap_tile_count = get_arg_val<uint32_t>(argidx++);
    const uint32_t kv_pad_q_post_wrap_start_tile = get_arg_val<uint32_t>(argidx++);
    const uint32_t kv_pad_q_valid_tile_count = get_arg_val<uint32_t>(argidx++);
    const uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);

    // Sparse-frames packed frame_allow bitmap (32 uint32 words). Runtime-arg slots so the same
    // kernel binary handles any windowed pattern that fits (nf_padded <= 32 -> at most 32 * 32
    // = 1024 bits). Only read when sparse_frames_enabled; when disabled the host passes zeros.
    uint32_t frame_allow_words[32];
#pragma GCC unroll 32
    for (uint32_t w = 0; w < 32; ++w) {
        frame_allow_words[w] = get_arg_val<uint32_t>(argidx++);
    }

    RingSDPAOpIndexer fused_op_indexer(
        ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    constexpr uint32_t cb_arg_offset = 49;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_scale_in = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_max_in = get_compile_time_arg_val(cb_arg_offset + 6);  // deferred norm: running max
    constexpr uint32_t cb_lse_in = cb_max_in;                                    // eager norm: LSE
    constexpr uint32_t cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_recip_scratch =
        get_compile_time_arg_val(cb_arg_offset + 9);  // 1-tile scratch for normalize_row_streaming
    constexpr uint32_t cb_sum_out = get_compile_time_arg_val(cb_arg_offset + 10);
    constexpr uint32_t cb_sum_in = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_signal = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_max_out = get_compile_time_arg_val(cb_arg_offset + 14);  // deferred norm: running max
    constexpr uint32_t cb_lse_out = cb_max_out;                                    // eager norm: LSE
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(cb_arg_offset + 15);
    constexpr uint32_t cb_out_im_A = get_compile_time_arg_val(cb_arg_offset + 16);
    constexpr uint32_t cb_out_im_B = get_compile_time_arg_val(cb_arg_offset + 17);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 18);
    constexpr uint32_t cb_max_B = get_compile_time_arg_val(cb_arg_offset + 19);
    constexpr uint32_t cb_sum_A = get_compile_time_arg_val(cb_arg_offset + 20);
    constexpr uint32_t cb_sum_B = get_compile_time_arg_val(cb_arg_offset + 21);
    constexpr uint32_t cb_exp_max_diff = get_compile_time_arg_val(cb_arg_offset + 22);

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
    matmul_init(cb_q_in, cb_k_in);

    CircularBuffer cb_identity_scale_in_obj(cb_identity_scale_in);
    CircularBuffer cb_mask_in_obj(cb_mask_in);

    // Wait once for identity scale; streaming v2 removes per-call waits inside reduce_c_row_group.
    cb_identity_scale_in_obj.wait_front(1);

    // Wait for all lightweight mask tiles once before the ring loop.
    // Writer generates them once and they stay permanently fronted.
    if constexpr (needs_lightweight_mask) {
        cb_mask_in_obj.wait_front(total_mask_tiles);
    }

    // Precompute padded tile counts that are constant across ring iterations
    constexpr uint32_t local_n_padded_tiles =
        (kv_local_padded_Nt % Sk_chunk_t != 0) ? (Sk_chunk_t - (kv_local_padded_Nt % Sk_chunk_t)) : 0;
    constexpr uint32_t joint_n_padded_tiles = (Lt % Sk_chunk_t != 0) ? (Sk_chunk_t - (Lt % Sk_chunk_t)) : 0;

    using Straddle = KCausalStraddleInfo<kv_local_padded_Nt, Sk_chunk_t>;
    constexpr bool has_straddle = Straddle::has_straddle;
    constexpr uint32_t straddle_chunk_id = Straddle::straddle_chunk_id;
    constexpr uint32_t straddle_num_padded_tiles = Straddle::straddle_num_padded_tiles;

    RingAccumulatorState acc_state = {
        {cb_sum_A, cb_max_A, cb_out_im_A},  // prev
        {cb_sum_B, cb_max_B, cb_out_im_B},  // cur
    };

    const uint32_t ring_index = fused_op_indexer.seq.ring_index;
    const uint32_t half_sequence = num_q_chunks / 2;
    const ChunkedContext chunked_context{
        q_start_idx_t,
        ring_index,
        KVPadRotationContext{
            kv_pad_q_pre_wrap_start_tile,
            kv_pad_q_pre_wrap_tile_count,
            kv_pad_q_post_wrap_start_tile,
            kv_pad_q_valid_tile_count}};
    // The first active iter starts with fresh accumulators; restoring would read stale staging.
    bool seen_active_iter = false;

    // Sparse-frames per-Q-frame accounting for correct is_first / is_last across ring iters.
    // The host-computed active_ring_iter_mask is OOB-only and does not consider sparse-frames,
    // so its "first active" and "last active" iters can be entirely drained for some Q frames.
    // Compute must instead drive is_first / is_last_k_of_last_ring_iter off per-Q-frame counts
    // of actually-processed K chunks. Precompute the expected total per Q frame here; track the
    // running processed count across ring-iter calls (persist in this outer scope).
    //
    // Q is SP-sharded across `ring_size` devices, and `frame_allow` is a broadcast global table
    // indexed by GLOBAL Q frame. `q_frame_offset` maps this device's local Q chunks to their
    // global Q-frame indices — without it, every device would look up frame_allow row 0 and
    // produce garbage. Arrays are sized to num_frames_padded_compile (max 32) and indexed by
    // global Q frame; each device only reads/writes the slots for its own Q shard.
    uint32_t q_frame_total_processed[num_frames_padded_compile > 0 ? num_frames_padded_compile : 1] = {};
    uint32_t q_frame_processed[num_frames_padded_compile > 0 ? num_frames_padded_compile : 1] = {};
    uint32_t q_frame_offset = 0;
    if constexpr (sparse_frames_enabled) {
        // Both divisions are guaranteed non-zero here: sparse_frames_enabled implies frame_seqlen
        // (tokens) is set and tile-aligned (TT_FATAL in device_operation.cpp), so frame_seqlen_tiles
        // > 0 and Sk_chunk_t > 0.
        constexpr uint32_t q_frames_per_shard = q_local_padded_Nt / frame_seqlen_tiles;
        q_frame_offset = ring_index * q_frames_per_shard;
        constexpr uint32_t chunks_per_frame = frame_seqlen_tiles / Sk_chunk_t;
        for (uint32_t qf = 0; qf < num_frames_padded_compile; ++qf) {
            uint32_t allowed_k_frames = 0;
            for (uint32_t kf = 0; kf < num_frames_padded_compile; ++kf) {
                const uint32_t bit_idx = qf * num_frames_padded_compile + kf;
                if ((frame_allow_words[bit_idx >> 5] >> (bit_idx & 31)) & 1u) {
                    allowed_k_frames++;
                }
            }
            q_frame_total_processed[qf] = allowed_k_frames * chunks_per_frame;
        }
    }

    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        uint32_t ring_id = fused_op_indexer.get_next_ring_id_and_sync();
        // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
        // still advances above so compute stays aligned with reader, writer, and all-gather.
        if (((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
            continue;
        }
        const bool do_joint_kv = ring_id == ring_size - 1;
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;
        const bool is_first_active_iter = !seen_active_iter;
        seen_active_iter = true;

        // Tile-aligned form. Chunked: real region ends on a per-chunk-region boundary
        // (k-chunk-aligned via q_local_padded_Nt % Sk_chunk_t TT_FATAL), so the per-k_chunk-
        // start skip handles it.
        const int32_t global_nt_within_ring_iter =
            static_cast<int32_t>(logical_nt) - static_cast<int32_t>(ring_id * kv_local_padded_Nt);
        const bool global_n_is_within_ring_iter =
            !chunked_enabled &&
            (global_nt_within_ring_iter > 0 && global_nt_within_ring_iter <= (int32_t)kv_local_padded_Nt);
        const bool global_n_needs_masking = (global_nt_within_ring_iter % (int32_t)Sk_chunk_t) != 0;
        const bool ring_iter_needs_global_n_mask = global_n_is_within_ring_iter && global_n_needs_masking;
        const uint32_t global_n_mask_chunk_id = global_nt_within_ring_iter / Sk_chunk_t;

        // LOCAL N MASK
        const bool local_n_needs_masking = kv_local_padded_Nt % Sk_chunk_t != 0;
        const uint32_t local_n_mask_chunk_id = kv_local_padded_Nt / Sk_chunk_t;

        // JOINT L MASK
        const bool joint_n_needs_masking = L % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
        const bool ring_iter_needs_joint_n_mask = joint_n_needs_masking && do_joint_kv;
        const uint32_t joint_n_mask_chunk_id = L / (Sk_chunk_t * tt::constants::TILE_HEIGHT);

        // is_causal: diagonal only on iter 0 (K is local-frame). Chunked: every iter (absolute coords).
        // The 9 compile-time-constant mask fields are template params (static constexpr, no stack
        // storage); only the 3 per-iter runtime fields are set below.
        RingStreamingMaskCtx<
            neginf_tile_idx,
            causal_diag_tile_idx,
            local_n_padded_tiles,
            joint_n_padded_tiles,
            global_n_partial_col,
            joint_l_partial_col,
            global_n_partial_tile_idx,
            joint_l_partial_tile_idx,
            straddle_chunk_id>
            lw_mask;
        lw_mask.is_causal = chunked_enabled || (is_causal && ring_iter == 0);
        // Straddle mask fires only on the rix>rid halved-range iters that would otherwise exclude
        // the straddle chunk. Must agree with the K-loop extension condition below.
        const bool ring_iter_needs_straddle_mask = has_straddle && is_causal && is_balanced && (ring_index > ring_id);
        lw_mask.straddle_num_padded_tiles = ring_iter_needs_straddle_mask ? straddle_num_padded_tiles : 0;
        if (ring_iter_needs_global_n_mask) {
            // Tile-aligned: valid_tiles == global_nt_within_ring_iter % Sk_chunk_t
            const uint32_t valid_tiles = global_nt_within_ring_iter % Sk_chunk_t;
            lw_mask.global_n_padded_tiles = Sk_chunk_t - valid_tiles;
        }

        const bool is_last_ring_iter = is_last_active_ring_iter(active_ring_iter_mask, ring_iter);

        // Per-ring-iter K-chunk count and Q-skip flag — shared by v1 (sdpa_ring) and v2
        // (sdpa_ring_v2) paths.
        //   rix > rid (Case 3): only sender's L half is sent — halve KV count, or extend to
        //     include the straddle chunk when it crosses the coarse-half boundary (its
        //     late-half columns are -inf-masked via lw_mask.straddle_*).
        //   rix < rid && balanced (Case 2): skip first-half (L) Q-chunks.
        uint32_t iter_num_kv_chunks = num_kv_chunks;
        if (is_causal && is_balanced && ring_index > ring_id) {
            if constexpr (has_straddle) {
                iter_num_kv_chunks = straddle_chunk_id + 1;
            } else {
                iter_num_kv_chunks /= 2;
            }
        }
        const bool skip_first_half_q = (ring_index >= ring_id ? false : is_balanced);

        if constexpr (use_streaming_compute) {
            sdpa_ring_v2<
                Sq_chunk_t,
                Sk_chunk_t,
                0,  // Skt — not used for ring
                DHt,
                vDHt,
                scale_fp32,
                qk_subblock_h,
                qk_subblock_w,
                out_subblock_h,
                out_subblock_w,
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
                cb_out,  // cb_normalized_out — output goes directly to cb_out
                cb_sum_out,
                cb_sum_in,
                cb_signal,
                needs_lightweight_mask,
                is_causal,
                is_balanced,
                chunked_enabled,
                kv_local_padded_Nt,
                q_local_padded_Nt,
                chunk_size_t,
                global_n_has_padding,
                local_n_has_padding,
                joint_has_padding,
                has_straddle && is_causal && is_balanced,
                kv_pad_rotation_enabled,
                v_cb_physical_width_t,
                v_shares_k_buffer,
                kt_inplace_v,
                sparse_frames_enabled,
                frame_seqlen_tiles,
                num_frames_padded_compile>(
                global_q_start,
                global_q_end,
                iter_num_kv_chunks,
                num_q_chunks,
                ring_iter,
                ring_id,
                num_local_k_chunks,
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
                lw_mask,
                skip_first_half_q,
                use_zigzag_balancing,
                chunked_context,
                is_first_active_iter,
                frame_allow_words,
                q_frame_total_processed,
                q_frame_processed,
                q_frame_offset);
        } else {
            assert_kv_pad_rotation_streaming_only<kv_pad_rotation_enabled>();
            sdpa_ring<
                cb_qk_im,
                cb_identity_scale_in,
                cb_scale_in,
                Sq_chunk_t,
                Sk_chunk_t,
                NH,
                DHt,
                vDHt,
                scale_fp32,
                needs_lightweight_mask,
                chunked_enabled,
                q_local_padded_Nt,
                chunk_size_t>(
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
                num_local_q_chunks,
                0,
                iter_num_kv_chunks,
                q_chunk_tiles,
                k_chunk_tiles,
                v_chunk_tiles,
                qk_chunk_tiles,
                out_chunk_tiles,
                ring_iter,
                ring_id,
                num_local_k_chunks,
                kv_local_padded_Nt,
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
                lw_mask,
                lw_mask.is_causal,
                skip_first_half_q,
                is_last_ring_iter,
                use_zigzag_balancing,
                chunked_context);
        }
    }
}
