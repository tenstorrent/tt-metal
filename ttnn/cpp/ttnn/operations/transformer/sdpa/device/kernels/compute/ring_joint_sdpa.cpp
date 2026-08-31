// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

// This kernel reconfigs ~30x; inlining the LLK Src zero-flag DEFAULT configurator at each site pushes
// the program over the kernel-config buffer. Force it out-of-line here.
#define LLK_ZEROFLAG_OUTLINE 1

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
    constexpr bool use_attention_sink = get_compile_time_arg_val(49) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(50);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    // Slot 52: sharded-joint flag (appended after upstream's kv_pad_from_metadata at slot 51, which is
    // declared further down next to cb_arg_offset). When true, one L/P shard arrives per ring iteration
    // and do_joint_kv fires on every iteration rather than only the last active iteration.
    constexpr bool joint_is_sharded = get_compile_time_arg_val(52) == 1;
    // Slot 53: true (unpadded) joint length in tiles (similar to spatial logical_nt). Drives the
    // per-ring-iteration joint tail mask and the joint out-of-bounds K-chunk skip.
    constexpr uint32_t logical_lt = get_compile_time_arg_val(53);
    constexpr uint32_t v_cb_physical_width_t = v_shares_k_buffer ? DHt : vDHt;
    // In-place latent-V (single-tile Q): read V straight from K^T instead of materializing it.
    // Shared with the program factory and reader via kt_inplace_v_enabled().
    constexpr bool kt_inplace_v = kt_inplace_v_enabled(v_shares_k_buffer, Sq_chunk_t);
    // Diagonal-mask tile slot is shared by the kernel's is_causal path and the chunked-prefill
    // path. kernel_is_causal is masked off by the program factory when chunked is on, so only
    // one of the two paths drives the stamp per program — but they share the CB slot layout.
    constexpr bool diag_tile_enabled = (is_causal || chunked_enabled) && !has_sliding_window;
    // Sharded joint: num_joint_k_chunks is the per-shard count (L_local / k_chunk). Replicated: full L.
    constexpr bool has_joint_k = num_joint_k_chunks > 0;
    constexpr bool has_gathered_joint_k = joint_is_sharded && has_joint_k;

    // Per-device joint shard length in tiles (used for the per-ring-iteration joint tail boundary).
    constexpr uint32_t Lt_local = has_gathered_joint_k ? Lt / ring_size : Lt;

    // Lightweight mask: all mask tiles live in cb_mask_in.
    // Layout: [neginf(0)] [causal_diag?(1)] [global_n_partial?] [joint_l_partial?]
    constexpr bool local_n_has_padding = kv_local_padded_Nt % Sk_chunk_t != 0;
    constexpr bool global_n_has_padding = logical_n_compile % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0;
    // Joint needs a mask when either the per-device joint shard is not a whole number of K-chunks
    // (Lt_local % Sk_chunk_t != 0 -> fully-padded trailing tiles, the local_n analogue), or real tokens
    // do not fill the last real shard's chunk (logical_lt % Sk_chunk_t != 0 or a sub-tile partial
    // column, the global_n analogue).
    constexpr bool joint_has_padding =
        L > 0 && ((Lt_local % Sk_chunk_t != 0) || (logical_lt % Sk_chunk_t != 0) || (joint_l_partial_col != 0));
    constexpr bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) || diag_tile_enabled || has_sliding_window;

    constexpr uint32_t neginf_tile_idx = 0;
    constexpr uint32_t causal_diag_tile_idx = diag_tile_enabled ? 1 : 0;
    constexpr uint32_t edge_mask_tiles = has_sliding_window ? kSlidingWindowEdgeTiles : (diag_tile_enabled ? 1 : 0);
    constexpr uint32_t base_partial_offset = 1 + edge_mask_tiles;
    constexpr uint32_t global_n_partial_tile_idx = (global_n_partial_col > 0) ? base_partial_offset : 0;
    constexpr uint32_t joint_l_partial_tile_idx =
        (joint_l_partial_col > 0) ? (base_partial_offset + (global_n_partial_col > 0 ? 1 : 0)) : 0;
    constexpr uint32_t total_mask_tiles =
        1 + edge_mask_tiles + (global_n_partial_col > 0 ? 1 : 0) + (joint_l_partial_col > 0 ? 1 : 0);

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
    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    uint32_t kv_pad_q_pre_wrap_start_tile = get_arg_val<uint32_t>(argidx++);
    uint32_t kv_pad_q_pre_wrap_tile_count = get_arg_val<uint32_t>(argidx++);
    uint32_t kv_pad_q_post_wrap_start_tile = get_arg_val<uint32_t>(argidx++);
    uint32_t kv_pad_q_valid_tile_count = get_arg_val<uint32_t>(argidx++);
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpIndexer fused_op_indexer(
        ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);

    constexpr uint32_t q_chunk_tiles = Sq_chunk_t * DHt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t qk_chunk_tiles = Sq_chunk_t * Sk_chunk_t;
    constexpr uint32_t out_chunk_tiles = Sq_chunk_t * vDHt;

    // Compute fixed slot 51: trace-safe KV-pad derivation flag. Slots 52/53 are the sharded-joint
    // scalars (joint_is_sharded, logical_lt) declared near the top of the kernel, so the CB block
    // starts at 54.
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(51) == 1;
    constexpr uint32_t cb_arg_offset = 54;
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
    constexpr uint32_t cb_kv_pad_derived = get_compile_time_arg_val(cb_arg_offset + 23);
    constexpr uint32_t cb_attention_sink = get_compile_time_arg_val(cb_arg_offset + 24);

    if constexpr (kv_pad_from_metadata) {
        CircularBuffer cb_derived(cb_kv_pad_derived);
        cb_derived.wait_front(1);
        logical_nt = ckernel::read_tile_value(cb_kv_pad_derived, 0, 0);
        kv_pad_q_pre_wrap_start_tile = ckernel::read_tile_value(cb_kv_pad_derived, 0, 1);
        kv_pad_q_pre_wrap_tile_count = ckernel::read_tile_value(cb_kv_pad_derived, 0, 2);
        kv_pad_q_post_wrap_start_tile = ckernel::read_tile_value(cb_kv_pad_derived, 0, 3);
        kv_pad_q_valid_tile_count = ckernel::read_tile_value(cb_kv_pad_derived, 0, 4);
        active_ring_iter_mask = ckernel::read_tile_value(cb_kv_pad_derived, 0, 5);
        cb_derived.pop_front(1);
    }

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

    // Precompute padded tile counts that are constant across ring iterations.
    constexpr uint32_t local_n_padded_tiles =
        (kv_local_padded_Nt % Sk_chunk_t != 0) ? (Sk_chunk_t - (kv_local_padded_Nt % Sk_chunk_t)) : 0;
    // joint_n_padded_tiles is per-ring-iter (the joint tail lands in a different shard each iteration),
    // so it is resolved inside the ring loop and stamped onto lw_mask, mirroring global_n_padded_tiles.

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
    constexpr uint32_t sdpa_ring_iterations = has_sliding_window ? 1 : ring_size;
    for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
        // Sliding folds all local/halo source ranges into one synthetic local iteration.
        // The dataflow reader has already waited for the required halo completion signals.
        uint32_t ring_id = has_sliding_window ? ring_index : fused_op_indexer.get_next_ring_id_and_sync();
        // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
        // still advances above so compute stays aligned with reader, writer, and all-gather.
        if (!has_sliding_window && ((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
            continue;
        }
        // Sharded joint: one L/P shard per ring iteration — process joint K/V on every iteration.
        // Replicated joint: All data already present process joint when ring_id == ring_size-1
        const bool do_joint_kv = has_gathered_joint_k ? true : (ring_id == ring_size - 1);
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;
        const bool is_first_active_iter = !seen_active_iter;
        seen_active_iter = true;

        // Spatial global_n tail mask. Anchor on the LAST REAL tile (within - 1), like the joint_l
        // derivation below, so a chunk-final sub-tile partial column is masked, not just whole trailing
        // pad tiles. Unlike joint, do NOT clamp a beyond-shard boundary: trailing per-shard pad tiles in
        // a wider chunk are handled by the separate local_n mask, so global_n fires only when the
        // boundary lands in this shard (<= kv_local_padded_Nt). Chunked mode ends on a k-chunk-aligned
        // boundary, so the per-k_chunk-start skip handles it instead.
        const int32_t global_nt_within_ring_iter =
            static_cast<int32_t>(logical_nt) - static_cast<int32_t>(ring_id * kv_local_padded_Nt);
        const bool global_n_is_within_ring_iter =
            !chunked_enabled &&
            (global_nt_within_ring_iter > 0 && global_nt_within_ring_iter <= (int32_t)kv_local_padded_Nt);
        // Shard-relative chunk holding the last real spatial tile.
        const uint32_t global_n_mask_chunk_id =
            global_n_is_within_ring_iter ? (uint32_t)(global_nt_within_ring_iter - 1) / Sk_chunk_t : 0;
        // Fully-padded tiles trailing the last real tile within that chunk (0 when tile-aligned).
        const uint32_t global_n_valid_tiles_in_chunk =
            global_n_is_within_ring_iter ? (uint32_t)global_nt_within_ring_iter - global_n_mask_chunk_id * Sk_chunk_t
                                         : 0;
        const uint32_t global_n_padded_tiles_iter =
            global_n_is_within_ring_iter ? Sk_chunk_t - global_n_valid_tiles_in_chunk : 0;
        // Sub-tile partial column exists only on the boundary shard (never a fully-real earlier one).
        const bool global_n_apply_partial_col = global_n_is_within_ring_iter && (global_n_partial_col > 0);
        const bool ring_iter_needs_global_n_mask =
            global_n_is_within_ring_iter && (global_n_padded_tiles_iter > 0 || global_n_apply_partial_col);

        // LOCAL N MASK
        const bool local_n_needs_masking = kv_local_padded_Nt % Sk_chunk_t != 0;
        const uint32_t local_n_mask_chunk_id = kv_local_padded_Nt / Sk_chunk_t;

        // JOINT L MASK — keyed off the true logical joint tile count, mirroring the spatial global_n
        // derivation. Sharded joint: this iteration serves shard `ring_id`, whose joint tiles begin at
        // ring_id * Lt_local. Replicated joint: full L is consumed on the last active iteration, so the
        // boundary is measured from tile 0 over the whole joint region (Lt tiles), independent of ring_id.
        // Anchor on the LAST REAL tile (within - 1) so a tile-aligned boundary with a sub-tile partial
        // column is masked too; the boundary chunk carries both trailing pad tiles and the partial column.
        const uint32_t joint_shard_base_tiles = has_gathered_joint_k ? (ring_id * Lt_local) : 0u;
        const uint32_t joint_shard_tiles = has_gathered_joint_k ? Lt_local : Lt;
        const int32_t joint_nt_within_ring_iter =
            static_cast<int32_t>(logical_lt) - static_cast<int32_t>(joint_shard_base_tiles);
        // Does the global logical boundary (last real joint tile) land inside THIS shard? Only then is
        // there a sub-tile partial column to mask; a fully-real earlier shard must not apply it.
        const bool joint_boundary_in_shard =
            (joint_nt_within_ring_iter > 0 && joint_nt_within_ring_iter <= (int32_t)joint_shard_tiles);
        // Real joint tiles present in this shard's chunk region: the boundary clamped to the shard's tile
        // span. 0 => pure-padding shard (skipped upstream). Beyond the boundary the shard is fully real
        // (joint_shard_tiles tiles), but a wider K chunk (Sk_chunk_t > joint_shard_tiles) still carries
        // trailing pad tiles that must be masked.
        const uint32_t joint_real_tiles_in_shard =
            joint_nt_within_ring_iter <= 0
                ? 0u
                : (joint_boundary_in_shard ? (uint32_t)joint_nt_within_ring_iter : joint_shard_tiles);
        const bool joint_n_is_within_ring_iter = joint_real_tiles_in_shard > 0;
        // Shard-relative chunk holding the last real joint tile.
        const uint32_t joint_n_mask_chunk_id =
            joint_n_is_within_ring_iter ? (joint_real_tiles_in_shard - 1) / Sk_chunk_t : 0;
        // Fully-padded tiles trailing the last real tile within that chunk (0 when tile-aligned).
        const uint32_t joint_valid_tiles_in_chunk =
            joint_n_is_within_ring_iter ? joint_real_tiles_in_shard - joint_n_mask_chunk_id * Sk_chunk_t : 0;
        const uint32_t joint_n_padded_tiles_iter =
            joint_n_is_within_ring_iter ? Sk_chunk_t - joint_valid_tiles_in_chunk : 0;
        // Partial sub-tile column exists only on the boundary shard, never on a fully-real earlier one.
        const bool joint_apply_partial_col = joint_boundary_in_shard && (joint_l_partial_col > 0);
        const bool ring_iter_needs_joint_n_mask =
            do_joint_kv && joint_n_is_within_ring_iter && (joint_n_padded_tiles_iter > 0 || joint_apply_partial_col);

        // is_causal: diagonal only on iter 0 (K is local-frame). Chunked: every iter (absolute coords).
        // The compile-time-constant mask fields are template params (static constexpr, no stack
        // storage); only the per-iter runtime fields (global/joint padded tiles, straddle, causal) are
        // set below.
        RingStreamingMaskCtx<
            neginf_tile_idx,
            causal_diag_tile_idx,
            local_n_padded_tiles,
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
            // Precomputed with the last-real-tile anchor above (mirrors joint_n_padded_tiles_iter);
            // 0 on a chunk-final partial tile, where only the partial column is stamped.
            lw_mask.global_n_padded_tiles = global_n_padded_tiles_iter;
        }
        if (ring_iter_needs_joint_n_mask) {
            lw_mask.joint_n_padded_tiles = joint_n_padded_tiles_iter;
        }

        const bool is_last_ring_iter = has_sliding_window || is_last_active_ring_iter(active_ring_iter_mask, ring_iter);

        // Per-ring-iter K-chunk count and Q-skip flag — shared by v1 (sdpa_ring) and v2
        // (sdpa_ring_v2) paths.
        //   rix > rid (Case 3): only sender's L half is sent — halve KV count, or extend to
        //     include the straddle chunk when it crosses the coarse-half boundary (its
        //     late-half columns are -inf-masked via lw_mask.straddle_*).
        //   rix < rid && balanced (Case 2): skip first-half (L) Q-chunks.
        uint32_t iter_num_kv_chunks = num_kv_chunks;
        if constexpr (!has_sliding_window) {
            if (is_causal && is_balanced && ring_index > ring_id) {
                if constexpr (has_straddle) {
                    iter_num_kv_chunks = straddle_chunk_id + 1;
                } else {
                    iter_num_kv_chunks /= 2;
                }
            }
        }
        const bool skip_first_half_q = !has_sliding_window && (ring_index >= ring_id ? false : is_balanced);

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
                false,  // use_l1_state_fifo — compile-time off: no FIFO branch overhead here
                kv_pad_rotation_enabled,
                v_cb_physical_width_t,
                v_shares_k_buffer,
                kt_inplace_v,
                sliding_window_size,
                ring_size,
                use_attention_sink,
                cb_attention_sink,
                has_gathered_joint_k,
                Lt_local>(
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
                logical_lt);
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
