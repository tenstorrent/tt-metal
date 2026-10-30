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
#include "api/compute/pack.h"
#include "api/tile_constants.h"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/compute/sliding_ring_sdpa_streaming.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"
#include "cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp"
#include "cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/sliding_ring_sdpa_derived_slots.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/ring_joint_ksplit.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

void kernel_main() {
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t q_local_padded_Nt [[maybe_unused]] = get_compile_time_arg_val(5);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t logical_n_compile = get_compile_time_arg_val(7);
    constexpr uint32_t logical_nt_compile [[maybe_unused]] = get_compile_time_arg_val(8);
    constexpr uint32_t Lt = get_compile_time_arg_val(9);
    constexpr uint32_t L = get_compile_time_arg_val(10);
    constexpr uint32_t num_local_q_chunks = get_compile_time_arg_val(11);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t ring_size = get_compile_time_arg_val(15);
    constexpr uint32_t qk_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t qk_subblock_w = get_compile_time_arg_val(17);
    constexpr uint32_t qk_subblock_h = get_compile_time_arg_val(18);
    constexpr uint32_t qk_in0_num_subblocks = get_compile_time_arg_val(19);
    constexpr uint32_t qk_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t out_in0_block_w = get_compile_time_arg_val(21);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(22);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(23);
    constexpr uint32_t out_in0_num_subblocks = get_compile_time_arg_val(24);
    constexpr uint32_t out_in1_num_subblocks = get_compile_time_arg_val(25);

    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(26);
    constexpr bool use_streaming_compute = get_compile_time_arg_val(27) == 1;
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(28);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(29);
    constexpr bool is_causal = get_compile_time_arg_val(30) == 1;
    constexpr bool is_balanced = get_compile_time_arg_val(31) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(32) == 1;
    constexpr bool chunked_enabled = get_compile_time_arg_val(33) == 1;
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(34);
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    static_assert(
        kv_stripe_split > 0 && q_local_padded_Nt % kv_stripe_split == 0, "Q slab must split into whole KV regions");
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt / kv_stripe_split;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");
    constexpr bool kv_pad_rotation_enabled = get_compile_time_arg_val(35) == 1;
    constexpr bool v_shares_k_buffer = get_compile_time_arg_val(36) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(37) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(38);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    // The host builds this op only for sliding attention on the streaming compute path; the shared op's dense
    // ring loop and its fp32_dest_acc_en (compute_common.hpp) path are not carried here.
    static_assert(has_sliding_window, "Gemma4 sliding ring SDPA requires a sliding window");
    static_assert(use_streaming_compute, "Gemma4 sliding ring SDPA requires the streaming compute path");
    // Slot 40: sharded-joint flag (appended after upstream's kv_pad_from_metadata at slot 39, which is
    // declared further down next to cb_arg_offset). When true, one L/P shard arrives per ring iteration
    // and do_joint_kv fires on every iteration rather than only the last active iteration.
    constexpr bool joint_is_sharded = get_compile_time_arg_val(40) == 1;
    // Slot 41: true (unpadded) joint length in tiles (similar to spatial logical_nt). Drives the
    // per-ring-iteration joint tail mask and the joint out-of-bounds K-chunk skip.
    constexpr uint32_t logical_lt_compile = get_compile_time_arg_val(41);
    // Slots 42-45: transport-to-tensor rank mapping.
    constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(42) == 1;
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(43));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(44);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(45);
    // Slot 46: circular sliding KV slab count (0 = unbounded). Wraps the sliding work plan's
    // local slab addressing (sliding_window_work_plan.hpp) — must match the reader and host layout.
    constexpr uint32_t circular_kv_slab_count = get_compile_time_arg_val(46);
    // Slots 47-48: logical-length transport. Compute cannot NoC-read DRAM, so the live values arrive
    // through cb_kv_pad_derived from the reader; the compile-time values are the placeholder's.
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(47) == 1;
    constexpr bool has_logical_l_tensor = get_compile_time_arg_val(48) == 1;
    constexpr bool has_logical_length_tensor = has_logical_n_tensor || has_logical_l_tensor;
    uint32_t logical_lt = logical_lt_compile;
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
    // A tensor-supplied length can land mid-chunk / mid-tile at any dispatch, so compile the mask path in
    // regardless of the placeholder. Keep in sync with the factory and the writer (which generates them).
    constexpr bool global_n_has_padding =
        (logical_n_compile % (Sk_chunk_t * tt::constants::TILE_HEIGHT) != 0) || has_logical_n_tensor;
    // Joint needs a mask when either the per-device joint shard is not a whole number of K-chunks
    // (Lt_local % Sk_chunk_t != 0 -> fully-padded trailing tiles, the local_n analogue), or real tokens
    // do not fill the last real shard's chunk (logical_lt % Sk_chunk_t != 0 or a sub-tile partial
    // column, the global_n analogue).
    constexpr bool joint_has_padding =
        L > 0 && ((Lt_local % Sk_chunk_t != 0) || (logical_lt_compile % Sk_chunk_t != 0) ||
                  (joint_l_partial_col != 0) || has_logical_l_tensor);
    constexpr bool needs_lightweight_mask =
        (local_n_has_padding || global_n_has_padding || joint_has_padding) || diag_tile_enabled || has_sliding_window;

    constexpr uint32_t neginf_tile_idx = 0;
    constexpr uint32_t causal_diag_tile_idx = diag_tile_enabled ? 1 : 0;
    constexpr uint32_t edge_mask_tiles = has_sliding_window ? kSlidingWindowEdgeTiles : (diag_tile_enabled ? 1 : 0);
    constexpr uint32_t base_partial_offset = 1 + edge_mask_tiles;
    constexpr bool has_global_n_partial_tile =
        ring_joint::partial_tile_present(global_n_partial_col, has_logical_n_tensor);
    constexpr bool has_joint_l_partial_tile =
        ring_joint::partial_tile_present(joint_l_partial_col, has_logical_l_tensor);
    constexpr uint32_t global_n_partial_tile_idx = has_global_n_partial_tile ? base_partial_offset : 0;
    constexpr uint32_t joint_l_partial_tile_idx =
        has_joint_l_partial_tile ? (base_partial_offset + (has_global_n_partial_tile ? 1 : 0)) : 0;
    constexpr uint32_t total_mask_tiles =
        1 + edge_mask_tiles + (has_global_n_partial_tile ? 1 : 0) + (has_joint_l_partial_tile ? 1 : 0);

    constexpr uint32_t q_start_idx_t =
        chunked_enabled && !kv_pad_rotation_enabled ? logical_nt_compile - chunk_size_t : 0;

    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    [[maybe_unused]] const uint32_t q_per_core = global_q_end - global_q_start;

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
    // The shared op's static multicast Q count: without multicast padding it is this core's own Q count.
    ASSERT(get_arg_val<uint32_t>(argidx) == q_per_core);
    argidx++;

    // The K split's band index follows the scalars above.
    const uint32_t ksplit_arg_index = argidx;

    RingSDPAOpIndexer fused_op_indexer(
        ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);

    // Compute fixed slot 39: trace-safe KV-pad derivation flag. Slots 40/41 are the sharded-joint scalars
    // (joint_is_sharded, logical_lt); the rank mapping (42-45), the bounded sliding KV slab count (46), and
    // the logical-length transport flags (47-48) are declared above, so the CB block starts at 49.
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(39) == 1;
    constexpr uint32_t cb_arg_offset = 49;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_v_in = get_compile_time_arg_val(cb_arg_offset + 2);
    constexpr uint32_t cb_mask_in = get_compile_time_arg_val(cb_arg_offset + 3);
    constexpr uint32_t cb_scale_in = get_compile_time_arg_val(cb_arg_offset + 4);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_max_in = get_compile_time_arg_val(cb_arg_offset + 6);  // K-split reducer: sender max
    constexpr uint32_t cb_prev_out = get_compile_time_arg_val(cb_arg_offset + 7);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_recip_scratch =
        get_compile_time_arg_val(cb_arg_offset + 9);  // 1-tile scratch for normalize_row_streaming
    constexpr uint32_t cb_sum_out = get_compile_time_arg_val(cb_arg_offset + 10);
    constexpr uint32_t cb_sum_in = get_compile_time_arg_val(cb_arg_offset + 11);
    constexpr uint32_t cb_signal = get_compile_time_arg_val(cb_arg_offset + 12);
    constexpr uint32_t cb_out = get_compile_time_arg_val(cb_arg_offset + 13);
    constexpr uint32_t cb_max_out = get_compile_time_arg_val(cb_arg_offset + 14);
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

    // The factory keeps the shared op's rotated Q split slot as the final compile-time arg of every kernel.
    // Sliding never rotates Q, so it is always 0; it also never schedules padded multicast Q slots, a dense
    // causal K skip, or segmented accumulation.
    constexpr uint32_t rotated_max_slots = get_ct_arg<kernel_compile_time_args.size() - 1>();
    static_assert(rotated_max_slots == 0, "Sliding attention has no rotated Q split");
    static_assert(get_named_compile_time_arg_val("padded_q_slots") == 0, "Sliding attention has no padded Q slots");
    static_assert(
        get_named_compile_time_arg_val("dense_causal_skip") == 0, "Sliding attention has no dense causal K skip");
    static_assert(get_named_compile_time_arg_val("seg_accum") == 0, "Sliding attention has no segmented accumulation");
    constexpr uint32_t ksplit_count = get_named_compile_time_arg_val("ksplit_count");
    constexpr bool ksplit_enabled = ksplit_count > 1;
    [[maybe_unused]] const uint32_t ksplit_idx = ksplit_enabled ? get_arg_val<uint32_t>(ksplit_arg_index) : 0;
    const bool ksplit_active = ksplit_enabled && q_per_core == 1;
    // Largest K-chunk count of the unit's sliding work plan: a band is non-empty iff it is non-empty at
    // that count, so the reducer knows which senders have state.
    [[maybe_unused]] uint32_t ksplit_max_valid = 0;

    // Placeholder-derived defaults, overwritten from the CB on the tensor path. 0 gates the stamp off.
    uint32_t global_n_partial_col_live = global_n_partial_col;
    uint32_t joint_l_partial_col_live = joint_l_partial_col;

    if constexpr (kv_pad_from_metadata || has_logical_length_tensor) {
        CircularBuffer cb_derived(cb_kv_pad_derived);
        cb_derived.wait_front(1);
        constexpr uint32_t kDerivedTile = 0;
        logical_nt = ckernel::read_tile_value(cb_kv_pad_derived, kDerivedTile, ring_joint::kDerivedLogicalNt);
        if constexpr (kv_pad_from_metadata) {
            kv_pad_q_pre_wrap_start_tile =
                ckernel::read_tile_value(cb_kv_pad_derived, kDerivedTile, ring_joint::kDerivedQPreWrapStartTile);
            kv_pad_q_pre_wrap_tile_count =
                ckernel::read_tile_value(cb_kv_pad_derived, kDerivedTile, ring_joint::kDerivedQPreWrapTileCount);
            kv_pad_q_post_wrap_start_tile =
                ckernel::read_tile_value(cb_kv_pad_derived, kDerivedTile, ring_joint::kDerivedQPostWrapStartTile);
            kv_pad_q_valid_tile_count =
                ckernel::read_tile_value(cb_kv_pad_derived, kDerivedTile, ring_joint::kDerivedQValidTileCount);
        }
        active_ring_iter_mask =
            ckernel::read_tile_value(cb_kv_pad_derived, kDerivedTile, ring_joint::kDerivedActiveRingIterMask);
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

    RingAccumulatorState acc_state = {
        {cb_sum_A, cb_max_A, cb_out_im_A},  // prev
        {cb_sum_B, cb_max_B, cb_out_im_B},  // cur
    };

    // acc_state.prev = exp(prev.max - m) * prev + exp(other.max - m) * other via the K loop's SALAD correction, where
    // m is the row max of both; consumes both states. The K-split reducer merges with it.
    [[maybe_unused]] auto merge_into_prev = [&](const AccumulatorHalf& other) {
        constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
        constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
        constexpr uint32_t qktv_h =
            ttnn::transformer::sdpa::streaming_qktv_h(out_subblock_h, out_subblock_w, dst_size, Sq_chunk_t);
        // kernel_main is not a template, so this is checked even for configs that never merge.
        static_assert(!ksplit_enabled || Sq_chunk_t % qktv_h == 0, "merges walk whole SALAD row groups");
        const AccumulatorHalf& a = acc_state.prev;
        const AccumulatorHalf& merged = acc_state.cur;
        max_block_sfpi(a.max, other.max, merged.max, Sq_chunk_t);
        CircularBuffer(merged.out).reserve_back(out_tiles);
        CircularBuffer(merged.sum).reserve_back(Sq_chunk_t);
        exp_packthread_tile_init<EXP_APPROX_MODE>();
        for (uint32_t group = 0; group < Sq_chunk_t / qktv_h; ++group) {
            for (uint32_t side = 0; side < 2; ++side) {
                const AccumulatorHalf& src = side == 0 ? a : other;
                CircularBuffer(cb_exp_max_diff).reserve_back(qktv_h);
                sub_exp_first_col_blocks<false, scale_fp32>(src.max, merged.max, cb_exp_max_diff, group, qktv_h);
                CircularBuffer(cb_exp_max_diff).push_back(qktv_h);
                pack_reconfig_l1_acc(side);
                salad_correct_fused<qktv_h, vDHt, dst_size>(
                    src.out, src.sum, cb_exp_max_diff, merged.out, merged.sum, 0, group, group);
                pack_reconfig_l1_acc(0);
                sdpa_cb_pop_front_out_of_line(cb_exp_max_diff, qktv_h);
                sdpa_cb_pop_front_out_of_line(src.out, qktv_h * vDHt);
            }
        }
        CircularBuffer(merged.out).push_back(out_tiles);
        CircularBuffer(merged.sum).push_back(Sq_chunk_t);
        for (uint32_t cb : {a.sum, a.max, other.sum, other.max}) {
            sdpa_cb_pop_front_out_of_line(cb, Sq_chunk_t);
        }
        std::swap(acc_state.prev, acc_state.cur);
    };

    const uint32_t ring_index =
        ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
            fused_op_indexer.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    const ChunkedContext chunked_context{
        q_start_idx_t,
        ring_index / kv_stripe_split,
        KVPadRotationContext{
            kv_pad_q_pre_wrap_start_tile,
            kv_pad_q_pre_wrap_tile_count,
            kv_pad_q_post_wrap_start_tile,
            kv_pad_q_valid_tile_count}};
    // The first active iter starts with fresh accumulators. After the loop it records whether this core
    // holds state (a K-split band can be empty).
    bool seen_active_iter = false;
    auto arrival_preview = fused_op_indexer.seq;
    const ring_joint::PackedKVSchedule packed_schedule = ring_joint::make_packed_kv_schedule<GROUPED_KV_SOURCE_COUNT>(
        ring_size,
        kv_local_padded_Nt,
        logical_nt,
        active_ring_iter_mask,
        kv_region_Nt,
        Sk_chunk_t,
        has_sliding_window,
        [&] {
            return ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                arrival_preview.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
        });
    const uint32_t sdpa_ring_iterations = packed_schedule.ring_iterations;
    for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
        // Sliding folds all local/halo source ranges into one synthetic local iteration.
        // The dataflow reader has already waited for the required halo completion signals.
        const uint32_t ring_id = ring_index;
        // Sharded joint: one L/P shard per ring iteration — process joint K/V on every iteration.
        // Replicated joint: All data already present process joint when ring_id == ring_size-1
        const bool do_joint_kv = has_gathered_joint_k ? true : (ring_id == ring_size - 1);
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
        const bool global_n_apply_partial_col = global_n_is_within_ring_iter && (global_n_partial_col_live > 0);
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
        const bool joint_apply_partial_col = joint_boundary_in_shard && (joint_l_partial_col_live > 0);
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
            global_n_partial_tile_idx,
            joint_l_partial_tile_idx,
            straddle_chunk_id>
            lw_mask;
        lw_mask.global_n_partial_col = global_n_partial_col_live;
        lw_mask.joint_l_partial_col = joint_l_partial_col_live;
        lw_mask.is_causal = chunked_enabled || (is_causal && ring_iter == 0);
        // Straddle mask fires only on the shared op's balanced rix>rid halved-range iters, which sliding
        // (unbalanced) never runs.
        lw_mask.straddle_num_padded_tiles = 0;
        if (ring_iter_needs_global_n_mask) {
            // Precomputed with the last-real-tile anchor above (mirrors joint_n_padded_tiles_iter);
            // 0 on a chunk-final partial tile, where only the partial column is stamped.
            lw_mask.global_n_padded_tiles = global_n_padded_tiles_iter;
        }
        if (ring_iter_needs_joint_n_mask) {
            lw_mask.joint_n_padded_tiles = joint_n_padded_tiles_iter;
        }

        // K-split cores normalize after the merge, not in the ring loop.
        const bool is_last_ring_iter = !ksplit_active;

        gemma4_sliding::sdpa_ring_v2<
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
            sliding_window_size,
            circular_kv_slab_count,
            ring_size,
            use_attention_sink,
            cb_attention_sink,
            kv_region_Nt,
            (GROUPED_KV_SOURCE_COUNT > 1)>(
            global_q_start,
            global_q_end,
            num_q_chunks,
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
            use_zigzag_balancing,
            chunked_context,
            is_first_active_iter,
            ksplit_active ? ksplit_idx : 0,
            ksplit_active ? ksplit_count : 1,
            nullptr,
            0,
            packed_schedule.rows(ring_iter));
        if constexpr (ksplit_enabled) {
            // Sliding bands split the unit's work plan, which can be shorter than the band count near the
            // sequence start: an empty band stages no state.
            if (ksplit_active) {
                seen_active_iter = acc_state.last_call_k_chunks > 0;
                ksplit_max_valid = acc_state.sliding_plan_k_chunks;
            }
        }
    }

    // cb_sum_in exists only on the streaming path; alias a live CB so the other modes still compile.
    [[maybe_unused]] constexpr uint32_t ksplit_cb_sum_in = ksplit_enabled ? cb_sum_in : cb_out;
    // K split epilogue: acc_state.prev holds this core's unnormalized state. Only primitives the K loop already
    // instantiates are used, since the program is at the kernel config buffer limit.
    if constexpr (ksplit_enabled) {
        if (ksplit_active) {
            // A single-chunk slice runs only a first K step, which leaves the packer at the QKT@V subblock width;
            // copy_block and max_block_sfpi pack one tile per pack_tile.
            configure_single_tile_pack(cb_max_A);
            constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;
            // The reader pushes Q with this core's first K chunk.
            if (seen_active_iter) {
                sdpa_cb_pop_front_out_of_line(cb_q_in, Sq_chunk_t * DHt);
            }
            if (ksplit_idx != ksplit_count - 1) {
                // Sender: leave the state in half A, where the reducer reads it, and signal the writer. An empty
                // slice stages no state; the reducer discards it using the same chunk counts.
                if (seen_active_iter && acc_state.prev.out != cb_out_im_A) {
                    copy_block(acc_state.prev.max, cb_max_A, Sq_chunk_t);
                    copy_block(acc_state.prev.sum, cb_sum_A, Sq_chunk_t);
                    copy_block(acc_state.prev.out, cb_out_im_A, out_tiles);
                }
                CircularBuffer(cb_signal).reserve_back(1);
                CircularBuffer(cb_signal).push_back(1);
            } else {
                // Reducer: its slice holds the diagonal chunk, so its own state always exists.
                ASSERT(seen_active_iter);
                constexpr uint32_t dst_size = compute_kernel_lib::DEST_AUTO_LIMIT;
                const AccumulatorHalf incoming = {ksplit_cb_sum_in, cb_max_in, cb_prev_out};
                for (uint32_t sender = 0; sender + 1 < ksplit_count; ++sender) {
                    const auto sender_range = ring_joint::sliding_ksplit_range(ksplit_max_valid, sender, ksplit_count);
                    if (sender_range.empty()) {
                        for (uint32_t cb : {cb_max_in, ksplit_cb_sum_in}) {
                            CircularBuffer(cb).wait_front(Sq_chunk_t);
                            sdpa_cb_pop_front_out_of_line(cb, Sq_chunk_t);
                        }
                        CircularBuffer(cb_prev_out).wait_front(out_tiles);
                        sdpa_cb_pop_front_out_of_line(cb_prev_out, out_tiles);
                        continue;
                    }
                    merge_into_prev(incoming);
                }
                normalize_row_streaming<
                    false,
                    vDHt,
                    dst_size,
                    cb_col_identity,
                    cb_recip_scratch,
                    cb_out,
                    scale_fp32,
                    use_attention_sink,
                    cb_attention_sink>(acc_state.prev.sum, acc_state.prev.out, Sq_chunk_t, acc_state.prev.max);
                sdpa_cb_pop_front_out_of_line(acc_state.prev.max, Sq_chunk_t);
            }
        }
    }
}
