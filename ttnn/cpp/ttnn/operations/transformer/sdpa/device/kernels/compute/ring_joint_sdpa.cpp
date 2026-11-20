// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring joint SDPA compute on the precision recipes (STANDARD, BALANCED, ACCURATE, FAST; host:
// ring_joint_sdpa_program_factory.cpp). The recipe core loop (recipe/recipe_streaming.hpp) is dense SDPA's; this kernel
// is its ring outer loop. Each Q chunk continues one recurrent state across the active ring iterations
// (recipe/recipe_ring.hpp) and only its last iteration normalizes. A Q chunk processes exactly the K chunks the reader
// sends it (ring_joint_reader.cpp); padding and positional masks (causal diagonal, chunked prefill and sliding windows
// in the sequence frame, KV-pad rotation) apply in the pack thread (recipe/recipe_tail.hpp: RecipeRingMask). A sliding
// window folds every local and halo K range of a Q chunk into one pass.

#include <cstdint>

// This kernel reconfigs ~30x; inlining the LLK Src zero-flag DEFAULT configurator at each site pushes
// the program over the kernel-config buffer. Force it out-of-line here.
#define LLK_ZEROFLAG_OUTLINE 1

// Optimization. Blackhole: unpack and pack at the kernel's level, which the host picks per recipe (O2 where -O3
// overflows the kernel config buffer); math at -O2. Wormhole (larger images, 2 KiB TRISC local memory): math at -Os,
// unpack and pack at -O2 (unpack at -Os deepens its stack past the local memory). Per-thread and per-architecture levels
// only the preprocessor can select. Watcher builds are size-optimized on every thread but Wormhole's unpack, which keeps
// -O2: at -Os its watcher build overflows the 2 KiB local memory.
#if (defined(WATCHER_ENABLED) && !(defined(ARCH_WORMHOLE) && defined(TRISC_UNPACK))) || \
    (defined(ARCH_WORMHOLE) && defined(TRISC_MATH))
#pragma GCC optimize("Os")
#elif defined(ARCH_WORMHOLE) || defined(TRISC_MATH)
#pragma GCC optimize("O2")
#endif

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/tile_constants.h"
#include "compute_common.hpp"
#include "recipe/recipe_tail.hpp"
#include "recipe/recipe_sfpu.hpp"
#include "recipe/recipe_streaming.hpp"
#include "recipe/recipe_ring.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"
#include "cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/ring_mla_packing_plan.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

namespace {
// One K chunk of a Q chunk's state through the recipe core (the hot path, at the kernel's optimization level). The
// ring control code below calls it per chunk and is size-optimized: the ring builds sit at the kernel config buffer.
__attribute__((noinline)) void ring_run_chunk(RecipeAccumulatorState& state, bool last, bool release_q) {
    using namespace recipe_ring_cb;
    // FP32 recipes process one Q tile row per group; paired BF16 recipes two (an odd chunk ends with one row).
    constexpr uint32_t subblock_h = recipe::fp32 ? 1 : 2;
    sdpa_recipe_segment<
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(26),
        subblock_h,
        recipe::qk_w,
        subblock_h,
        recipe::pv_w,
        q,
        k,
        v,
        qk,
        identity_scale,
        exp_max_diff,
        col_identity,
        recip_scratch,
        out,
        true>(state, 1, last, release_q);
}
}  // namespace

__attribute__((optimize("Os"))) void kernel_main() {
    // Positional compile-time args: the ring geometry slots the reader and writer share (factory: compute args).
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    // Length-geometry slots (the _ct values): placeholders with dynamic_local_n, see ring_joint_derived_slots.hpp.
    constexpr uint32_t q_local_padded_Nt_ct = get_compile_time_arg_val(5);
    constexpr uint32_t kv_local_padded_Nt_ct = get_compile_time_arg_val(6);
    constexpr uint32_t logical_nt_compile = get_compile_time_arg_val(8);
    constexpr uint32_t Lt = get_compile_time_arg_val(9);
    constexpr uint32_t num_local_q_chunks_ct = get_compile_time_arg_val(11);
    constexpr uint32_t num_local_k_chunks_ct = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_q_chunks_ct = get_compile_time_arg_val(14);
    constexpr uint32_t ring_size = get_compile_time_arg_val(15);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(26);
    constexpr uint32_t global_n_partial_col = get_compile_time_arg_val(28);
    constexpr uint32_t joint_l_partial_col = get_compile_time_arg_val(29);
    // Slots 30-33: causal (the shard-frame diagonal; off for chunked prefill, which masks in the sequence frame),
    // balanced, the balanced zigzag Q order, chunked prefill.
    constexpr bool is_causal = get_compile_time_arg_val(30) == 1;
    constexpr bool is_balanced = get_compile_time_arg_val(31) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(32) == 1;
    constexpr bool chunked_enabled = get_compile_time_arg_val(33) == 1;
    constexpr uint32_t chunk_size_t_ct = get_compile_time_arg_val(34);
    constexpr bool kv_pad_rotation_enabled = get_compile_time_arg_val(35) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(38);
    constexpr bool has_sliding_window = sliding_window_size > 0;
    // Slot 39: trace-safe KV-pad derivation (the reader derives the rotation from the metadata tensors).
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(39) == 1;
    // Slot 40: sharded joint (one L/P shard per ring iteration). Slot 41: logical joint tiles.
    constexpr bool joint_is_sharded = get_compile_time_arg_val(40) == 1;
    constexpr uint32_t logical_lt_compile = get_compile_time_arg_val(41);
    // Slots 42-45: transport-to-tensor rank mapping.
    constexpr uint32_t full_mesh_rank_mapping = get_compile_time_arg_val(42);
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(43));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(44);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(45);
    // Slot 46: circular sliding KV slab count (0: unbounded).
    constexpr uint32_t circular_kv_slab_count = get_compile_time_arg_val(46);
    // Slots 47-48: logical_n / logical_l arrive as device tensors; the compile-time values are placeholders.
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(47) == 1;
    constexpr bool has_logical_l_tensor = get_compile_time_arg_val(48) == 1;

    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    constexpr bool dynamic_local_n = get_named_compile_time_arg_val("dynamic_local_n") == 1;
    constexpr bool dense_causal_skip = get_named_compile_time_arg_val("dense_causal_skip") == 1;
    constexpr uint32_t cb_kv_pad_derived = recipe_ring_cb::derived;
    constexpr uint32_t sliding_max_source_ranges = get_named_compile_time_arg_val("sliding_max_source_ranges");
    constexpr bool sliding_local_first = get_named_compile_time_arg_val("sliding_local_first") == 1;
    static_assert(
        kv_stripe_split > 0 && q_local_padded_Nt_ct % kv_stripe_split == 0, "Q slab must split into whole KV regions");
    constexpr uint32_t kv_region_Nt_ct = q_local_padded_Nt_ct / kv_stripe_split;
    static_assert(kv_region_Nt_ct * ring_size == chunk_size_t_ct, "chunk_size_t must span one KV region per ring rank");
    static_assert(!has_sliding_window || chunked_enabled, "Sliding windows require chunked prefill");
    static_assert(
        recipe::ring_mask == (chunked_enabled ? 2u : (is_causal ? 1u : 0u)), "the host's positional mask mode");
    static_assert(recipe::ring_window == sliding_window_size && recipe::ring_rotation == kv_pad_rotation_enabled);
    static_assert(recipe::ring_circular == (circular_kv_slab_count > 1));

    // Joint Q chunks stay compile-time; the live local counts come from the common word.
    constexpr uint32_t num_joint_q_chunks_ct = num_q_chunks_ct - num_local_q_chunks_ct;
    const ring_joint::LocalNGeometry local_n =
        dynamic_local_n ? ring_joint::derive_local_n_geometry(
                              get_common_arg_val<uint32_t>(ring_joint::kDynamicLocalNComputeCommonArg),
                              Sq_chunk_t,
                              Sk_chunk_t,
                              kv_stripe_split,
                              ring_size,
                              num_joint_q_chunks_ct)
                        : ring_joint::LocalNGeometry{
                              q_local_padded_Nt_ct,
                              kv_local_padded_Nt_ct,
                              num_local_q_chunks_ct,
                              num_local_k_chunks_ct,
                              num_q_chunks_ct,
                              kv_region_Nt_ct,
                              chunk_size_t_ct};
    const uint32_t q_local_padded_Nt = local_n.q_local_padded_Nt;
    const uint32_t kv_local_padded_Nt = local_n.kv_local_padded_Nt;
    const uint32_t num_local_k_chunks = local_n.num_local_k_chunks;
    const uint32_t num_q_chunks = local_n.num_q_chunks;

    constexpr bool has_joint_k = num_joint_k_chunks > 0;
    constexpr bool has_gathered_joint_k = joint_is_sharded && has_joint_k;
    // Per-device joint shard length in tiles (the per-ring-iteration joint tail boundary).
    constexpr uint32_t Lt_local = has_gathered_joint_k ? Lt / ring_size : Lt;
    constexpr uint32_t k_chunk_tiles = Sk_chunk_t * DHt;
    constexpr uint32_t v_chunk_tiles = Sk_chunk_t * vDHt;
    constexpr uint32_t k_chunk_rows = Sk_chunk_t * tt::constants::TILE_HEIGHT;

    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_per_core = global_q_end - global_q_start;
    const uint32_t ring_size_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t forward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t backward_writes_expected = get_arg_val<uint32_t>(argidx++);
    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    KVPadRotationContext rotation{};
    rotation.q_pre_wrap_start_tile = get_arg_val<uint32_t>(argidx++);
    rotation.q_pre_wrap_tile_count = get_arg_val<uint32_t>(argidx++);
    rotation.q_post_wrap_start_tile = get_arg_val<uint32_t>(argidx++);
    rotation.q_valid_tile_count = get_arg_val<uint32_t>(argidx++);
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);
    // The reader's multicast loop count; slots past this core's own Q chunks are padding (their K/V are drained).
    const uint32_t padded_q_slots = get_arg_val<uint32_t>(argidx++) - q_per_core;

    RingSDPAOpIndexer fused_op_indexer(
        ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);

    // Live lengths (and the ring work mask re-derived from them) from the reader, which read the device tensors;
    // compute cannot NoC-read DRAM. Must precede compute_kernel_hw_startup: read_tile_value rendezvouses
    // UNPACK -> MATH/PACK through the mailboxes.
    uint32_t global_n_partial_col_live = global_n_partial_col;
    uint32_t joint_l_partial_col_live = joint_l_partial_col;
    uint32_t logical_lt = logical_lt_compile;
    if constexpr (kv_pad_from_metadata || has_logical_n_tensor || has_logical_l_tensor) {
        CircularBuffer derived(cb_kv_pad_derived);
        derived.wait_front(1);
        logical_nt = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedLogicalNt);
        if constexpr (kv_pad_from_metadata) {
            rotation.q_pre_wrap_start_tile =
                ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedQPreWrapStartTile);
            rotation.q_pre_wrap_tile_count =
                ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedQPreWrapTileCount);
            rotation.q_post_wrap_start_tile =
                ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedQPostWrapStartTile);
            rotation.q_valid_tile_count =
                ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedQValidTileCount);
        }
        active_ring_iter_mask = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedActiveRingIterMask);
        if constexpr (has_logical_n_tensor) {
            global_n_partial_col_live = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedGlobalNPartialCol);
        }
        if constexpr (has_logical_l_tensor) {
            logical_lt = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedLogicalLt);
            joint_l_partial_col_live = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedJointLPartialCol);
        }
        derived.pop_front(1);
    }

    using namespace recipe_ring_cb;
    compute_kernel_hw_startup<SrcOrder::Reverse>(q, k, qk);
    if constexpr (recipe::lofi) {
        recipe_mm_init(q, k, true, 1, 1, 1);  // no MOP matmul code (see normalize_row_streaming)
    } else {
        matmul_init(q, k);
    }
    // The reduce scaler and the column identity stay fronted for the whole kernel.
    CircularBuffer(identity_scale).wait_front(1);
    init_sdpa_streaming_semaphores();
    CircularBuffer(col_identity).wait_front(1);

    const uint32_t ring_index = ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
        fused_op_indexer.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    // Sequence-frame masks: the chunk groups' geometry, the logical length and the Q slab's rotation segments.
    recipe_ring_frame.group_tiles = local_n.q_chunk_group_tile_count;
    recipe_ring_frame.region_tiles = local_n.kv_region_Nt;
    recipe_ring_frame.local_tiles = kv_local_padded_Nt;
    recipe_ring_frame.logical_tiles = logical_nt;
    recipe_ring_frame.rotation = rotation;
    // Chunked prefill's first absolute Q tile of this device's slab (the Q slab rank is the tensor rank over the
    // KV/Q split); KV-pad rotation maps the slab's tiles through its wrap segments instead.
    const uint32_t chunked_q_tile0 =
        (chunked_enabled && !kv_pad_rotation_enabled ? logical_nt_compile - chunk_size_t_ct : 0) +
        (ring_index / kv_stripe_split) * q_local_padded_Nt;
    // Dense chunked prefill: K chunks past this device's last Q row are fully masked; the reader skips them.
    const uint32_t causal_end_nt =
        dense_causal_skip ? chunked_q_global_end_tile<kv_pad_rotation_enabled, q_local_padded_Nt_ct>(
                                logical_nt,
                                ring_index,
                                ring_size,
                                rotation.q_pre_wrap_start_tile,
                                rotation.q_pre_wrap_tile_count,
                                rotation.q_post_wrap_start_tile,
                                rotation.q_valid_tile_count)
                          : logical_nt;
    // Logical key rows (the last tile may be partial).
    const uint32_t valid_n_rows =
        logical_nt * tt::constants::TILE_HEIGHT - (global_n_partial_col_live ? 32 - global_n_partial_col_live : 0);
    const uint32_t valid_l_rows =
        logical_lt * tt::constants::TILE_HEIGHT - (joint_l_partial_col_live ? 32 - joint_l_partial_col_live : 0);

    // Balanced causal: the last active step on K from this or an earlier device, after which the early half's Q
    // chunks see nothing more (walks the reader's ring order without its syncs).
    [[maybe_unused]] uint32_t early_last_iter = 0;
    if constexpr (is_causal && is_balanced && !has_sliding_window) {
        RingIdSequencer order = fused_op_indexer.seq;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<
                full_mesh_rank_mapping>(order.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
            if (((active_ring_iter_mask >> ring_iter) & 1u) != 0 && ring_id <= ring_index) {
                early_last_iter = ring_iter;
            }
        }
    }

    const auto run_chunk = [](RecipeAccumulatorState& state, bool last, bool release_q) {
        ring_run_chunk(state, last, release_q);
    };

    // Whether the reader sends spatial K chunk `k_chunk` of `source_ring_id`'s shard (ring_joint_reader.cpp: the
    // logical_n skip, the dense causal skip of chunked prefill).
    const auto local_chunk_sent = [&](uint32_t source_ring_id, uint32_t k_chunk) -> bool {
        if constexpr (dynamic_local_n) {
            return chunked_kv_chunk_is_live<kv_pad_rotation_enabled, chunked_enabled>(
                source_ring_id,
                k_chunk,
                Sk_chunk_t,
                logical_nt,
                causal_end_nt,
                kv_local_padded_Nt,
                local_n.q_chunk_group_tile_count,
                local_n.kv_region_Nt);
        } else {
            return chunked_kv_chunk_is_live<
                kv_pad_rotation_enabled,
                chunked_enabled,
                kv_local_padded_Nt_ct,
                chunk_size_t_ct,
                kv_region_Nt_ct>(source_ring_id, k_chunk, Sk_chunk_t, logical_nt, causal_end_nt);
        }
    };

    RecipeAccumulatorState resident{{sum_a, max_a, out_a}, {sum_b, max_b, out_b}};
    const bool staged = q_per_core > 1;
    bool seen_active_iter = false;
    const uint32_t sdpa_ring_iterations = ring_joint::ring_pass_count(ring_size, has_sliding_window);
    for (uint32_t ring_iter = 0; ring_iter < sdpa_ring_iterations; ++ring_iter) {
        // Sliding folds all local and halo source ranges into one local pass; the reader waits for the halo.
        const uint32_t ring_id =
            has_sliding_window
                ? ring_index
                : ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                      fused_op_indexer.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
        // The host (or the reader, from device lengths) marks the ring iterations with work; the ring-id sequencing
        // above still advances so compute stays aligned with the reader, writer and all-gather.
        if (!has_sliding_window && ((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
            continue;
        }
        const bool first_active_iter = !seen_active_iter;
        seen_active_iter = true;
        const bool is_last_ring_iter = has_sliding_window || is_last_active_ring_iter(active_ring_iter_mask, ring_iter);
        // Sharded joint: one L/P shard per ring iteration. Replicated joint: all of L with the last device's K.
        const bool do_joint_kv = has_gathered_joint_k ? true : (ring_id == ring_size - 1);
        uint32_t num_kv_chunks = num_local_k_chunks;
        if constexpr (has_joint_k) {
            if (do_joint_kv) {
                num_kv_chunks += num_joint_k_chunks;
            }
        }
        // Balanced causal: K from an earlier device is visible in its shard's early half only (the reader sends the
        // chunks up to the one straddling the half); the early half's Q chunks see no K from a later device.
        const bool earlier_balanced = is_causal && is_balanced && !has_sliding_window && ring_index > ring_id;
        const bool skip_early = is_causal && is_balanced && !has_sliding_window && ring_index < ring_id;
        const bool early_last = is_causal && is_balanced && !has_sliding_window && ring_iter == early_last_iter;
        if (earlier_balanced) {
            using Straddle = KCausalStraddleInfo<kv_local_padded_Nt_ct, Sk_chunk_t>;
            if constexpr (Straddle::has_straddle) {
                num_kv_chunks = Straddle::straddle_chunk_id + 1;
            } else {
                num_kv_chunks /= 2;
            }
        }
        // Valid key rows of this shard's spatial K: the logical tail clipped to the shard (the balanced early half
        // for an earlier device); chunked prefill caches pad only past the local slab.
        uint32_t shard_rows = kv_local_padded_Nt * tt::constants::TILE_HEIGHT;
        if constexpr (!chunked_enabled) {
            const uint32_t n_origin = ring_id * kv_local_padded_Nt * tt::constants::TILE_HEIGHT;
            shard_rows = valid_n_rows <= n_origin               ? 0
                         : valid_n_rows - n_origin < shard_rows ? valid_n_rows - n_origin
                                                                : shard_rows;
            if (earlier_balanced && shard_rows > kv_local_padded_Nt * (tt::constants::TILE_HEIGHT / 2)) {
                shard_rows = kv_local_padded_Nt * (tt::constants::TILE_HEIGHT / 2);
            }
        }
        // Valid joint key rows of this iteration's joint shard.
        const uint32_t joint_base_rows = (has_gathered_joint_k ? ring_id * Lt_local : 0u) * tt::constants::TILE_HEIGHT;
        const uint32_t joint_shard_rows = (has_gathered_joint_k ? Lt_local : Lt) * tt::constants::TILE_HEIGHT;
        const uint32_t joint_rows = !do_joint_kv || valid_l_rows <= joint_base_rows ? 0
                                    : valid_l_rows - joint_base_rows < joint_shard_rows
                                        ? valid_l_rows - joint_base_rows
                                        : joint_shard_rows;
        // Whether the reader sends iteration chunk k (spatial: the logical_n / causal skips; sharded joint: chunks
        // past the logical joint length are skipped).
        const auto chunk_sent = [&](uint32_t k_chunk) -> bool {
            if (k_chunk < num_local_k_chunks) {
                return local_chunk_sent(ring_id, k_chunk);
            }
            if constexpr (has_gathered_joint_k) {
                return ring_id * Lt_local + (k_chunk - num_local_k_chunks) * Sk_chunk_t < logical_lt;
            }
            return true;
        };

        uint32_t chunks_popped = 0;  // every K/V pair this iteration pops, for the phase alignment below
        for (uint32_t q_flat = global_q_start; q_flat < global_q_end; ++q_flat) {
            const uint32_t q_chunk = remap_q_index(q_flat, num_q_chunks, use_zigzag_balancing) % num_q_chunks;
            const bool early = q_chunk < num_q_chunks / 2;
            if (skip_early && early) {
                continue;
            }
            const bool q_last = is_last_ring_iter || (early && early_last);
            if constexpr (recipe::sink) {
                if (q_last) {
                    recipe_read_sink();
                }
            }
            const bool next_block_unrestored = first_active_iter && q_flat + 1 < global_q_end;

            if constexpr (has_sliding_window) {
                // One pass over the Q chunk's work plan: the local slab and the halo, in the reader's order.
                const ring_joint::ChunkedQMapping q_mapping{
                    rotation.q_pre_wrap_start_tile,
                    rotation.q_pre_wrap_tile_count,
                    rotation.q_post_wrap_start_tile,
                    rotation.q_valid_tile_count};
                const auto plan = ring_joint::build_sliding_q_work_plan<sliding_max_source_ranges>(
                    q_chunk * Sq_chunk_t,
                    Sq_chunk_t,
                    ring_index,
                    q_local_padded_Nt,
                    ring_size,
                    sliding_window_size,
                    tt::constants::TILE_HEIGHT,
                    kv_local_padded_Nt,
                    Sk_chunk_t,
                    logical_nt,
                    circular_kv_slab_count,
                    kv_pad_rotation_enabled ? &q_mapping : nullptr,
                    sliding_local_first);
                ASSERT(plan.is_valid && plan.total_k_chunk_count > 0);
                const uint32_t q_tile0 =
                    kv_pad_rotation_enabled
                        ? q_chunk * Sq_chunk_t
                        : logical_nt - ring_size * q_local_padded_Nt + ring_index * q_local_padded_Nt + q_chunk * Sq_chunk_t;
                recipe_ring_q_block<Sq_chunk_t, vDHt>(
                    resident,
                    staged,
                    true,
                    true,
                    q_flat,
                    false,
                    plan.total_k_chunk_count,
                    [&](RecipeAccumulatorState& state, uint32_t chunk, bool last_k) {
                        const auto ref = plan.k_chunk_at(chunk);
                        recipe_ring_mask.valid_rows = k_chunk_rows;
                        recipe_ring_mask.masked = true;
                        recipe_ring_mask.positional = true;
                        recipe_ring_mask.q_tile0 = q_tile0;
                        recipe_ring_mask.k_tile0 = ref.source_k_chunk * Sk_chunk_t;
                        if constexpr (circular_kv_slab_count > 1) {
                            recipe_ring_mask.k_seq_tile0 = plan.global_k_chunk_at(chunk) * Sk_chunk_t;
                        }
                        recipe_ring_mask.ring_id = ref.source_ring_id;
                        run_chunk(state, last_k, last_k);
                    });
                continue;
            }

            // The K chunks this Q chunk processes: those the reader sends, less (on the diagonal shard) the chunks
            // starting past the Q chunk's last row, which are masked whole and popped unread.
            const bool diagonal = is_causal && !chunked_enabled && ring_id == ring_index;
            const uint32_t q_tile0 = chunked_enabled ? (kv_pad_rotation_enabled ? q_chunk * Sq_chunk_t
                                                                                : chunked_q_tile0 + q_chunk * Sq_chunk_t)
                                                     : q_chunk * Sq_chunk_t;
            const uint32_t k_end = diagonal ? (q_tile0 + Sq_chunk_t + Sk_chunk_t - 1) / Sk_chunk_t : num_kv_chunks;
            uint32_t live_chunks = 0;
            for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
                live_chunks += (chunk_sent(k_chunk) && k_chunk < k_end) ? 1 : 0;
            }
            ASSERT(live_chunks > 0);
            uint32_t k_chunk = 0;
            recipe_ring_q_block<Sq_chunk_t, vDHt>(
                resident,
                staged,
                first_active_iter,
                q_last,
                q_flat,
                next_block_unrestored,
                live_chunks,
                [&](RecipeAccumulatorState& state, uint32_t, bool last_k) {
                    // The next chunk the reader sent (processed chunks precede the masked-whole ones).
                    while (!chunk_sent(k_chunk)) {
                        k_chunk++;
                    }
                    const bool joint = k_chunk >= num_local_k_chunks;
                    const uint32_t origin =
                        (joint ? k_chunk - num_local_k_chunks : k_chunk) * k_chunk_rows;
                    const uint32_t rows = joint ? joint_rows : shard_rows;
                    const uint32_t valid = origin >= rows ? 0 : (rows - origin < k_chunk_rows ? rows - origin : k_chunk_rows);
                    bool positional = false;
                    if constexpr (recipe::ring_mask == 1) {
                        // The shard-frame diagonal: some key of the chunk lies past the Q chunk's first row.
                        positional = diagonal && !joint && k_chunk * Sk_chunk_t + Sk_chunk_t > q_tile0;
                    } else if constexpr (recipe::ring_mask == 2) {
                        if (!joint) {
                            if constexpr (kv_pad_rotation_enabled) {
                                positional = true;
                            } else {
                                // Global keys grow with the local tile: the chunk's last key against the first Q row.
                                const uint32_t last_local = k_chunk * Sk_chunk_t + Sk_chunk_t <= kv_local_padded_Nt
                                                                ? k_chunk * Sk_chunk_t + Sk_chunk_t - 1
                                                                : kv_local_padded_Nt - 1;
                                positional = chunked_kv_global_tile_for_local(
                                                 ring_id,
                                                 last_local,
                                                 local_n.q_chunk_group_tile_count,
                                                 local_n.kv_region_Nt) >= q_tile0;
                            }
                        }
                    }
                    recipe_ring_mask.valid_rows = valid;
                    recipe_ring_mask.positional = positional;
                    recipe_ring_mask.masked = positional || valid < k_chunk_rows;
                    recipe_ring_mask.q_tile0 = q_tile0;
                    recipe_ring_mask.k_tile0 = k_chunk * Sk_chunk_t;
                    recipe_ring_mask.ring_id = ring_id;
                    chunks_popped++;
                    k_chunk++;
                    run_chunk(state, q_last && last_k, last_k && (staged || q_last));
                });
            // The masked-whole chunks after the last processed one.
            for (; k_chunk < num_kv_chunks; ++k_chunk) {
                if (chunk_sent(k_chunk)) {
                    recipe_ring_pop_chunk<k_chunk_tiles, v_chunk_tiles>();
                    chunks_popped++;
                }
            }
        }
        if constexpr (!has_sliding_window) {
            // Padded multicast slots: the reader runs its multicast group's longest Q count and pushes the K/V chunks
            // each padded slot receives, as for a real Q chunk (none for one it skips), so this core's CBs advance with
            // the injector's. Pop them unread.
            for (uint32_t slot = 0; slot < padded_q_slots; ++slot) {
                const uint32_t q_flat = padded_slot_flat_q<false>({}, global_q_start, q_per_core, q_per_core + slot, 1);
                const uint32_t q_chunk = remap_q_index(q_flat, num_q_chunks, use_zigzag_balancing) % num_q_chunks;
                if (skip_early && q_chunk < num_q_chunks / 2) {
                    continue;
                }
                for (uint32_t k_chunk = 0; k_chunk < num_kv_chunks; ++k_chunk) {
                    if (chunk_sent(k_chunk)) {
                        recipe_ring_pop_chunk<k_chunk_tiles, v_chunk_tiles>();
                        chunks_popped++;
                    }
                }
            }
            // Dummy K/V traffic aligns the chained readers' CB phases (none with sliding: independent per-core reads).
            for (uint32_t dummy = 0; dummy < dummy_kv_chunks_for_phase_alignment<false, false>(chunks_popped); ++dummy) {
                recipe_ring_pop_chunk<k_chunk_tiles, v_chunk_tiles>();
            }
        }
    }
}
