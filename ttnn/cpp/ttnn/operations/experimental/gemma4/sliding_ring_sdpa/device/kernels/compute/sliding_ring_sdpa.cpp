// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Gemma4 sliding ring SDPA compute on the precision recipes (host: sliding_ring_sdpa_program_factory.cpp). The recipe
// core loop (recipe/recipe_streaming.hpp) is dense SDPA's; this kernel is the shared ring joint wrapper's sliding pass
// (ring_joint_sdpa.cpp) plus lanes: each Q chunk runs one pass over its sliding work plan (the local slab and the
// predecessor halo, in the reader's order) and normalizes on its last K chunk. Masks are the recipe's sequence-frame
// ring masks (recipe/recipe_tail.hpp: RecipeRingMask, causal window k > q - W, KV-pad rotation, circular KV).

#include <cstdint>

// This kernel reconfigs ~30x; inlining the LLK Src zero-flag DEFAULT configurator at each site pushes
// the program over the kernel-config buffer. Force it out-of-line here.
#define LLK_ZEROFLAG_OUTLINE 1

// Optimization levels as the shared ring joint wrapper (ring_joint_sdpa.cpp): Blackhole unpack and pack at the kernel's
// level, math at -O2; Wormhole math at -Os, unpack and pack at -O2; watcher builds size-optimized but Wormhole's unpack.
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
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/compute_common.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/recipe/recipe_tail.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/recipe/recipe_sfpu.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/recipe/recipe_streaming.hpp"
#include "ttnn/operations/transformer/sdpa/device/kernels/compute/recipe/recipe_ring.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"
#include "cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp"
#include "cpp/ttnn/operations/experimental/gemma4/sliding_ring_sdpa/device/kernels/sliding_ring_sdpa_derived_slots.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/sliding_window_work_plan.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

namespace {
// Lanes: several requests stacked in one call. A lane owns NH * num_q_chunks consecutive flat units and has its own
// logical length and KV-pad Q mapping (the reader derives them from the metadata tensors, cb derived). At file scope,
// not in kernel_main's frame, which lies under the whole recipe step on the TRISC stacks.
struct LaneQMapping {
    uint32_t q_pre_wrap_start_tile;
    uint32_t q_pre_wrap_tile_count;
    uint32_t q_post_wrap_start_tile;
    uint32_t q_valid_tile_count;
};
uint32_t lane_logical_nt[ring_joint::kMaxLanes];
LaneQMapping lane_q_mapping[ring_joint::kMaxLanes];
// Q tile rows of the current Q chunk that have seen a visible key (RecipeRingMask::reduce).
uint32_t seeded_rows;

// One K chunk of a Q chunk's state through the recipe core (the hot path, at the kernel's optimization level).
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
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(5);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(14);
    constexpr uint32_t ring_size = get_compile_time_arg_val(15);
    constexpr bool is_causal = get_compile_time_arg_val(30) == 1;
    constexpr bool is_balanced = get_compile_time_arg_val(31) == 1;
    constexpr bool use_zigzag_balancing = get_compile_time_arg_val(32) == 1;
    constexpr bool chunked_enabled = get_compile_time_arg_val(33) == 1;
    constexpr uint32_t chunk_size_t = get_compile_time_arg_val(34);
    constexpr bool kv_pad_rotation_enabled = get_compile_time_arg_val(35) == 1;
    constexpr bool use_attention_sink = get_compile_time_arg_val(37) == 1;
    constexpr uint32_t sliding_window_size = get_compile_time_arg_val(38);
    constexpr bool kv_pad_from_metadata = get_compile_time_arg_val(39) == 1;
    constexpr uint32_t full_mesh_rank_mapping = get_compile_time_arg_val(42);
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(43));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(44);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(45);
    constexpr uint32_t circular_kv_slab_count = get_compile_time_arg_val(46);
    constexpr bool has_logical_length_tensor = get_compile_time_arg_val(47) == 1 || get_compile_time_arg_val(48) == 1;

    constexpr uint32_t num_lanes = get_named_compile_time_arg_val("num_lanes");
    constexpr uint32_t kv_stripe_split = get_named_compile_time_arg_val("kv_stripe_split");
    constexpr uint32_t sliding_max_source_ranges = get_named_compile_time_arg_val("sliding_max_source_ranges");
    constexpr bool sliding_local_first = get_named_compile_time_arg_val("sliding_local_first") == 1;
    static_assert(
        kv_stripe_split > 0 && q_local_padded_Nt % kv_stripe_split == 0, "Q slab must split into whole KV regions");
    constexpr uint32_t kv_region_Nt = q_local_padded_Nt / kv_stripe_split;
    static_assert(kv_region_Nt * ring_size == chunk_size_t, "chunk_size_t must span one KV region per ring rank");
    // The host builds this op only for causal chunked sliding attention over separate K and V, without joint tokens,
    // an attention sink or logical-length tensors (validate_on_program_cache_miss). Slot 30 is the shard-frame causal
    // diagonal, off for chunked prefill (causality is the sequence-frame mask's).
    static_assert(sliding_window_size > 0 && chunked_enabled && !is_causal && !is_balanced);
    static_assert(num_joint_k_chunks == 0 && !use_attention_sink && !has_logical_length_tensor);
    static_assert(num_lanes >= 1 && num_lanes <= ring_joint::kMaxLanes);
    static_assert(num_lanes == 1 || kv_pad_from_metadata, "Lanes take their lengths from the metadata tensors");
    static_assert(recipe::ring_mask == 2, "the sequence-frame positional masks");
    static_assert(recipe::ring_window == sliding_window_size && recipe::ring_rotation == kv_pad_rotation_enabled);
    static_assert(recipe::ring_circular == (circular_kv_slab_count > 1));

    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t q_per_core = global_q_end - global_q_start;
    const uint32_t ring_size_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t forward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t backward_writes_expected = get_arg_val<uint32_t>(argidx++);
    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    // The Q slab's KV-pad rotation segments, read straight into the pack thread's mask frame.
    KVPadRotationContext& rotation = recipe_ring_frame.rotation;
    rotation.q_pre_wrap_start_tile = get_arg_val<uint32_t>(argidx++);
    rotation.q_pre_wrap_tile_count = get_arg_val<uint32_t>(argidx++);
    rotation.q_post_wrap_start_tile = get_arg_val<uint32_t>(argidx++);
    rotation.q_valid_tile_count = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpIndexer fused_op_indexer(
        ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);

    // Live lengths from the reader, which read the metadata tensors (compute cannot NoC-read DRAM). Must precede
    // compute_kernel_hw_startup: read_tile_value rendezvouses UNPACK -> MATH/PACK through the mailboxes.
    if constexpr (kv_pad_from_metadata) {
        constexpr uint32_t derived_cb = recipe_ring_cb::derived;
        CircularBuffer derived(derived_cb);
        derived.wait_front(1);
        logical_nt = ckernel::read_tile_value(derived_cb, 0, ring_joint::kDerivedLogicalNt);
        rotation.q_pre_wrap_start_tile = ckernel::read_tile_value(derived_cb, 0, ring_joint::kDerivedQPreWrapStartTile);
        rotation.q_pre_wrap_tile_count = ckernel::read_tile_value(derived_cb, 0, ring_joint::kDerivedQPreWrapTileCount);
        rotation.q_post_wrap_start_tile = ckernel::read_tile_value(derived_cb, 0, ring_joint::kDerivedQPostWrapStartTile);
        rotation.q_valid_tile_count = ckernel::read_tile_value(derived_cb, 0, ring_joint::kDerivedQValidTileCount);
        if constexpr (num_lanes > 1) {
            for (uint32_t lane = 0; lane < num_lanes; ++lane) {
                const uint32_t base = ring_joint::kDerivedLaneBase + lane * ring_joint::kDerivedLaneStride;
                lane_logical_nt[lane] = ckernel::read_tile_value(derived_cb, 0, base + ring_joint::kDerivedLogicalNt);
                lane_q_mapping[lane] = LaneQMapping{
                    ckernel::read_tile_value(derived_cb, 0, base + ring_joint::kDerivedQPreWrapStartTile),
                    ckernel::read_tile_value(derived_cb, 0, base + ring_joint::kDerivedQPreWrapTileCount),
                    ckernel::read_tile_value(derived_cb, 0, base + ring_joint::kDerivedQPostWrapStartTile),
                    ckernel::read_tile_value(derived_cb, 0, base + ring_joint::kDerivedQValidTileCount)};
            }
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
    recipe_ring_frame.group_tiles = chunk_size_t;
    recipe_ring_frame.region_tiles = kv_region_Nt;
    recipe_ring_frame.local_tiles = kv_local_padded_Nt;
    recipe_ring_frame.logical_tiles = logical_nt;

    const auto run_chunk = [](RecipeAccumulatorState& state, bool last, bool release_q) {
        ring_run_chunk(state, last, release_q);
    };

    // Every Q chunk normalizes in its one pass, so no state crosses Q chunks (no checkpoints).
    RecipeAccumulatorState resident;
    recipe_ring_reset_state(resident);
    const bool staged = q_per_core > 1;
    for (uint32_t q_flat = global_q_start; q_flat < global_q_end; ++q_flat) {
        const uint32_t q_unit = remap_q_index(q_flat, num_q_chunks, use_zigzag_balancing);
        const uint32_t q_chunk = q_unit % num_q_chunks;
        if constexpr (num_lanes > 1) {
            // This unit's request: its logical length and Q mapping, here and in the pack thread's mask frame.
            const uint32_t lane = q_unit / (NH * num_q_chunks);
            logical_nt = lane_logical_nt[lane];
            recipe_ring_frame.logical_tiles = logical_nt;
            rotation.q_pre_wrap_start_tile = lane_q_mapping[lane].q_pre_wrap_start_tile;
            rotation.q_pre_wrap_tile_count = lane_q_mapping[lane].q_pre_wrap_tile_count;
            rotation.q_post_wrap_start_tile = lane_q_mapping[lane].q_post_wrap_start_tile;
            rotation.q_valid_tile_count = lane_q_mapping[lane].q_valid_tile_count;
        }
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
        // Q tile rows that have seen a visible key (RecipeRingMask::reduce); a row maps to no position counts as
        // seen. Key tile k reaches every row of Q tile q iff k <= q and (q - k) * 32 < window.
        static_assert(Sq_chunk_t <= 32, "one bit per Q tile row");
        constexpr uint32_t all_rows = Sq_chunk_t == 32 ? UINT32_MAX : (1u << Sq_chunk_t) - 1;
        seeded_rows = 0;
        recipe_ring_q_block<Sq_chunk_t, vDHt>(
            resident,
            staged,
            true,
            true,
            q_flat,
            false,
            plan.total_k_chunk_count,
            [&](RecipeAccumulatorState& state, uint32_t chunk, bool last_k) __attribute__((always_inline)) {
                const auto ref = plan.k_chunk_at(chunk);
                recipe_ring_mask.valid_rows = Sk_chunk_t * tt::constants::TILE_HEIGHT;
                recipe_ring_mask.masked = true;
                recipe_ring_mask.positional = true;
                recipe_ring_mask.reduce = seeded_rows != all_rows;
                recipe_ring_mask.q_tile0 = q_tile0;
                recipe_ring_mask.k_tile0 = ref.source_k_chunk * Sk_chunk_t;
                if constexpr (circular_kv_slab_count > 1) {
                    recipe_ring_mask.k_seq_tile0 = plan.global_k_chunk_at(chunk) * Sk_chunk_t;
                }
                recipe_ring_mask.ring_id = ref.source_ring_id;
                if (seeded_rows != all_rows) {
                    for (uint32_t r = 0; r < Sq_chunk_t; ++r) {
                        const uint32_t q = q_global_tile_for_mask_row<kv_pad_rotation_enabled>(r, q_tile0, rotation);
                        if (q == KV_PAD_ROTATION_INVALID_TILE) {
                            seeded_rows |= 1u << r;
                            continue;
                        }
                        for (uint32_t t = 0; t < Sk_chunk_t; ++t) {
                            const uint32_t local = recipe_ring_mask.k_tile0 + t;
                            if (kv_pad_rotation_enabled && local >= kv_local_padded_Nt) {
                                break;
                            }
                            const uint32_t k = circular_kv_slab_count > 1
                                                   ? recipe_ring_mask.k_seq_tile0 + t
                                                   : chunked_kv_global_tile_for_local(
                                                         ref.source_ring_id, local, chunk_size_t, kv_region_Nt);
                            if (k <= q && (q - k) * tt::constants::TILE_HEIGHT < sliding_window_size) {
                                seeded_rows |= 1u << r;
                                break;
                            }
                        }
                    }
                }
                run_chunk(state, last_k, last_k);
            });
    }
}
