// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Ring joint SDPA compute for the named precision recipes B/C/D/E (precision unset keeps ring_joint_sdpa.cpp).
// Every active ring contribution continues one recurrent state (streaming/recipe_ring.hpp); only the last
// active contribution normalizes. Compile-time and runtime arguments share the ring-joint layout built by
// RingJointSDPARecipeMeshWorkloadFactory. Causal and balanced attention build with SDPA_RECIPE_RING_CAUSAL
// (streaming/recipe_ring.hpp); sliding-window, chunked, KV-pad rotation and attention sinks are rejected on the host. Device-tensor logical lengths arrive from the reader through
// cb_kv_pad_derived. Any tile-aligned Q/K chunk and head dim (subblock widths: SDPA_RECIPE_QK_W/PV_W).

#include <cstdint>

// This kernel reconfigs ~30x; inlining the LLK Src zero-flag DEFAULT configurator at each site pushes
// the program over the kernel-config buffer. Force it out-of-line here.
#define LLK_ZEROFLAG_OUTLINE 1
// The recipe headers key their ring hooks (valid-row key tail masking) on SDPA_RECIPE_RING.
#define SDPA_RECIPE_RING 1

// Optimization: unpack and pack at -O3, math at -O2. Even-row Q chunks (the host rounds a paired recipe's odd Q
// chunk up) keep every recipe inside the kernel config buffer (70656 B at the default Blackhole worker L1);
// -O3 on unpack/pack is worth 10-25% of ring SDPA time over -Os, -O3 on math nothing. Watcher builds are
// size-optimized on every thread.
#if defined(WATCHER_ENABLED) || \
    (defined(ARCH_WORMHOLE) && (defined(TRISC_MATH) || (defined(TRISC_UNPACK) && defined(SDPA_RECIPE_FP32))))
// Wormhole: math, and the FP32-state recipes' unpack, are size-optimized to fit the kernel config buffer.
#pragma GCC optimize("Os")
#elif defined(TRISC_MATH) || defined(SDPA_RECIPE_FUSED) || defined(SDPA_RECIPE_RING_CAUSAL)
// Fused FAST chunks (and the causal ring step's masking and skips) add enough code that -O3 unpack/pack overflows
// the kernel config buffer.
#pragma GCC optimize("O2")
#else
#pragma GCC optimize("O3")
#endif

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include <tt-metalium/constants.hpp>
#include "sdpa_block_ops.hpp"
#include "streaming/recipe_tail.hpp"
#include "streaming/recipe_sfpu.hpp"
#include "streaming/recipe_streaming.hpp"
#include "streaming/recipe_ring.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/fused_op_indexer.hpp"
#include "cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/ring_attention_rank_mapping.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

void kernel_main() {
    constexpr uint32_t DHt = get_compile_time_arg_val(1);
    constexpr uint32_t vDHt = get_compile_time_arg_val(2);
    constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(3);
    constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(4);
    constexpr uint32_t kv_local_padded_Nt = get_compile_time_arg_val(6);
    constexpr uint32_t Lt = get_compile_time_arg_val(9);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(12);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(13);
    constexpr uint32_t ring_size = get_compile_time_arg_val(15);
    constexpr uint32_t scale_fp32 = get_compile_time_arg_val(26);
    constexpr uint32_t global_n_partial_col_ct = get_compile_time_arg_val(28);
    constexpr uint32_t joint_l_partial_col_ct = get_compile_time_arg_val(29);
    // Slot 40: sharded joint (one L/P shard arrives per ring iteration). Slot 41: logical joint tiles.
    constexpr bool joint_is_sharded = get_compile_time_arg_val(40) == 1;
    constexpr uint32_t logical_lt_ct = get_compile_time_arg_val(41);
    // Slots 42-45: transport-to-tensor rank mapping.
    constexpr bool full_mesh_rank_mapping = get_compile_time_arg_val(42) == 1;
    constexpr auto snake_orientation = static_cast<ttnn::ccl::snake_ring::Orientation>(get_compile_time_arg_val(43));
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(44);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(45);
    // Slots 30-33: causal (off for chunked prefill, which masks in the sequence's frame), balanced, the balanced zigzag
    // Q order, chunked prefill.
#ifdef SDPA_RECIPE_RING_CHUNKED
    static_assert(get_compile_time_arg_val(33) == 1, "SDPA_RECIPE_RING_CHUNKED builds serve chunked prefill");
    constexpr uint32_t q_local_padded_Nt = get_compile_time_arg_val(5);
    static_assert(q_local_padded_Nt == SDPA_RECIPE_RING_CHUNKED && get_compile_time_arg_val(34) == SDPA_RECIPE_RING_GROUP_TILES);
#elif defined(SDPA_RECIPE_RING_CAUSAL)
    static_assert(get_compile_time_arg_val(33) == 0, "Chunked ring recipes build with SDPA_RECIPE_RING_CHUNKED");
#endif
#ifdef SDPA_RECIPE_RING_CAUSAL
    static_assert(
        get_compile_time_arg_val(30) == 1 || get_compile_time_arg_val(33) == 1,
        "SDPA_RECIPE_RING_CAUSAL builds serve causal ring attention");
    constexpr bool is_balanced = get_compile_time_arg_val(31) == 1;
    constexpr bool zigzag = get_compile_time_arg_val(32) == 1;
    constexpr uint32_t NH = get_compile_time_arg_val(0);
    constexpr uint32_t num_q_chunks = get_compile_time_arg_val(14);
#else
    static_assert(
        get_compile_time_arg_val(30) == 0 && get_compile_time_arg_val(31) == 0,
        "Causal ring recipes build with SDPA_RECIPE_RING_CAUSAL");
#endif
    static_assert(
        get_compile_time_arg_val(35) == 0 && get_compile_time_arg_val(37) == 0 && get_compile_time_arg_val(38) == 0 &&
            get_compile_time_arg_val(39) == 0,
        "Named ring recipes reject KV-pad rotation, sinks and sliding windows");
    // Slots 47-48: logical_n / logical_l arrive as device tensors; the compile-time values are placeholders.
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(47) == 1;
    constexpr bool has_logical_l_tensor = get_compile_time_arg_val(48) == 1;

    constexpr bool has_joint_k = num_joint_k_chunks > 0;
    constexpr bool has_gathered_joint_k = joint_is_sharded && has_joint_k;
    // Per-device joint shard length in tiles (used for the per-ring-iteration joint tail boundary).
    constexpr uint32_t Lt_local = has_gathered_joint_k ? Lt / ring_size : Lt;

    uint32_t argidx = 0;
    const uint32_t global_q_start = get_arg_val<uint32_t>(argidx++);
    const uint32_t global_q_end = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_size_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t ring_index_runtime = get_arg_val<uint32_t>(argidx++);
    const uint32_t forward_writes_expected = get_arg_val<uint32_t>(argidx++);
    const uint32_t backward_writes_expected = get_arg_val<uint32_t>(argidx++);
    uint32_t logical_nt = get_arg_val<uint32_t>(argidx++);
    argidx += 4;  // KV-pad Q mapping (rotation is rejected)
    uint32_t active_ring_iter_mask = get_arg_val<uint32_t>(argidx++);

    RingSDPAOpIndexer fused_op_indexer(
        ring_size_runtime, ring_index_runtime, forward_writes_expected, backward_writes_expected);

    constexpr uint32_t cb_arg_offset = 49;
    constexpr uint32_t cb_q_in = get_compile_time_arg_val(cb_arg_offset + 0);
    constexpr uint32_t cb_k_in = get_compile_time_arg_val(cb_arg_offset + 1);
    constexpr uint32_t cb_identity_scale_in = get_compile_time_arg_val(cb_arg_offset + 5);
    constexpr uint32_t cb_col_identity = get_compile_time_arg_val(cb_arg_offset + 8);
    constexpr uint32_t cb_qk_im = get_compile_time_arg_val(cb_arg_offset + 15);
    constexpr uint32_t cb_out_im_A = get_compile_time_arg_val(cb_arg_offset + 16);
    constexpr uint32_t cb_out_im_B = get_compile_time_arg_val(cb_arg_offset + 17);
    constexpr uint32_t cb_max_A = get_compile_time_arg_val(cb_arg_offset + 18);
    constexpr uint32_t cb_max_B = get_compile_time_arg_val(cb_arg_offset + 19);
    constexpr uint32_t cb_sum_A = get_compile_time_arg_val(cb_arg_offset + 20);
    constexpr uint32_t cb_sum_B = get_compile_time_arg_val(cb_arg_offset + 21);
    constexpr uint32_t cb_kv_pad_derived = get_compile_time_arg_val(cb_arg_offset + 23);

    // Live lengths (and the ring work mask re-derived from them) from the reader, which read the device
    // tensors; compute cannot NoC-read DRAM. Must precede compute_kernel_hw_startup: read_tile_value
    // rendezvouses UNPACK -> MATH/PACK through the mailboxes.
    uint32_t global_n_partial_col = global_n_partial_col_ct;
    uint32_t joint_l_partial_col = joint_l_partial_col_ct;
    uint32_t logical_lt = logical_lt_ct;
    if constexpr (has_logical_n_tensor || has_logical_l_tensor) {
        CircularBuffer derived(cb_kv_pad_derived);
        derived.wait_front(1);
        logical_nt = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedLogicalNt);
        active_ring_iter_mask = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedActiveRingIterMask);
        if constexpr (has_logical_n_tensor) {
            global_n_partial_col = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedGlobalNPartialCol);
        }
        if constexpr (has_logical_l_tensor) {
            logical_lt = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedLogicalLt);
            joint_l_partial_col = ckernel::read_tile_value(cb_kv_pad_derived, 0, ring_joint::kDerivedJointLPartialCol);
        }
        derived.pop_front(1);
    }

    compute_kernel_hw_startup<SrcOrder::Reverse>(cb_q_in, cb_k_in, cb_qk_im);
#ifdef SDPA_RECIPE_LOFI
    recipe_mm_init(cb_q_in, cb_k_in, true, 1, 1, 1);  // no MOP matmul code (see normalize_row_streaming)
#else
    matmul_init(cb_q_in, cb_k_in);
#endif

    // Wait once for the reduce scaler and column identity. The recipe masks key tails from valid-row
    // counts in the pack thread, so it never reads the writer's lightweight mask tiles.
    CircularBuffer(cb_identity_scale_in).wait_front(1);

    RecipeAccumulatorState acc_state = {
        {cb_sum_A, cb_max_A, cb_out_im_A},  // prev
        {cb_sum_B, cb_max_B, cb_out_im_B},  // cur
    };
    init_sdpa_streaming_semaphores();
    CircularBuffer(cb_col_identity).wait_front(1);

#ifdef SDPA_RECIPE_RING_CAUSAL
    const uint32_t ring_index = ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
        fused_op_indexer.seq.ring_index, mesh_rows, mesh_cols, snake_orientation);
    // Balanced: the last active step on K from this or an earlier device, after which the early half's Q chunks
    // see nothing more (walks the reader's ring order without its syncs).
    uint32_t early_last_iter = 0;
    if constexpr (is_balanced) {
        RingIdSequencer order = fused_op_indexer.seq;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id =
                ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                    order.get_next_ring_id([](uint32_t, uint32_t) {}), mesh_rows, mesh_cols, snake_orientation);
            if (((active_ring_iter_mask >> ring_iter) & 1u) != 0 && ring_id <= ring_index) {
                early_last_iter = ring_iter;
            }
        }
    }
#endif

    // The first active iter starts with fresh state; restoring would read stale staging.
    bool seen_active_iter = false;
    for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
        const uint32_t ring_id =
            ttnn::ring_attention_all_gather::tensor_rank_from_transport_rank<full_mesh_rank_mapping>(
                fused_op_indexer.get_next_ring_id_and_sync(), mesh_rows, mesh_cols, snake_orientation);
        // Host precomputes which ring iterations have useful SDPA work; sync/ring-id sequencing
        // still advances above so compute stays aligned with reader, writer, and all-gather.
        if (((active_ring_iter_mask >> ring_iter) & 1u) == 0) {
            continue;
        }
        // Sharded joint: one L/P shard per ring iteration. Replicated joint: all of L on ring_id == ring_size-1.
        const bool do_joint_kv = has_gathered_joint_k ? true : (ring_id == ring_size - 1);
        const uint32_t num_kv_chunks = do_joint_kv ? num_local_k_chunks + num_joint_k_chunks : num_local_k_chunks;
        const bool is_first_active_iter = !seen_active_iter;
        seen_active_iter = true;
        const uint32_t joint_shard_base_tiles = has_gathered_joint_k ? (ring_id * Lt_local) : 0u;
        const uint32_t joint_shard_tiles = has_gathered_joint_k ? Lt_local : Lt;
        const bool is_last_ring_iter = is_last_active_ring_iter(active_ring_iter_mask, ring_iter);

        // Valid key rows of this ring contribution: the global logical_n tail (including its sub-tile
        // column) clipped to this shard, and the joint logical tail clipped to this iteration's joint shard.
        const uint32_t valid_n_rows = logical_nt * 32 - (global_n_partial_col ? 32 - global_n_partial_col : 0);
        const uint32_t n_origin = ring_id * kv_local_padded_Nt * 32;
#ifdef SDPA_RECIPE_RING_CHUNKED
        // Chunked prefill: the shard is the device's slab of every chunk group; the reader sends the chunks before
        // logical_n and the mask covers the rest (recipe_ring.hpp).
        (void)valid_n_rows;
        (void)n_origin;
        uint32_t primary_rows = kv_local_padded_Nt * 32;
#else
        uint32_t primary_rows = valid_n_rows <= n_origin                            ? 0
                                : valid_n_rows - n_origin < kv_local_padded_Nt * 32 ? valid_n_rows - n_origin
                                                                                    : kv_local_padded_Nt * 32;
#endif
#ifdef SDPA_RECIPE_RING_CAUSAL
        // Balanced, K from an earlier device: only its shard's early half precedes this device's Q (the reader
        // sends the chunks up to the one straddling the half; the tail mask covers the straddle).
        if (is_balanced && ring_index > ring_id && primary_rows > kv_local_padded_Nt * 16) {
            primary_rows = kv_local_padded_Nt * 16;
        }
        const RecipeRingCausal causal{
            .q_chunks = num_q_chunks,
            .heads = NH,
            .zigzag = zigzag,
            .diagonal = ring_id == ring_index,
            .skip_early = is_balanced && ring_index < ring_id,
            .early_last = is_balanced && ring_iter == early_last_iter,
#ifdef SDPA_RECIPE_RING_CHUNKED
            // The current chunk group ends at logical_n; the reader's dense causal skip stops at this device's last
            // Q row (ring_joint_reader_impl.hpp: causal_end_nt).
            .q_tile0 = logical_nt - SDPA_RECIPE_RING_GROUP_TILES + ring_index * q_local_padded_Nt,
            .ring_id = ring_id,
            .logical_nt = logical_nt,
            .live_end_nt = get_named_compile_time_arg_val("dense_causal_skip") == 1
                               ? logical_nt - SDPA_RECIPE_RING_GROUP_TILES + (ring_index + 1) * q_local_padded_Nt
                               : logical_nt,
            .local_tiles = kv_local_padded_Nt,
#endif
        };
#endif
        const uint32_t valid_l_rows = logical_lt * 32 - (joint_l_partial_col ? 32 - joint_l_partial_col : 0);
        const uint32_t l_origin = joint_shard_base_tiles * 32;
        const uint32_t joint_rows = !do_joint_kv || valid_l_rows <= l_origin           ? 0
                                    : valid_l_rows - l_origin < joint_shard_tiles * 32 ? valid_l_rows - l_origin
                                                                                       : joint_shard_tiles * 32;
        sdpa_recipe_ring_segment<
            Sq_chunk_t,
            scale_fp32,
#ifdef SDPA_RECIPE_FP32
            1,
#else
            2,
#endif
            Sk_chunk_t,
            DHt,
            vDHt>(
            acc_state,
            global_q_start,
            global_q_end,
            num_local_k_chunks,
            num_kv_chunks,
            primary_rows,
            joint_rows,
            is_first_active_iter,
            is_last_ring_iter
#ifdef SDPA_RECIPE_RING_CAUSAL
            ,
            causal
#endif
        );
    }
}
