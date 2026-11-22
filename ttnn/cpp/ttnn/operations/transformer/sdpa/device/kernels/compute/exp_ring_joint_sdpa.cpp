// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exp ring joint SDPA compute on the precision recipes (STANDARD, BALANCED, ACCURATE, FAST; host:
// exp_ring_joint_sdpa_program_factory.cpp). The recipe core loop (recipe/recipe_streaming.hpp) is dense SDPA's; this
// kernel is its exp ring outer loop. A core walks its head-segments as serial passes, pass-outer and ring-inner (as the
// reader and writer): each pass keeps one Q chunk and one recurrent state resident in L1 across the pass's active ring
// iterations (recipe/recipe_ring.hpp), and only its last iteration normalizes. A pass processes exactly the K chunks
// the reader sends it (the logical_n skip); key padding (the shard's local padding, the logical_n tail, the joint tail)
// is masked in the pack thread (recipe/recipe_tail.hpp: RecipeRingMask).

#include <cstdint>

// This kernel reconfigs many times; inlining the LLK Src zero-flag DEFAULT configurator at each site pushes the program
// over the kernel-config buffer. Force it out-of-line here.
#define LLK_ZEROFLAG_OUTLINE 1

// Optimization: unpack and pack at the kernel's level, which the host picks per recipe (O2 where -O3 overflows the
// kernel config buffer); math at -O2. Watcher builds are size-optimized on every thread. Per-thread levels only the
// preprocessor can select. Blackhole only (no Wormhole exp ring).
#if defined(WATCHER_ENABLED)
#pragma GCC optimize("Os")
#elif defined(TRISC_MATH)
#pragma GCC optimize("O2")
#endif

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/tile_constants.h"
#include "recipe/recipe_block_ops.hpp"
#include "recipe/recipe_tail.hpp"
#include "recipe/recipe_sfpu.hpp"
#include "recipe/recipe_streaming.hpp"
#include "recipe/recipe_ring.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/dataflow/exp_fused_op_indexer.hpp"
#include "cpp/ttnn/operations/transformer/sdpa/device/kernels/ring_joint_derived_slots.hpp"

namespace ring_joint = ttnn::operations::transformer::sdpa::ring_joint;

namespace {
// Positional compile-time args (factory: compute_compile_time_args).
constexpr uint32_t DHt = get_compile_time_arg_val(0);
constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(1);
constexpr uint32_t Sk_chunk_t = get_compile_time_arg_val(2);
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(11);

// One K chunk of a pass's state through the recipe core (the hot path, at the kernel's optimization level). The ring
// control code below calls it per chunk.
__attribute__((noinline)) void exp_ring_run_chunk(RecipeAccumulatorState& state, bool last) {
    using namespace recipe_ring_cb;
    // FP32 recipes process one Q tile row per group; paired BF16 recipes two (an odd chunk ends with one row).
    constexpr uint32_t subblock_h = recipe::fp32 ? 1 : 2;
    sdpa_recipe_segment<
        Sq_chunk_t,
        Sk_chunk_t,
        DHt,
        DHt,
        scale_fp32,
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
        true>(state, 1, last, last);
}
}  // namespace

void kernel_main() {
    constexpr uint32_t local_padded_N = get_compile_time_arg_val(3);
    constexpr uint32_t local_padded_Nt = get_compile_time_arg_val(4);
    // With a device-tensor logical_n these are the worst-case placeholders; the live values arrive below.
    constexpr uint32_t logical_n_ct = get_compile_time_arg_val(5);
    constexpr uint32_t logical_nt_ct = get_compile_time_arg_val(6);
    constexpr uint32_t L = get_compile_time_arg_val(7);
    constexpr uint32_t num_local_k_chunks = get_compile_time_arg_val(8);
    constexpr uint32_t num_joint_k_chunks = get_compile_time_arg_val(9);
    constexpr uint32_t ring_size = get_compile_time_arg_val(10);
    constexpr bool has_logical_n_tensor = get_compile_time_arg_val(12) == 1;
    static_assert(recipe::ring && recipe::ring_mask == 0 && !recipe::ring_stream_state, "the host's exp ring recipe");
    constexpr uint32_t k_chunk_rows = Sk_chunk_t * tt::constants::TILE_HEIGHT;
    constexpr uint32_t kv_chunk_tiles = Sk_chunk_t * DHt;

    uint32_t argidx = 0;
    // Head-serial passes: this core owns one Q chunk per pass; the reader pushes each pass's chunk into the single-slot
    // Q CB, so only the count matters here.
    argidx += 2;  // q_base, q_stride
    const uint32_t q_count = get_arg_val<uint32_t>(argidx++);
    RingSDPAOpIndexer fused_op_indexer(argidx);

    // The live length from the reader (compute cannot NoC-read DRAM), read once so compute and the reader derive the
    // same chunk skips. Must precede compute_kernel_hw_startup: read_tile_value rendezvouses UNPACK -> MATH/PACK
    // through the mailboxes.
    uint32_t logical_n = logical_n_ct;
    uint32_t logical_nt = logical_nt_ct;
    if constexpr (has_logical_n_tensor) {
        constexpr uint32_t derived_cb = recipe_ring_cb::derived;
        CircularBuffer derived(derived_cb);
        derived.wait_front(1);
        logical_nt = ckernel::read_tile_value(derived_cb, 0, ring_joint::kDerivedLogicalNt);
        const uint32_t partial_col = ckernel::read_tile_value(derived_cb, 0, ring_joint::kDerivedGlobalNPartialCol);
        derived.pop_front(1);
        // Exact inverse of the reader's (logical_nt, partial_col) derivation.
        logical_n = logical_nt == 0 ? 0u
                                    : (logical_nt - 1) * ring_joint::kTileHeight +
                                          (partial_col == 0 ? ring_joint::kTileHeight : partial_col);
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

    // Same activity rule as the reader and writer: a ring iteration works iff its shard has a key before logical_n or
    // it carries the joint K/V.
    const uint32_t last_active_ring_iter =
        find_last_active_ring_iter(fused_op_indexer.seq, local_padded_Nt, logical_nt, L);
    RecipeAccumulatorState state;
    for (uint32_t pass = 0; pass < q_count; ++pass) {
        recipe_ring_reset_state(state);
        RingIdSequencer order = fused_op_indexer.seq;
        bool seen_active_iter = false;
        for (uint32_t ring_iter = 0; ring_iter < ring_size; ++ring_iter) {
            const uint32_t ring_id = order.get_next_ring_id([](uint32_t, uint32_t) {});
            const bool do_joint_kv = ring_id == ring_size - 1;
            if (ring_id * local_padded_Nt >= logical_nt && !(do_joint_kv && L != 0)) {
                continue;
            }
            // The spatial chunks the reader sends: those starting before logical_n (they come first), then the joint
            // chunks (never skipped).
            const uint32_t shard_tile0 = ring_id * local_padded_Nt;
            const uint32_t live_tiles =
                logical_nt - shard_tile0 < local_padded_Nt ? logical_nt - shard_tile0 : local_padded_Nt;
            const uint32_t spatial_chunks = shard_tile0 >= logical_nt ? 0 : (live_tiles + Sk_chunk_t - 1) / Sk_chunk_t;
            const uint32_t chunks = spatial_chunks + (do_joint_kv ? num_joint_k_chunks : 0);
            // Valid key rows of this shard: its local padding and the logical_n tail.
            const uint32_t n_origin = ring_id * local_padded_N;
            const uint32_t shard_rows = logical_n <= n_origin                   ? 0
                                        : logical_n - n_origin < local_padded_N ? logical_n - n_origin
                                                                                : local_padded_N;
            const bool last_iteration = ring_iter == last_active_ring_iter;
            recipe_ring_q_block<Sq_chunk_t, DHt>(
                state,
                false,
                !seen_active_iter,
                last_iteration,
                0,
                false,
                chunks,
                [&](RecipeAccumulatorState& s, uint32_t chunk, bool last_k) __attribute__((always_inline)) {
                    const bool joint = chunk >= spatial_chunks;
                    const uint32_t origin = (joint ? chunk - spatial_chunks : chunk) * k_chunk_rows;
                    const uint32_t rows = joint ? L : shard_rows;
                    const uint32_t valid =
                        origin >= rows ? 0 : (rows - origin < k_chunk_rows ? rows - origin : k_chunk_rows);
                    recipe_ring_mask.valid_rows = valid;
                    recipe_ring_mask.masked = valid < k_chunk_rows;
                    exp_ring_run_chunk(s, last_iteration && last_k);
                });
            seen_active_iter = true;
            // The reader's per-(pass, iteration) phase-alignment K/V pair, sent iff the chunk count is even.
            if (chunks % 2 == 0) {
                recipe_ring_pop_chunk<kv_chunk_tiles, kv_chunk_tiles>();
            }
        }
    }
}
