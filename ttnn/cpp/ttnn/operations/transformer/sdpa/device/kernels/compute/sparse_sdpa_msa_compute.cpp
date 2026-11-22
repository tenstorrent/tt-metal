// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// sparse_sdpa_msa compute on the precision recipes (host: sparse_sdpa_msa_program_factory.cpp). The recipe core loop
// (recipe/recipe_streaming.hpp) is dense SDPA's; this kernel is the MSA outer loop: per token, tilize the Q rows (one
// row per head) and run each selected pre-tiled K/V block (one recipe chunk; the reader lays K out as K^T) through the
// recipe step, streamed or read in place from the reader's per-core block cache. Causal: the diagonal block masks the
// keys after the query's position (pack-thread key mask, recipe_tail.hpp).

#include <cstdint>

// The tilize and the recipe step reconfigure many times: the LLK Src zero-flag configurator stays out of line.
#define LLK_ZEROFLAG_OUTLINE 1

// Optimization: unpack and pack at the kernel's level (the host picks it per recipe; watcher builds: -Os), math at -O2
// (the fast untilize's LLK needs its constants folded, which -Os does not do). Blackhole only.
#if defined(TRISC_MATH)
#pragma GCC optimize("O2")
#elif defined(WATCHER_ENABLED)
#pragma GCC optimize("Os")
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
#include "sparse_recipe.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa/device/kernels/sparse_sdpa_msa_common.hpp"

namespace {
namespace ctrl = sparse_sdpa_msa::ctrl;
namespace cb = sparse_sdpa_msa::recipe_cb;
using namespace recipe_ring_cb;

constexpr uint32_t q_tiles = get_named_compile_time_arg_val("q_tiles");        // Q tile rows (with padding)
constexpr uint32_t head_tiles = get_named_compile_time_arg_val("head_tiles");  // the token's head tile rows
constexpr uint32_t k_tiles = get_named_compile_time_arg_val("k_tiles");        // one block
constexpr uint32_t d_tiles = get_named_compile_time_arg_val("d_tiles");
constexpr uint32_t vd_tiles = get_named_compile_time_arg_val("vd_tiles");
constexpr uint32_t scale_fp32 = get_named_compile_time_arg_val("scale");
constexpr bool causal = get_named_compile_time_arg_val("causal") != 0;
// Per-core K/V block cache: the K and V CBs hold kv_cache_slots blocks each. The reader fills a slot, then pushes one
// block's worth of each CB without writing it (keeping the CB counters in step) and names the slot over cb::slot;
// compute points the CBs' read pointers at the slot before the step (only UNPACK reads them) and the step's waits and
// pops balance those pushes.
constexpr uint32_t kv_cache_slots = get_named_compile_time_arg_val("kv_cache_slots");
static_assert(recipe::ring && recipe::ring_mask == 0 && !recipe::ring_stream_state, "the host's MSA recipe");
constexpr uint32_t block_keys = k_tiles * tt::constants::TILE_WIDTH;

// UNPACK only: point cb's read pointer at slot `slot` of its block slots.
ALWI void point_at_slot(uint32_t cb_id, uint32_t slot, uint32_t block_tiles) {
    UNPACK({
        auto& iface = get_local_cb_interface(cb_id);
        iface.fifo_rd_ptr = (iface.fifo_limit - iface.fifo_size) + slot * block_tiles * iface.fifo_page_size;
    })
}
}  // namespace

void kernel_main() {
    const uint32_t tok_count = get_arg_val<uint32_t>(1);

    compute_kernel_hw_startup<SrcOrder::Reverse>(q, k, out);
    if constexpr (recipe::lofi) {
        recipe_mm_init(q, k, true, 1, 1, 1);
    } else {
        matmul_init(q, k);
    }
    // The reduce scaler and the column identity stay fronted for the whole kernel.
    CircularBuffer(identity_scale).wait_front(1);
    CircularBuffer(col_identity).wait_front(1);
    init_sdpa_streaming_semaphores();
    sparse_recipe::restore_state();

    RecipeAccumulatorState state;
    for (uint32_t tok = 0; tok < tok_count; ++tok) {
        sparse_recipe::tilize_q<cb::q_rm, d_tiles, q_tiles>();
        sparse_recipe::restore_state();
        // The reader's control page: selected blocks, and (causal) the diagonal block's chunk and the first future
        // key within it.
        CircularBuffer(cb::ctrl).wait_front(1);
        const uint32_t active_blocks = ckernel::read_tile_value(cb::ctrl, 0, ctrl::ACTIVE_BLOCKS);
        [[maybe_unused]] uint32_t diag_chunk = 0xFFFFFFFFu, first_future = block_keys;
        if constexpr (causal) {
            diag_chunk = ckernel::read_tile_value(cb::ctrl, 0, ctrl::DIAG_CHUNK);
            first_future = ckernel::read_tile_value(cb::ctrl, 0, ctrl::BOUNDARY_TILE) * tt::constants::TILE_WIDTH +
                           ckernel::read_tile_value(cb::ctrl, 0, ctrl::BOUNDARY_COL);
        }
        CircularBuffer(cb::ctrl).pop_front(1);
        recipe_ring_reset_state(state);
        for (uint32_t chunk = 0; chunk < active_blocks; ++chunk) {
            if constexpr (kv_cache_slots > 0) {
                CircularBuffer(cb::slot).wait_front(1);
                const uint32_t slot = ckernel::read_tile_value(cb::slot, 0, 0);
                point_at_slot(k, slot, k_tiles * d_tiles);
                point_at_slot(v, slot, k_tiles * vd_tiles);
            }
            // The diagonal block's keys from the query's position on are in its future.
            const bool masked = causal && chunk == diag_chunk && first_future < block_keys;
            recipe_ring_mask.valid_rows = masked ? first_future : block_keys;
            recipe_ring_mask.masked = masked;
            const bool last = chunk + 1 == active_blocks;
            sparse_recipe::run_chunk<q_tiles, k_tiles, d_tiles, vd_tiles, scale_fp32, v>(state, last, last);
            if constexpr (kv_cache_slots > 0) {
                CircularBuffer(cb::slot).pop_front(1);  // the reader may now refill this block's slot
            }
        }
        sparse_recipe::emit_rows<q_tiles, vd_tiles, cb::out_rm>(head_tiles);
    }
}
