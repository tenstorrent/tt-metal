// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
///
// Optimized compute kernel for reduce_to_all operation using fused SDPA reduction.
//
// KEY OPTIMIZATIONS:
// 1. Fused SFPI kernel: max, exp, scale, add computed in single DST pass
// 2. SRCB reuse: Broadcast scalars kept in SRCB for column broadcast multiply
// 3. Combined MS format: max in col 0, sum in col 1 of single tile
// 4. Final normalization fused into R2 reduction
// 5. Streaming: L data processed in chunks to overlap with fabric transfer
//
// CHUNKING APPROACH:
// L tiles are processed in chunks to enable streaming:
// - After MS reduction (computes P1/P2), L tiles can be processed independently
// - Each L chunk is processed as soon as it arrives from fabric
// - This hides fabric transfer latency behind compute
//
// Data Flow (per core):
//   R1: sdpa_tail streaming (local + r1_neighbor) → r1_result (L + MS)
//   R2: sdpa_tail streaming (r1_result + r2_neighbor) → final_output (normalized L only)
//
// CB Layout:
//   cb_local_l, cb_local_ms: Local input (aliased to input tensor shard)
//   cb_r1_neighbor_l, cb_r1_neighbor_ms: R1 neighbor data
//   cb_r1_result_l, cb_r1_result_ms: R1 output / R2 local input
//   cb_r2_neighbor_l, cb_r2_neighbor_ms: R2 neighbor data
//   cb_l_out: Final normalized L output (aliased to output tensor)
//   cb_ms_out: R1 MS output (intermediate)
//
// SDPA Reduction Formula:
//   m = max(m1, m2)
//   P1 = exp((m1 - m) * scale)
//   P2 = exp((m2 - m) * scale)
//   s = s1 * P1 + s2 * P2
//   l = l1 * P1 + l2 * P2
//
// When final_reduction=true: l = l / s (fused into SFPI kernel)

#include <cstdint>

#define REDUCE_OP (PoolType::MAX)
#define REDUCE_DIM (ReduceDim::REDUCE_ROW)
#define EXP_APPROX_MODE false

#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/eltwise_unary/exp.h"
#include "compute_kernel_api/eltwise_unary/recip.h"
#include "compute_kernel_api/bcast.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/matmul.h"
#include "compute_kernel_api/reduce.h"
#include "compute_kernel_api/pack.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "api/debug/dprint.h"

// Include SDPA LLK APIs for srcB reuse pattern and sdpa_tail reduction
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"

// CB indices
static constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
static constexpr uint32_t cb_local_ms = get_compile_time_arg_val(1);
static constexpr uint32_t cb_r1_neighbor_l = get_compile_time_arg_val(2);
static constexpr uint32_t cb_r1_neighbor_ms = get_compile_time_arg_val(3);
static constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(4);
static constexpr uint32_t cb_r1_result_ms = get_compile_time_arg_val(5);
static constexpr uint32_t cb_r2_neighbor_l = get_compile_time_arg_val(6);
static constexpr uint32_t cb_r2_neighbor_ms = get_compile_time_arg_val(7);
static constexpr uint32_t cb_l_out = get_compile_time_arg_val(8);
static constexpr uint32_t cb_ms_out = get_compile_time_arg_val(9);  // Only used for R1 intermediate

// Compute parameters
static constexpr uint32_t scale_fp32 = get_compile_time_arg_val(10);
static constexpr uint32_t block_size =
    get_compile_time_arg_val(11);  // tiles per SDPA block (= tiles_per_l_chunk, max 8)
static constexpr uint32_t num_blocks = get_compile_time_arg_val(12);         // total blocks (= num_l_chunks)
static constexpr uint32_t num_l_chunks = get_compile_time_arg_val(13);       // number of L chunks
static constexpr uint32_t tiles_per_l_chunk = get_compile_time_arg_val(14);  // tiles per L chunk (= block_size)

/**
 * Streaming SDPA tail reduction that processes L tiles in chunks.
 *
 * This function implements the same reduction as sdpa_tail but processes
 * L tiles in chunks to enable streaming from fabric transfer.
 *
 * Flow:
 * 1. MS reduction: computes P1/P2, sets up SRCB
 * 2. For each chunk:
 *    - Cumulative wait for L tiles (ensures chunk i data is available)
 *    - Process L tiles using P1/P2 from SRCB with global tile indexing
 *    - Push output tiles (cb_push_back)
 *    - NO pop of input L CBs (allows multiple readers: compute + writer)
 * 3. Finalize: postamble + pop MS tiles
 *
 * Key design: L input CBs are NOT popped. This allows:
 * - Writer to read R1 result L tiles for R2 send (cumulative waits)
 * - Compute R2 to read R1 result L tiles (global indexing)
 * - No race condition between writer and compute
 */
template <
    bool SDPA_EXP_APPROX_MODE,
    bool normalize,
    uint32_t _block_size,
    uint32_t _num_blocks,
    uint32_t _scale_fp32,
    uint32_t _num_l_chunks,
    uint32_t _tiles_per_l_chunk,
    int vector_mode = (int)VectorMode::C>
ALWI void sdpa_tail_streaming(
    uint32_t cb_worker_max_sum,
    uint32_t cb_prev_max_sum,
    uint32_t cb_cur_max_sum,
    uint32_t cb_l1,
    uint32_t cb_l2,
    uint32_t cb_l_out) {
    // Phase 1: MS reduction - computes P1/P2, sets up SRCB
    // This also reserves regs if normalize=true for first L block
    ckernel::sdpa_tail_ms_reduce<SDPA_EXP_APPROX_MODE, normalize, _block_size, _scale_fp32, vector_mode>(
        cb_worker_max_sum, cb_prev_max_sum, cb_cur_max_sum, cb_l1);

    // Phase 2: Process L chunks as they arrive
    // Use cumulative waits and global tile indexing (no popping)
    constexpr uint32_t blocks_per_chunk = _tiles_per_l_chunk / _block_size;

    DPRINT << "num_l_chunks=" << _num_l_chunks << ", tiles_per_l_chunk=" << _tiles_per_l_chunk
           << ", blocks_per_chunk=" << blocks_per_chunk << ", _block_size=" << _block_size << ENDL();

    for (uint32_t chunk = 0; chunk < _num_l_chunks; chunk++) {
        // Cumulative wait: ensure at least (chunk+1) * tiles_per_chunk tiles are available
        // This works with writer's cumulative waits and avoids race conditions
        cb_wait_front(cb_l1, (chunk + 1) * _tiles_per_l_chunk);
        cb_wait_front(cb_l2, (chunk + 1) * _tiles_per_l_chunk);
        cb_reserve_back(cb_l_out, _tiles_per_l_chunk);

        // Process all blocks in this chunk using GLOBAL tile indices
        for (uint32_t block_in_chunk = 0; block_in_chunk < blocks_per_chunk; block_in_chunk++) {
            // Global block index (for acquire_regs logic)
            uint32_t global_block = chunk * blocks_per_chunk + block_in_chunk;
            // Global tile index (since we don't pop, read pointer stays at 0)
            uint32_t tile_index = global_block * _block_size;
            DPRINT << "Processing chunk=" << chunk << ", block_in_chunk=" << block_in_chunk
                   << ", global_block=" << global_block << ", tile_index=" << tile_index << ENDL();

            // For normalize=true, first block uses regs still held from MS phase
            bool acquire_regs = !(normalize && global_block == 0);
            DPRINT << "  acquire_regs=" << (uint32_t)acquire_regs << ENDL();
            ckernel::sdpa_tail_l_block<_block_size>(cb_l1, cb_l2, cb_l_out, tile_index, acquire_regs);
        }

        // Push output tiles (output CB is not shared, so push is fine)
        cb_push_back(cb_l_out, _tiles_per_l_chunk);
        // NOTE: No cb_pop_front for cb_l1/cb_l2 - tiles remain for other readers
    }

    // Phase 3: Finalize (postamble + pop MS)
    ckernel::sdpa_tail_finalize(cb_worker_max_sum, cb_prev_max_sum);
}

void kernel_main() {
    constexpr int vector_mode = VectorMode::RC_custom;
    constexpr uint32_t out_tiles = block_size * num_blocks;

    binary_op_init_common(cb_local_l, cb_local_l, cb_l_out);
    exp_tile_init<EXP_APPROX_MODE, false>();

    // =========================================================================
    // ROUND 1: reduce(local, r1_neighbor) → r1_result
    // =========================================================================
    // Non-final reduction: outputs L and MS (streaming)
    sdpa_tail_streaming<
        EXP_APPROX_MODE,
        false /* normalize */,
        block_size,
        num_blocks,
        scale_fp32,
        num_l_chunks,
        tiles_per_l_chunk,
        vector_mode>(
        cb_r1_neighbor_ms,  // worker (neighbor)
        cb_local_ms,        // prev (local)
        cb_r1_result_ms,    // cur output MS
        cb_r1_neighbor_l,   // l1 (neighbor)
        cb_local_l,         // l2 (local)
        cb_r1_result_l);    // l_out

    // =========================================================================
    // ROUND 2: reduce(r1_result, r2_neighbor) → final output (normalized L)
    // =========================================================================
    // Final reduction with normalization: outputs only normalized L (streaming)
    sdpa_tail_streaming<
        EXP_APPROX_MODE,
        true /* normalize */,
        block_size,
        num_blocks,
        scale_fp32,
        num_l_chunks,
        tiles_per_l_chunk,
        vector_mode>(
        cb_r2_neighbor_ms,  // worker (neighbor)
        cb_r1_result_ms,    // prev (R1 result)
        cb_ms_out,          // cur output MS (unused when final=true)
        cb_r2_neighbor_l,   // l1 (neighbor)
        cb_r1_result_l,     // l2 (R1 result)
        cb_l_out);          // l_out (normalized, aliased to output tensor)

    DPRINT << "compute done" << ENDL();
}
