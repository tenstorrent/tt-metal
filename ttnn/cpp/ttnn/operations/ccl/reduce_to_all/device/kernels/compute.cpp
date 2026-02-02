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
//
// Data Flow (per core):
//   R1: sdpa_tail<false>(local, r1_neighbor) → r1_result (L + MS)
//   R2: sdpa_tail<true>(r1_result, r2_neighbor) → final_output (normalized L only)
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
constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
constexpr uint32_t cb_local_ms = get_compile_time_arg_val(1);
constexpr uint32_t cb_r1_neighbor_l = get_compile_time_arg_val(2);
constexpr uint32_t cb_r1_neighbor_ms = get_compile_time_arg_val(3);
constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(4);
constexpr uint32_t cb_r1_result_ms = get_compile_time_arg_val(5);
constexpr uint32_t cb_r2_neighbor_l = get_compile_time_arg_val(6);
constexpr uint32_t cb_r2_neighbor_ms = get_compile_time_arg_val(7);
constexpr uint32_t cb_l_out = get_compile_time_arg_val(8);
constexpr uint32_t cb_ms_out = get_compile_time_arg_val(9);  // Only used for R1 intermediate
constexpr uint32_t cb_compute_to_writer_sync = get_compile_time_arg_val(10);  // Compute -> Writer sync
constexpr uint32_t cb_writer_to_compute_sync = get_compile_time_arg_val(11);  // Writer -> Compute sync

// Compute parameters
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(12);
constexpr uint32_t block_size = get_compile_time_arg_val(13);  // tiles per row (vDHt)
constexpr uint32_t num_blocks = get_compile_time_arg_val(14);  // number of rows (Sq_chunk_t)

void kernel_main() {
    constexpr int vector_mode = VectorMode::RC_custom;
    constexpr uint32_t out_tiles = block_size * num_blocks;

    binary_op_init_common(cb_local_l, cb_local_l, cb_l_out);
    exp_tile_init<EXP_APPROX_MODE, false>();

    // =========================================================================
    // ROUND 1: reduce(local, r1_neighbor) → r1_result
    // =========================================================================
    // Non-final reduction: outputs L and MS
    ckernel::sdpa_tail<EXP_APPROX_MODE, false /* normalize */, block_size, num_blocks, scale_fp32, vector_mode>(
        cb_r1_neighbor_ms,  // worker (neighbor)
        cb_local_ms,        // prev (local)
        cb_r1_result_ms,    // cur output MS
        cb_r1_neighbor_l,   // l1 (neighbor)
        cb_local_l,         // l2 (local)
        cb_r1_result_l);    // l_out

    // Signal writer that R1 is done
    cb_reserve_back(cb_compute_to_writer_sync, 1);
    cb_push_back(cb_compute_to_writer_sync, 1);

    // =========================================================================
    // Wait for writer to finish reading R1 results before R2 can pop them
    // =========================================================================
    // The writer reads cb_r1_result_l and cb_r1_result_ms to build R2 packet.
    // R2 sdpa_tail uses these as inputs and pops them at the end.
    // We must wait for writer to finish its memcpy before R2 sdpa_tail pops.
    cb_wait_front(cb_writer_to_compute_sync, 1);
    cb_pop_front(cb_writer_to_compute_sync, 1);

    // =========================================================================
    // ROUND 2: reduce(r1_result, r2_neighbor) → final output (normalized L)
    // =========================================================================
    // Final reduction with normalization: outputs only normalized L
    ckernel::sdpa_tail<EXP_APPROX_MODE, true /* normalize */, block_size, num_blocks, scale_fp32, vector_mode>(
        cb_r2_neighbor_ms,  // worker (neighbor)
        cb_r1_result_ms,    // prev (R1 result)
        cb_ms_out,          // cur output MS (unused when final=true)
        cb_r2_neighbor_l,   // l1 (neighbor)
        cb_r1_result_l,     // l2 (R1 result)
        cb_l_out);          // l_out (normalized, aliased to output tensor)
}
