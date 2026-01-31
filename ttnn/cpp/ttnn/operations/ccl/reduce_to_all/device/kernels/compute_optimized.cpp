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

// Include SDPA LLK APIs for srcB reuse pattern
#include "models/demos/deepseek_v3_b1/kernel_includes/tt_metal/include/compute_kernel_api/sdpa.h"

// =============================================================================
// Fused SFPI kernel for SDPA reduction
// =============================================================================
// This kernel computes in a single DST register pass:
//   cur_max = max(prev_max, worker_max)
//   P1 = exp((prev_max - cur_max) * scale)
//   P2 = exp((worker_max - cur_max) * scale)
//   cur_sum = P2 * worker_sum + P1 * prev_sum
//
// If final_norm=true:
//   P1 = P1 / cur_sum
//   P2 = P2 / cur_sum
// (Normalization factors applied to L tiles via broadcast multiply)
//
// DST register layout (combined MS tiles):
//   dst_reg[0] = prev_max (col 0 of tile 0)
//   dst_reg[1] = prev_sum (col 1 of tile 0)
//   dst_reg[32] = worker_max (col 0 of tile 1)
//   dst_reg[33] = worker_sum (col 1 of tile 1)
//   dst_reg[64] = cur_max (output, col 0 of tile 2)
//   dst_reg[65] = cur_sum (output, col 1 of tile 2)

#ifdef TRISC_MATH
template <bool SDPA_EXP_APPROX_MODE, bool final_norm = false>
void calculate_fused_max_sub_exp_add_tile(int scale_bf16) {
    static_assert(!(final_norm && SDPA_EXP_APPROX_MODE), "Approx mode must be disabled when final_norm is true");

    // 8 rows in half-tile, 2 iterations to cover full tile height
    constexpr int ITERATIONS_HALF_FACE = 2;
    constexpr uint32_t prev_max_base_idx = 0;     // Tile 0, col 0
    constexpr uint32_t prev_sum_base_idx = 1;     // Tile 0, col 1
    constexpr uint32_t worker_max_base_idx = 32;  // Tile 1, col 0
    constexpr uint32_t worker_sum_base_idx = 33;  // Tile 1, col 1
    constexpr uint32_t cur_max_base_idx = 64;     // Tile 2, col 0 (output)
    constexpr uint32_t cur_sum_base_idx = 65;     // Tile 2, col 1 (output)

    for (int d = 0; d < ITERATIONS_HALF_FACE; d++) {
        // Load inputs
        sfpi::vFloat prev_max_vec = sfpi::dst_reg[prev_max_base_idx];
        sfpi::vFloat worker_max_vec = sfpi::dst_reg[worker_max_base_idx];
        sfpi::vFloat prev_sum_vec = sfpi::dst_reg[prev_sum_base_idx];
        sfpi::vFloat worker_sum_vec = sfpi::dst_reg[worker_sum_base_idx];

        // cur_max = max(prev_max, worker_max)
        sfpi::vFloat cur_max;
        v_if(prev_max_vec < worker_max_vec) { cur_max = worker_max_vec; }
        v_else { cur_max = prev_max_vec; }
        v_endif;

        if constexpr (!final_norm) {
            sfpi::dst_reg[cur_max_base_idx] = cur_max;
        }

        // Compute scaled exponentials
        sfpi::vFloat diff_prev = prev_max_vec - cur_max;
        sfpi::vFloat diff_worker = worker_max_vec - cur_max;

        sfpi::vFloat exp_prev = ckernel::sfpu::
            _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_prev, scale_bf16);
        sfpi::vFloat exp_worker = ckernel::sfpu::
            _calculate_exponential_piecewise_<EXP_APPROX_MODE, true /*SCALE_EN*/, true /*SKIP_POSITIVE_CHECK*/>(
                diff_worker, scale_bf16);

        if constexpr (!final_norm) {
            // cur_sum = P2 * worker_sum + P1 * prev_sum
            sfpi::dst_reg[cur_sum_base_idx] = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            // Store P1, P2 for L tile broadcast multiply
            sfpi::dst_reg[prev_max_base_idx] = exp_prev;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker;
        } else {
            // Final normalization: divide by sum
            sfpi::vFloat curr_sum = exp_worker * worker_sum_vec + exp_prev * prev_sum_vec;
            sfpi::vFloat recip_sum = ckernel::sfpu::sfpu_reciprocal<SDPA_EXP_APPROX_MODE>(curr_sum);
            sfpi::dst_reg[prev_max_base_idx] = exp_prev * recip_sum;
            sfpi::dst_reg[worker_max_base_idx] = exp_worker * recip_sum;
        }
        sfpi::dst_reg += 2;
    }
}

template <bool SDPA_EXP_APPROX_MODE, int vector_mode = (int)VectorMode::C, bool final_norm = false>
void fused_max_sub_exp_add_tile(uint32_t idst, int scale_bf16) {
    _llk_math_eltwise_unary_sfpu_params_<false /*APPROXIMATE*/>(
        calculate_fused_max_sub_exp_add_tile<SDPA_EXP_APPROX_MODE, final_norm>, idst, vector_mode, scale_bf16);
}
#endif

// =============================================================================
// SDPA Tail Reduction
// =============================================================================
// Combines fused SFPI kernel with SRCB reuse broadcast multiply for L tiles.
//
// Parameters:
//   cb_worker_ms: Neighbor MS tile (worker)
//   cb_prev_ms: Local MS tile (prev)
//   cb_cur_ms: Output MS tile (only used when normalize=false)
//   cb_l1: Local L tiles
//   cb_l2: Neighbor L tiles
//   cb_l_out: Output L tiles

template <
    bool normalize,
    uint32_t block_size,
    uint32_t num_blocks,
    uint32_t scale_fp32,
    int vector_mode = (int)VectorMode::C>
void sdpa_tail(
    uint32_t cb_worker_ms, uint32_t cb_prev_ms, uint32_t cb_cur_ms, uint32_t cb_l1, uint32_t cb_l2, uint32_t cb_l_out) {
    copy_tile_to_dst_init_short(cb_worker_ms);

    cb_wait_front(cb_worker_ms, 1);
    cb_wait_front(cb_prev_ms, 1);

    constexpr uint32_t dst_reg_0 = 0;  // prev_ms
    constexpr uint32_t dst_reg_1 = 1;  // worker_ms
    constexpr uint32_t dst_reg_2 = 2;  // cur_ms output

    // Convert scale from fp32 to bf16
    constexpr uint16_t scale_bf16 = scale_fp32 >> 16;

    tile_regs_acquire();
    copy_tile(cb_prev_ms, 0, dst_reg_0);
    copy_tile(cb_worker_ms, 0, dst_reg_1);
    MATH((fused_max_sub_exp_add_tile<EXP_APPROX_MODE, vector_mode, normalize>(0, scale_bf16)));
    ckernel::sdpa_bcast_col_reuse_preamble<normalize>();

    // Pack output MS if not final reduction
    if constexpr (!normalize) {
        tile_regs_commit();
        cb_reserve_back(cb_cur_ms, 1);
        tile_regs_wait();
        pack_tile(dst_reg_2, cb_cur_ms);
        cb_push_back(cb_cur_ms, 1);
        tile_regs_release();
    }

    // Initialize SRCB reuse for L tile broadcast multiply
    ckernel::sdpa_mul_bcast_col_reuse_tiles_init<block_size>(cb_l1);

    cb_wait_front(cb_l2, num_blocks * block_size);
    cb_wait_front(cb_l1, num_blocks * block_size);
    cb_reserve_back(cb_l_out, num_blocks * block_size);

    // Compute l_out = l1 * P1 + l2 * P2
    // For final reduction, first block computed without spilling (regs already acquired)
    if constexpr (normalize) {
        ckernel::sdpa_mul_bcast_col_reuse_tiles<block_size>(cb_l2, cb_l1, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_block(0, cb_l_out, block_size);
        tile_regs_release();
    }

    for (uint32_t i = (normalize ? 1 : 0); i < num_blocks; i++) {
        tile_regs_acquire();
        ckernel::sdpa_mul_bcast_col_reuse_tiles<block_size>(cb_l2, cb_l1, i * block_size, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile_block(0, cb_l_out, block_size);
        tile_regs_release();
    }

    ckernel::sdpa_bcast_col_reuse_postamble();
    cb_push_back(cb_l_out, num_blocks * block_size);

    // Pop inputs
    cb_pop_front(cb_prev_ms, 1);
    cb_pop_front(cb_worker_ms, 1);
    cb_pop_front(cb_l2, num_blocks * block_size);
    cb_pop_front(cb_l1, num_blocks * block_size);
}

// =============================================================================
// Compile-time args
// =============================================================================
namespace NAMESPACE {

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

// Compute parameters
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(10);
constexpr uint32_t block_size = get_compile_time_arg_val(11);  // tiles per row (vDHt)
constexpr uint32_t num_blocks = get_compile_time_arg_val(12);  // number of rows (Sq_chunk_t)
constexpr uint32_t cb_sync = get_compile_time_arg_val(13);
constexpr uint32_t cb_sync_writer_done = get_compile_time_arg_val(14);  // Writer -> Compute sync

void MAIN {
    constexpr int vector_mode = VectorMode::RC_custom;
    constexpr uint32_t out_tiles = block_size * num_blocks;

    binary_op_init_common(cb_local_l, cb_local_l, cb_l_out);
    exp_tile_init<EXP_APPROX_MODE, false>();

    // Debug: Print input CB addresses and first values
    DPRINT << "Compute: cb_local_l=" << cb_local_l << " cb_local_ms=" << cb_local_ms << ENDL();
    DPRINT << "Compute: cb_r1_neighbor_l=" << cb_r1_neighbor_l << " cb_r1_neighbor_ms=" << cb_r1_neighbor_ms << ENDL();
    DPRINT << "Compute: block_size=" << block_size << " num_blocks=" << num_blocks << ENDL();

    // =========================================================================
    // ROUND 1: reduce(local, r1_neighbor) → r1_result
    // =========================================================================
    // Non-final reduction: outputs L and MS
    {
        DeviceZoneScopedN("R1-SDPA-REDUCE");
        sdpa_tail<false, block_size, num_blocks, scale_fp32, vector_mode>(
            cb_r1_neighbor_ms,  // worker (neighbor)
            cb_local_ms,        // prev (local)
            cb_r1_result_ms,    // cur output MS
            cb_r1_neighbor_l,   // l1 (neighbor)
            cb_local_l,         // l2 (local)
            cb_r1_result_l);    // l_out

        DPRINT << "R1 reduction complete" << ENDL();

        // Signal writer that R1 is done
        DPRINT << "Compute pushing cb_sync" << ENDL();
        cb_reserve_back(cb_sync, 1);
        cb_push_back(cb_sync, 1);
        DPRINT << "Compute cb_sync pushed" << ENDL();
    }

    // =========================================================================
    // Wait for writer to finish reading R1 results before R2 can pop them
    // =========================================================================
    // The writer reads cb_r1_result_l and cb_r1_result_ms to build R2 packet.
    // R2 sdpa_tail uses these as inputs and pops them at the end.
    // We must wait for writer to finish its memcpy before R2 sdpa_tail pops.
    {
        DeviceZoneScopedN("R2-WAIT-WRITER");
        cb_wait_front(cb_sync_writer_done, 1);
        cb_pop_front(cb_sync_writer_done, 1);
    }

    // =========================================================================
    // ROUND 2: reduce(r1_result, r2_neighbor) → final output (normalized L)
    // =========================================================================
    // Final reduction with normalization: outputs only normalized L
    {
        DeviceZoneScopedN("R2-SDPA-REDUCE");
        sdpa_tail<true, block_size, num_blocks, scale_fp32, vector_mode>(
            cb_r2_neighbor_ms,  // worker (neighbor)
            cb_r1_result_ms,    // prev (R1 result)
            cb_ms_out,          // cur output MS (unused when final=true)
            cb_r2_neighbor_l,   // l1 (neighbor)
            cb_r1_result_l,     // l2 (R1 result)
            cb_l_out);          // l_out (normalized, aliased to output tensor)
        DPRINT << "R2 reduction complete" << ENDL();
    }

    // Output is now in cb_l_out (aliased to output_tensor_l shard)
    // No additional move needed!
}

}  // namespace NAMESPACE
