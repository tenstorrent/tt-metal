// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0
///
// Simplified compute kernel for reduce_to_all operation.
// This kernel performs BOTH Round 1 and Round 2 reductions on the SAME core.
//
// Data Flow (per core):
//   R1: reduce(local_data, r1_neighbor_data) → r1_result
//   R2: reduce(r1_result, r2_neighbor_data) → final_output
//
// CB Layout:
//   R1 Inputs:
//     - cb_local_l/s/m: Local input data (aliased to input tensor shard)
//     - cb_r1_neighbor_l/s/m: R1 neighbor data (aliased to R1 MeshBuffer)
//   R1 Output / R2 Local Input:
//     - cb_r1_result_l/s/m: R1 reduction result (writer sends to R2 neighbor)
//   R2 Inputs:
//     - cb_r1_result_l/s/m: R1 result (reused as R2 local)
//     - cb_r2_neighbor_l/s/m: R2 neighbor data (aliased to R2 MeshBuffer)
//   Temp / Output CBs:
//     - cb_l_out, cb_s_out, cb_m_out: ALIASED to output tensor shard!
//       These are used as temp CBs during reduction, and the final R2 result
//       ends up here automatically. No extra move needed.
//
// ZERO-COPY OUTPUT:
//   The "temp" CBs for L/S/M results are aliased to output tensor shards.
//   After R2 reduction + normalization, the final output is already at the
//   correct memory location. No cb_push_back or move needed for output!
//
// SDPA Reduction Formula:
//   m = max(m1, m2)
//   P1 = exp((m1 - m) * scale)
//   P2 = exp((m2 - m) * scale)
//   s = s1 * P1 + s2 * P2
//   l = l1 * P1 + l2 * P2
//
// After R2, applies final normalization: l = l / s

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
#include "ttnn/operations/transformer/sdpa_decode/device/kernels/rt_args_common.hpp"
#include "ttnn/cpp/ttnn/operations/transformer/sdpa_decode/device/kernels/compute/compute_common.hpp"

struct OutputCBs {
    uint32_t l_cb;
    uint32_t s_cb;
    uint32_t m_cb;
};

// Perform SDPA reduction of two (l, s, m) tuples
// Returns CB IDs containing the results (may be temp CBs)
template <uint32_t scale_fp32>
inline OutputCBs sdpa_reduce(
    uint32_t cb_l1,       // l1 input (local)
    uint32_t cb_s1,       // s1 input (local)
    uint32_t cb_m1,       // m1 input (local)
    uint32_t cb_l2,       // l2 input (neighbor)
    uint32_t cb_s2,       // s2 input (neighbor)
    uint32_t cb_m2,       // m2 input (neighbor)
    uint32_t cb_exp_p1,   // temp for exp((m1-m)*scale)
    uint32_t cb_exp_p2,   // temp for exp((m2-m)*scale)
    uint32_t cb_m_temp,   // temp for max result
    uint32_t cb_s1_temp,  // temp for s1
    uint32_t cb_s2_temp,  // temp for s2
    uint32_t cb_l1_temp,  // temp for l1
    uint32_t cb_l2_temp,  // temp for l2
    uint32_t Sq_chunk_t,
    uint32_t vDHt) {
    constexpr int mode = VectorMode::R;
    const uint32_t out_tiles = Sq_chunk_t * vDHt;

    // Wait for all inputs
    cb_wait_front(cb_l1, out_tiles);
    cb_wait_front(cb_l2, out_tiles);
    cb_wait_front(cb_s1, Sq_chunk_t);
    cb_wait_front(cb_s2, Sq_chunk_t);
    cb_wait_front(cb_m1, Sq_chunk_t);
    cb_wait_front(cb_m2, Sq_chunk_t);

    // Move s values to temp CBs (they get modified in-place)
    move_block<false>(cb_s1, cb_s1_temp, Sq_chunk_t);
    move_block<false>(cb_s2, cb_s2_temp, Sq_chunk_t);

    // m = max(m1, m2)
    max_block<mode>(cb_m1, cb_m2, cb_m_temp, Sq_chunk_t);

    // P1 = exp((m1 - m) * scale)
    sub_exp_block<scale_fp32, mode>(cb_m1, cb_m_temp, cb_exp_p1, Sq_chunk_t);

    // P2 = exp((m2 - m) * scale)
    sub_exp_block<scale_fp32, mode>(cb_m2, cb_m_temp, cb_exp_p2, Sq_chunk_t);

    // s1 = s1 * P1
    mul_block_inplace(cb_s1_temp, cb_exp_p1, Sq_chunk_t);

    // s2 = s2 * P2
    mul_block_inplace(cb_s2_temp, cb_exp_p2, Sq_chunk_t);

    // s = s1 * P1 + s2 * P2
    add_block_inplace<true>(cb_s1_temp, cb_s2_temp, Sq_chunk_t);

    // l1 = l1 * P1 (broadcast P1 across columns)
    mul_block_bcast_cols(cb_l1, cb_exp_p1, cb_l1_temp, Sq_chunk_t, vDHt);

    // l2 = l2 * P2 (broadcast P2 across columns)
    mul_block_bcast_cols(cb_l2, cb_exp_p2, cb_l2_temp, Sq_chunk_t, vDHt);

    // l = l1 * P1 + l2 * P2
    add_block_inplace<true>(cb_l1_temp, cb_l2_temp, out_tiles);

    // Pop input CBs
    cb_pop_front(cb_l1, out_tiles);
    cb_pop_front(cb_l2, out_tiles);
    cb_pop_front(cb_s1, Sq_chunk_t);
    cb_pop_front(cb_s2, Sq_chunk_t);
    cb_pop_front(cb_m1, Sq_chunk_t);
    cb_pop_front(cb_m2, Sq_chunk_t);

    // Results are in temp CBs: l=cb_l1_temp, s=cb_s1_temp, m=cb_m_temp
    return {cb_l1_temp, cb_s1_temp, cb_m_temp};
}

namespace NAMESPACE {

// ==========================================================================
// Compile-time args - CB IDs for all data paths
// ==========================================================================

// R1 local input (aliased to input tensor shard)
constexpr uint32_t cb_local_l = get_compile_time_arg_val(0);
constexpr uint32_t cb_local_s = get_compile_time_arg_val(1);
constexpr uint32_t cb_local_m = get_compile_time_arg_val(2);

// R1 neighbor input (aliased to R1 MeshBuffer)
constexpr uint32_t cb_r1_neighbor_l = get_compile_time_arg_val(3);
constexpr uint32_t cb_r1_neighbor_s = get_compile_time_arg_val(4);
constexpr uint32_t cb_r1_neighbor_m = get_compile_time_arg_val(5);

// R1 result / R2 local input (writer sends this to R2 neighbor)
constexpr uint32_t cb_r1_result_l = get_compile_time_arg_val(6);
constexpr uint32_t cb_r1_result_s = get_compile_time_arg_val(7);
constexpr uint32_t cb_r1_result_m = get_compile_time_arg_val(8);

// R2 neighbor input (aliased to R2 MeshBuffer - DIFFERENT sender than R1!)
constexpr uint32_t cb_r2_neighbor_l = get_compile_time_arg_val(9);
constexpr uint32_t cb_r2_neighbor_s = get_compile_time_arg_val(10);
constexpr uint32_t cb_r2_neighbor_m = get_compile_time_arg_val(11);

// Temp CBs for reduction computation
// IMPORTANT: cb_l_out, cb_s_out, cb_m_out are ALIASED to output tensor shards!
// After R2 reduction + normalization, the final result is already at the output address.
constexpr uint32_t cb_exp_p1 = get_compile_time_arg_val(12);   // exp((m1-m)*scale)
constexpr uint32_t cb_exp_p2 = get_compile_time_arg_val(13);   // exp((m2-m)*scale)
constexpr uint32_t cb_m_out = get_compile_time_arg_val(14);    // max result → ALIASED to output_m
constexpr uint32_t cb_s1_temp = get_compile_time_arg_val(15);  // s1 copy (intermediate only)
constexpr uint32_t cb_s2_temp = get_compile_time_arg_val(16);  // s2 copy (intermediate only)
constexpr uint32_t cb_l_out = get_compile_time_arg_val(17);    // l result → ALIASED to output_l
constexpr uint32_t cb_l2_temp = get_compile_time_arg_val(18);  // l2 intermediate
constexpr uint32_t cb_s_out = get_compile_time_arg_val(19);    // s result → ALIASED to output_s

// Compute parameters
constexpr uint32_t scale_fp32 = get_compile_time_arg_val(20);
constexpr uint32_t Sq_chunk_t = get_compile_time_arg_val(21);
constexpr uint32_t vDHt = get_compile_time_arg_val(22);

void MAIN {
    // ==========================================================================
    // Two-round SDPA reduction on the same core
    // ==========================================================================
    //
    // R1: reduce(local_data, r1_neighbor_data) → r1_result
    //     - local_data comes from input tensor shard (CB aliased)
    //     - r1_neighbor_data comes from forward mux neighbor
    //     - r1_result goes to writer for sending to R2 neighbor
    //
    // R2: reduce(r1_result, r2_neighbor_data) → output (in aliased CBs!)
    //     - r1_result from R1 (still in CB, no memcpy)
    //     - r2_neighbor_data comes from backward mux neighbor (DIFFERENT device!)
    //     - Final output lands in cb_l_out/cb_s_out/cb_m_out (aliased to output tensor!)
    //
    // ZERO-COPY OUTPUT:
    // cb_l_out, cb_s_out, cb_m_out are aliased to output tensor shards.
    // After R2 + normalization, the output is ALREADY at the correct address.
    // No extra move operations needed!

    const bool use_half_tile = true;
    constexpr int vector_mode = use_half_tile ? VectorMode::R : VectorMode::RC;
    constexpr uint32_t out_tiles = Sq_chunk_t * vDHt;

    mm_init(cb_local_l, cb_local_l, cb_local_l);

    // =========================================================================
    // ROUND 1: reduce(local, r1_neighbor) → r1_result
    // =========================================================================
    // Note: sdpa_reduce writes results to temp CBs.
    // We then copy to r1_result CBs for the writer to send.

    OutputCBs r1_output = sdpa_reduce<scale_fp32>(
        cb_local_l,
        cb_local_s,
        cb_local_m,  // local (input 1)
        cb_r1_neighbor_l,
        cb_r1_neighbor_s,
        cb_r1_neighbor_m,  // neighbor (input 2)
        cb_exp_p1,
        cb_exp_p2,  // temp for P1, P2
        cb_m_out,
        cb_s1_temp,
        cb_s2_temp,  // temp: m→cb_m_out, s1→cb_s1_temp
        cb_l_out,
        cb_l2_temp,  // temp: l→cb_l_out
        Sq_chunk_t,
        vDHt);

    // sdpa_reduce returns {cb_l_out, cb_s1_temp, cb_m_out}
    // Move R1 results to r1_result CBs (writer needs these for R2 send)
    cb_reserve_back(cb_r1_result_l, out_tiles);
    cb_reserve_back(cb_r1_result_s, Sq_chunk_t);
    cb_reserve_back(cb_r1_result_m, Sq_chunk_t);

    move_block<true>(r1_output.l_cb, cb_r1_result_l, out_tiles);
    move_block<true>(r1_output.s_cb, cb_r1_result_s, Sq_chunk_t);
    move_block<true>(r1_output.m_cb, cb_r1_result_m, Sq_chunk_t);

    cb_push_back(cb_r1_result_l, out_tiles);
    cb_push_back(cb_r1_result_s, Sq_chunk_t);
    cb_push_back(cb_r1_result_m, Sq_chunk_t);

    // =========================================================================
    // ROUND 2: reduce(r1_result, r2_neighbor) → output (in aliased CBs!)
    // =========================================================================
    // The output CBs (cb_l_out, cb_m_out, cb_s_out) are aliased to output tensor shards.
    // After this reduction + normalization, the final values are at their output addresses!

    OutputCBs r2_output = sdpa_reduce<scale_fp32>(
        cb_r1_result_l,
        cb_r1_result_s,
        cb_r1_result_m,  // R1 result = R2 local
        cb_r2_neighbor_l,
        cb_r2_neighbor_s,
        cb_r2_neighbor_m,  // R2 neighbor
        cb_exp_p1,
        cb_exp_p2,  // temp (reuse from R1)
        cb_m_out,
        cb_s1_temp,
        cb_s2_temp,  // m→cb_m_out (ALIASED!), s1→cb_s1_temp
        cb_l_out,
        cb_l2_temp,  // l→cb_l_out (ALIASED!)
        Sq_chunk_t,
        vDHt);

    // r2_output = {cb_l_out, cb_s1_temp, cb_m_out}
    // cb_l_out and cb_m_out are already aliased to output tensor!
    // cb_s1_temp contains final S - need to copy to cb_s_out (which is aliased to output_s)

    // =========================================================================
    // Final normalization: l = l / s
    // =========================================================================
    // First, copy S to output (cb_s_out is aliased to output_s)
    move_block<false>(r2_output.s_cb, cb_s_out, Sq_chunk_t);

    // Compute 1/s in a separate temp (reuse cb_s2_temp since we're done with it)
    move_block<false>(cb_s_out, cb_s2_temp, Sq_chunk_t);
    recip_block_inplace<vector_mode>(cb_s2_temp, Sq_chunk_t);

    // l = l * (1/s) - broadcast 1/s across columns
    // r2_output.l_cb is cb_l_out, which is aliased to output_l
    // After this, output_l has the normalized final result!
    mul_block_bcast_cols_inplace(r2_output.l_cb, cb_s2_temp, Sq_chunk_t, vDHt);

    // =========================================================================
    // Output is now in place!
    // =========================================================================
    // cb_l_out → aliased to output_tensor_l shard (contains normalized L)
    // cb_s_out → aliased to output_tensor_s shard (contains final S)
    // cb_m_out → aliased to output_tensor_m shard (contains final M)
    //
    // No move_block needed! The data is already at the output tensor addresses.
    // Program completion signals to host that output is ready.
}

}  // namespace NAMESPACE
