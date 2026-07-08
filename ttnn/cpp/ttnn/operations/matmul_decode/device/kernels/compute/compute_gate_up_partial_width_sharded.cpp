// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "partial_phases.hpp"

using std::uint32_t;

// Fused gate+up+GeGLU partial-width-sharded matmul compute.
//
// Runs the partial matmul (phase 1) and the base-core K-reduction (phase 2) TWICE over the SINGLE
// gathered A -- once for gate_b (in1, fused gelu) and once for up_b (in1b, no activation) -- then
// phase 3 multiplies the two fully-reduced results into the single GeGLU output
// hid = gelu(A @ gate_w) * (A @ up_w), folding the downstream eltwise multiply into the op.
//
//   full_in0_cb (c_3): gathered A (published by reader, shared by both phase-1 matmuls)
//   in1_cb  (c_1): gate_b block -> gate_partial (c_4) -> gate_reduce (c_5) -> gate_out (c_2, gelu)
//   in1b_cb (c_6): up_b   block -> up_partial   (c_7) -> up_reduce   (c_8) -> up_out   (c_9)
//   -> hid_out (c_10): gate_out * up_out (the single device output shard)
//
// full_in0 is consumed (cb_pop_front) ONCE after BOTH phase-1 matmuls read it.

using namespace ckernel;
void kernel_main() {
    constexpr uint32_t M_tiles = get_compile_time_arg_val(0);
    constexpr uint32_t K_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t Kc_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t Nc_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t K_blocks = get_compile_time_arg_val(4);
    constexpr uint32_t inA_K_tiles_per_core = get_compile_time_arg_val(5);
    constexpr uint32_t fused_gelu_approx = get_compile_time_arg_val(6);

    const uint32_t k_idx = get_arg_val<uint32_t>(0);
    const uint32_t is_base = get_arg_val<uint32_t>(1);

    constexpr uint32_t full_in0_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t in1_cb_id = tt::CBIndex::c_1;  // gate_b
    constexpr uint32_t gate_out_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t gate_partial_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t gate_reduce_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t in1b_cb_id = tt::CBIndex::c_6;  // up_b
    constexpr uint32_t up_partial_cb_id = tt::CBIndex::c_7;
    constexpr uint32_t up_reduce_cb_id = tt::CBIndex::c_8;
    constexpr uint32_t up_out_cb_id = tt::CBIndex::c_9;
    constexpr uint32_t hid_out_cb_id = tt::CBIndex::c_10;  // gate_out * up_out (single output)

    constexpr uint32_t full_in0_num_tiles = M_tiles * K_tiles;
    constexpr uint32_t in1_num_tiles = Kc_tiles * Nc_tiles;
    constexpr uint32_t block_num_tiles = M_tiles * Nc_tiles;
    constexpr uint32_t sender_slice_tiles = M_tiles * inA_K_tiles_per_core;

    constexpr uint32_t out_block_h = M_tiles;
    constexpr uint32_t out_block_w = 1;
    constexpr uint32_t in0_block_w = inA_K_tiles_per_core;

    const uint32_t k_offset = k_idx * Kc_tiles;

    // ---- Phase 1: partial matmul for BOTH weights over the single gathered A ----
    cb_wait_front(full_in0_cb_id, full_in0_num_tiles);
    cb_wait_front(in1_cb_id, in1_num_tiles);
    cb_wait_front(in1b_cb_id, in1_num_tiles);

    phase1_partial<
        M_tiles,
        Kc_tiles,
        Nc_tiles,
        inA_K_tiles_per_core,
        out_block_w,
        out_block_h,
        in0_block_w,
        block_num_tiles,
        sender_slice_tiles>(full_in0_cb_id, in1_cb_id, gate_partial_cb_id, k_offset);
    phase1_partial<
        M_tiles,
        Kc_tiles,
        Nc_tiles,
        inA_K_tiles_per_core,
        out_block_w,
        out_block_h,
        in0_block_w,
        block_num_tiles,
        sender_slice_tiles>(full_in0_cb_id, in1b_cb_id, up_partial_cb_id, k_offset);

    // A is consumed once, after BOTH partial matmuls have read it.
    cb_pop_front(full_in0_cb_id, full_in0_num_tiles);

    if (is_base == 0) {
        return;
    }
    // ---- Phase 2: reduce K_blocks partials for gate (gelu) and up (no activation) ----
    phase2_reduce<M_tiles, Nc_tiles, K_blocks, block_num_tiles, /*do_gelu=*/true, (bool)fused_gelu_approx>(
        gate_reduce_cb_id, gate_out_cb_id);
    phase2_reduce<M_tiles, Nc_tiles, K_blocks, block_num_tiles, /*do_gelu=*/false, false>(
        up_reduce_cb_id, up_out_cb_id);

    // ---- Phase 3: GeGLU multiply gate_out * up_out -> hid_out (single device output) ----
    phase3_multiply<block_num_tiles>(gate_out_cb_id, up_out_cb_id, hid_out_cb_id);
}
