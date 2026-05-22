// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Diagnostic stub for the fused SwiGLU compute kernel.
//
// This kernel just consumes everything the reader pushes and produces
// `d_out_block_num_tiles` zero tiles for each iteration of cb_out. It's
// designed to validate the CB protocol / runtime-arg wiring without
// running the actual matmul logic. If THIS kernel runs to completion, the
// reader/writer/factory are correctly wired and the hang in the real
// kernel is purely in the matmul/silu/multiply compute logic.

#include <cstdint>

#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

void kernel_main() {
    // Phase shape positional args (must match fused_swiglu.cpp's layout
    // because the same factory feeds both).
    constexpr uint32_t g_in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t g_in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t g_num_blocks = get_compile_time_arg_val(7);

    constexpr uint32_t u_in0_block_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t u_in1_block_num_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t u_num_blocks = get_compile_time_arg_val(15);

    constexpr uint32_t d_in0_block_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t d_in1_block_num_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t d_num_blocks = get_compile_time_arg_val(23);

    constexpr uint32_t gu_out_block_num_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t d_out_subblock_num_tiles = get_compile_time_arg_val(27) * get_compile_time_arg_val(28);
    constexpr uint32_t d_out_block_num_tiles = get_compile_time_arg_val(29);

    constexpr uint32_t cb_in0_x = get_named_compile_time_arg_val("cb_in0_x");
    constexpr uint32_t cb_in1_gate = get_named_compile_time_arg_val("cb_in1_gate");
    constexpr uint32_t cb_in1_up = get_named_compile_time_arg_val("cb_in1_up");
    constexpr uint32_t cb_in1_down = get_named_compile_time_arg_val("cb_in1_down");
    constexpr uint32_t cb_gate_intermed = get_named_compile_time_arg_val("cb_gate_intermed");
    constexpr uint32_t cb_up_intermed = get_named_compile_time_arg_val("cb_up_intermed");
    constexpr uint32_t cb_activated = get_named_compile_time_arg_val("cb_activated");
    constexpr uint32_t cb_in0_down_full = get_named_compile_time_arg_val("cb_in0_down_full");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    CircularBuffer in0_cb(cb_in0_x);
    CircularBuffer gate_w_cb(cb_in1_gate);
    CircularBuffer up_w_cb(cb_in1_up);
    CircularBuffer down_w_cb(cb_in1_down);
    CircularBuffer gate_int_cb(cb_gate_intermed);
    CircularBuffer up_int_cb(cb_up_intermed);
    CircularBuffer activated_cb(cb_activated);
    CircularBuffer in0_down_full_cb(cb_in0_down_full);
    CircularBuffer out_cb(cb_out);

    // ----- Phase 1: drain in0_x + in1_gate, push zeros to gate_intermed ----
    for (uint32_t kb = 0; kb < g_num_blocks; ++kb) {
        in0_cb.wait_front(g_in0_block_num_tiles);
        gate_w_cb.wait_front(g_in1_block_num_tiles);
        in0_cb.pop_front(g_in0_block_num_tiles);
        gate_w_cb.pop_front(g_in1_block_num_tiles);
    }
    // Push one full block of zeros to gate_intermed.
    gate_int_cb.reserve_back(gu_out_block_num_tiles);
    gate_int_cb.push_back(gu_out_block_num_tiles);

    // ----- Phase 2: drain in0_x + in1_up, push zeros to up_intermed --------
    for (uint32_t kb = 0; kb < u_num_blocks; ++kb) {
        in0_cb.wait_front(u_in0_block_num_tiles);
        up_w_cb.wait_front(u_in1_block_num_tiles);
        in0_cb.pop_front(u_in0_block_num_tiles);
        up_w_cb.pop_front(u_in1_block_num_tiles);
    }
    up_int_cb.reserve_back(gu_out_block_num_tiles);
    up_int_cb.push_back(gu_out_block_num_tiles);

    // ----- Phase 3: pop gate_intermed + up_intermed, push activated --------
    gate_int_cb.wait_front(gu_out_block_num_tiles);
    up_int_cb.wait_front(gu_out_block_num_tiles);
    gate_int_cb.pop_front(gu_out_block_num_tiles);
    up_int_cb.pop_front(gu_out_block_num_tiles);
    activated_cb.reserve_back(gu_out_block_num_tiles);
    activated_cb.push_back(gu_out_block_num_tiles);

    // ----- Phase 4: drain in0_down_full + in1_down, push d_out_block to out
    for (uint32_t kb = 0; kb < d_num_blocks; ++kb) {
        in0_down_full_cb.wait_front(d_in0_block_num_tiles);
        down_w_cb.wait_front(d_in1_block_num_tiles);
        in0_down_full_cb.pop_front(d_in0_block_num_tiles);
        down_w_cb.pop_front(d_in1_block_num_tiles);
    }
    // Push d_out_block_num_tiles tiles to cb_out subblock-by-subblock (so the
    // writer's wait_front pattern matches what the real kernel does).
    static_assert(
        d_out_block_num_tiles % d_out_subblock_num_tiles == 0,
        "d_out_block_num_tiles must be a multiple of d_out_subblock_num_tiles");
    constexpr uint32_t d_num_subblocks_out = d_out_block_num_tiles / d_out_subblock_num_tiles;
    for (uint32_t sb = 0; sb < d_num_subblocks_out; ++sb) {
        out_cb.reserve_back(d_out_subblock_num_tiles);
        out_cb.push_back(d_out_subblock_num_tiles);
    }
}
