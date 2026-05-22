// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// DIAGNOSTIC: ONLY does the gate matmul phase 1 and packs to cb_out.
// Skips silu, up, multiply, down phases. Used to verify the matmul logic
// works in isolation.
//
// To make the writer + reader happy, also consumes the phase 2 / phase 4
// CB pushes that the reader still does, and pushes a stub block to
// cb_activated (the writer drains it).

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"

#include "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"

void kernel_main() {
    // Phase 1 (gate) positional CT args.
    constexpr uint32_t g_in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t g_in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t g_in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t g_in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t g_in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t g_in1_block_w = get_compile_time_arg_val(6);
    constexpr uint32_t g_num_blocks = get_compile_time_arg_val(7);

    // Phase 2 (up) - we still drain reader output.
    constexpr uint32_t u_in0_block_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t u_in1_block_num_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t u_num_blocks = get_compile_time_arg_val(15);

    // Phase 4 (down) - drain reader output.
    constexpr uint32_t d_in0_block_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t d_in1_block_num_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t d_num_blocks = get_compile_time_arg_val(23);

    constexpr uint32_t gu_out_subblock_h = get_compile_time_arg_val(24);
    constexpr uint32_t gu_out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t gu_out_subblock_num_tiles = gu_out_subblock_h * gu_out_subblock_w;
    constexpr uint32_t gu_out_block_num_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(27);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(28);
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_out_block_num_tiles = get_compile_time_arg_val(29);

    constexpr uint32_t cb_in0_x = get_named_compile_time_arg_val("cb_in0_x");
    constexpr uint32_t cb_in1_gate = get_named_compile_time_arg_val("cb_in1_gate");
    constexpr uint32_t cb_in1_up = get_named_compile_time_arg_val("cb_in1_up");
    constexpr uint32_t cb_in1_down = get_named_compile_time_arg_val("cb_in1_down");
    constexpr uint32_t cb_activated = get_named_compile_time_arg_val("cb_activated");
    constexpr uint32_t cb_in0_down_full = get_named_compile_time_arg_val("cb_in0_down_full");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    // ----- Phase 1 (gate matmul without silu): pack each subblock to cb_out
    // directly as a sanity check that matmul works.
    constexpr uint32_t g_in0_subblock_num_tiles = gu_out_subblock_h * g_in0_block_w;
    constexpr uint32_t in0_subblock_offset = g_in0_subblock_num_tiles;
    constexpr uint32_t in1_subblock_offset = gu_out_subblock_w;

    mm_block_init(cb_in0_x, cb_in1_gate, cb_out, /*transpose=*/0, gu_out_subblock_w, gu_out_subblock_h, g_in0_block_w);

    for (uint32_t kb = 0; kb < g_num_blocks; ++kb) {
        cb_wait_front(cb_in0_x, g_in0_block_num_tiles);
        cb_wait_front(cb_in1_gate, g_in1_block_num_tiles);

        uint32_t in0_off = 0;
        for (uint32_t sm = 0; sm < g_in0_num_subblocks; ++sm) {
            uint32_t in1_off = 0;
            for (uint32_t sn = 0; sn < g_in1_num_subblocks; ++sn) {
                tile_regs_acquire();
                uint32_t a_idx = in0_off, b_idx = in1_off;
                for (uint32_t k = 0; k < g_in0_block_w; ++k) {
                    matmul_block(
                        cb_in0_x,
                        cb_in1_gate,
                        a_idx,
                        b_idx,
                        /*dst_index=*/0,
                        /*transpose=*/false,
                        gu_out_subblock_w,
                        gu_out_subblock_h,
                        g_in0_block_w);
                    a_idx += 1;
                    b_idx += g_in1_block_w;
                }
                tile_regs_commit();
                // tile_regs MUST be waited+released on PACK side every iteration,
                // even when we don't actually pack — otherwise MATH's next
                // tile_regs_acquire will block waiting for PACK to consume dst.
                tile_regs_wait();
                if (kb == g_num_blocks - 1) {
                    // Last K-block: pack to cb_out (one-time, doesn't accumulate
                    // across K — so the result is just the last K-block's
                    // contribution, but it confirms matmul_block is working).
                    cb_reserve_back(cb_out, gu_out_subblock_num_tiles);
                    pack_tile_block(0, cb_out, gu_out_subblock_num_tiles);
                    cb_push_back(cb_out, gu_out_subblock_num_tiles);
                }
                tile_regs_release();
                in1_off += gu_out_subblock_w;
            }
            in0_off += in0_subblock_offset;
        }

        cb_pop_front(cb_in0_x, g_in0_block_num_tiles);
        cb_pop_front(cb_in1_gate, g_in1_block_num_tiles);
    }

    // ----- Phase 2: drain reader without computing (we don't need up).
    for (uint32_t kb = 0; kb < u_num_blocks; ++kb) {
        cb_wait_front(cb_in0_x, u_in0_block_num_tiles);
        cb_wait_front(cb_in1_up, u_in1_block_num_tiles);
        cb_pop_front(cb_in0_x, u_in0_block_num_tiles);
        cb_pop_front(cb_in1_up, u_in1_block_num_tiles);
    }

    // ----- Phase 3: push a stub block to cb_activated so writer drains.
    for (uint32_t sb = 0; sb < gu_out_block_num_tiles / gu_out_subblock_num_tiles; ++sb) {
        cb_reserve_back(cb_activated, gu_out_subblock_num_tiles);
        cb_push_back(cb_activated, gu_out_subblock_num_tiles);
    }

    // ----- Phase 4: drain reader without computing.
    for (uint32_t kb = 0; kb < d_num_blocks; ++kb) {
        cb_wait_front(cb_in0_down_full, d_in0_block_num_tiles);
        cb_wait_front(cb_in1_down, d_in1_block_num_tiles);
        cb_pop_front(cb_in0_down_full, d_in0_block_num_tiles);
        cb_pop_front(cb_in1_down, d_in1_block_num_tiles);
    }

    // Also push enough subblocks to cb_out so writer's expected count drains.
    // We already pushed g_in0_num_subblocks * g_in1_num_subblocks subblocks
    // of size gu_out_subblock_num_tiles in phase 1 (but those are gate-matmul
    // size, not the down's writer-expected size). The writer drains
    // d_out_block_num_tiles total, in d_out_subblock_num_tiles chunks.
    constexpr uint32_t d_subblocks_total = d_out_block_num_tiles / d_out_subblock_num_tiles;
    constexpr uint32_t gate_subblocks_pushed = (g_in0_num_subblocks * g_in1_num_subblocks);
    // Each gate subblock was gu_out_subblock_num_tiles tiles, total
    // = gate_subblocks_pushed * gu_out_subblock_num_tiles. Writer drains in
    // d_out_subblock_num_tiles chunks. So writer is ALREADY draining the gate
    // output (different subblock size but same total tile count if shapes
    // happen to match). For v1 sanity, just push enough zero subblocks of
    // the down's subblock size to make up any deficit.
    constexpr uint32_t pushed_total_tiles = gate_subblocks_pushed * gu_out_subblock_num_tiles;
    constexpr uint32_t expected_total_tiles = d_subblocks_total * d_out_subblock_num_tiles;
    if constexpr (pushed_total_tiles < expected_total_tiles) {
        constexpr uint32_t needed_extra = expected_total_tiles - pushed_total_tiles;
        constexpr uint32_t extra_subblocks = needed_extra / d_out_subblock_num_tiles;
        for (uint32_t i = 0; i < extra_subblocks; ++i) {
            cb_reserve_back(cb_out, d_out_subblock_num_tiles);
            cb_push_back(cb_out, d_out_subblock_num_tiles);
        }
    }
}
