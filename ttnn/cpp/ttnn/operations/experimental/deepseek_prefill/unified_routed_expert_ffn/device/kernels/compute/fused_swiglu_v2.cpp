// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// v2 fused SwiGLU compute kernel — based on the simpler bmm_large_block_zm.cpp
// pattern instead of the more complex bmm_large_block_zm_fused_bias_activation.
//
// Key insight: matmul_tiles ADDS to dst[dst_index] (accumulates). To
// accumulate across K-blocks, we spill each block's dst to intermed CB and
// reload on the NEXT block via copy_tile (which loads intermed into dst).
// The new matmul_tiles then accumulates on top.
//
// Pattern per phase:
//   for each K-block:
//     for each (sub_m, sub_n) subblock:
//       acquire_dst
//       if not first block: copy from intermed_cb to dst (loads prior partial)
//       for each output tile (h, w) of the subblock:
//         for each K iteration: matmul_tiles accumulating to dst[h*w + w]
//       if last K-block: pack to final_cb
//       else: pack to intermed_cb (overwriting prior partial with new sum)
//       release_dst
//     pop in0 + in1 K-block

#include <cstdint>

#include "api/compute/matmul.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "api/debug/dprint.h"

#include "ttnn/cpp/ttnn/operations/matmul/device/kernels/compute/bmm_fused_activation.hpp"

template <
    uint32_t in0_block_w,
    uint32_t in0_num_subblocks,
    uint32_t in0_block_num_tiles,
    uint32_t in0_subblock_num_tiles,
    uint32_t in1_num_subblocks,
    uint32_t in1_block_num_tiles,
    uint32_t in1_per_core_w,
    uint32_t num_blocks,
    uint32_t out_subblock_h,
    uint32_t out_subblock_w,
    uint32_t out_subblock_num_tiles,
    bool apply_silu_on_final>
FORCE_INLINE void matmul_phase_v2(
    uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t intermed_cb_id, uint32_t final_cb_id) {
    constexpr bool spill = num_blocks > 1;
    bool enable_reload = false;

    for (uint32_t block = 0; block < num_blocks; ++block) {
        const bool last_out = (block == num_blocks - 1);

        DPRINT << "MM: block " << block << " wait in0 " << in0_block_num_tiles << ENDL();
        cb_wait_front(in0_cb_id, in0_block_num_tiles);
        DPRINT << "MM: block " << block << " wait in1 " << in1_block_num_tiles << ENDL();
        cb_wait_front(in1_cb_id, in1_block_num_tiles);
        DPRINT << "MM: block " << block << " inner loops" << ENDL();

        int in0_index_subblock_offset = 0;
        for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
            int in1_index_subblock_offset = 0;
            for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
                DPRINT << "MM: sb(" << in0_subblock << "," << in1_subblock << ") acquire reload=" << (uint32_t)enable_reload << ENDL();
                acquire_dst();
                DPRINT << "MM: sb after acquire" << ENDL();

                if (enable_reload) {
                    copy_tile_to_dst_init_short_with_dt(in1_cb_id, intermed_cb_id);
                    cb_wait_front(intermed_cb_id, out_subblock_num_tiles);
                    for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                        copy_tile(intermed_cb_id, i, i);
                    }
                    cb_pop_front(intermed_cb_id, out_subblock_num_tiles);
                    mm_init_short_with_dt(in0_cb_id, in1_cb_id, intermed_cb_id);
                }

                DPRINT << "MM: sb compute start" << ENDL();
                // Compute output sub-block via per-tile inner-product accumulation.
                int dst_index = 0;
                int in0_index_h_offset = 0;
                for (uint32_t h = 0; h < out_subblock_h; ++h) {
                    for (uint32_t w = 0; w < out_subblock_w; ++w) {
                        int in1_index_inner_dim_offset = 0;
                        for (uint32_t inner_dim = 0; inner_dim < in0_block_w; ++inner_dim) {
                            int in0_index = in0_index_subblock_offset + in0_index_h_offset + inner_dim;
                            int in1_index = in1_index_subblock_offset + in1_index_inner_dim_offset + w;
                            matmul_tiles(in0_cb_id, in1_cb_id, in0_index, in1_index, dst_index);
                            in1_index_inner_dim_offset += in1_per_core_w;
                        }
                        dst_index++;
                    }
                    in0_index_h_offset += in0_block_w;
                }
                DPRINT << "MM: sb compute done" << ENDL();

                if (last_out) {
                    DPRINT << "MM: pack last_out reserve" << ENDL();
                    if constexpr (apply_silu_on_final) {
                        apply_activation_from_pack<KernelActivation::SILU>(out_subblock_num_tiles);
                    }
                    cb_reserve_back(final_cb_id, out_subblock_num_tiles);
                    DPRINT << "MM: pack last_out reserved" << ENDL();
                    for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                        pack_tile(i, final_cb_id);
                    }
                    cb_push_back(final_cb_id, out_subblock_num_tiles);
                    DPRINT << "MM: pack last_out pushed" << ENDL();
                } else {
                    DPRINT << "MM: pack intermed reserve" << ENDL();
                    cb_reserve_back(intermed_cb_id, out_subblock_num_tiles);
                    DPRINT << "MM: pack intermed reserved" << ENDL();
                    for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
                        pack_tile(i, intermed_cb_id);
                    }
                    cb_push_back(intermed_cb_id, out_subblock_num_tiles);
                    DPRINT << "MM: pack intermed pushed" << ENDL();
                }

                DPRINT << "MM: release_dst" << ENDL();
                release_dst();
                in1_index_subblock_offset += out_subblock_w;
            }
            in0_index_subblock_offset += in0_subblock_num_tiles;
        }

        if constexpr (spill) {
            enable_reload = true;
        }

        cb_pop_front(in0_cb_id, in0_block_num_tiles);
        cb_pop_front(in1_cb_id, in1_block_num_tiles);
    }
}

template <uint32_t out_block_num_tiles, uint32_t out_subblock_num_tiles>
FORCE_INLINE void multiply_phase_v2(uint32_t gate_cb_id, uint32_t up_cb_id, uint32_t activated_cb_id) {
    cb_wait_front(gate_cb_id, out_block_num_tiles);
    cb_wait_front(up_cb_id, out_block_num_tiles);

    mul_tiles_init(gate_cb_id, up_cb_id);

    constexpr uint32_t num_subblocks = out_block_num_tiles / out_subblock_num_tiles;
    uint32_t base = 0;
    for (uint32_t sb = 0; sb < num_subblocks; ++sb) {
        acquire_dst();
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            mul_tiles(gate_cb_id, up_cb_id, base + i, base + i, i);
        }
        cb_reserve_back(activated_cb_id, out_subblock_num_tiles);
        for (uint32_t i = 0; i < out_subblock_num_tiles; ++i) {
            pack_tile(i, activated_cb_id);
        }
        cb_push_back(activated_cb_id, out_subblock_num_tiles);
        release_dst();
        base += out_subblock_num_tiles;
    }
    cb_pop_front(gate_cb_id, out_block_num_tiles);
    cb_pop_front(up_cb_id, out_block_num_tiles);
}

void kernel_main() {
    // Phase 1 (gate)
    constexpr uint32_t g_in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t g_in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t g_in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t g_in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t g_in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t g_in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t g_in1_per_core_w = get_compile_time_arg_val(6);
    constexpr uint32_t g_num_blocks = get_compile_time_arg_val(7);
    // Phase 2 (up)
    constexpr uint32_t u_in0_block_w = get_compile_time_arg_val(8);
    constexpr uint32_t u_in0_num_subblocks = get_compile_time_arg_val(9);
    constexpr uint32_t u_in0_block_num_tiles = get_compile_time_arg_val(10);
    constexpr uint32_t u_in0_subblock_num_tiles = get_compile_time_arg_val(11);
    constexpr uint32_t u_in1_num_subblocks = get_compile_time_arg_val(12);
    constexpr uint32_t u_in1_block_num_tiles = get_compile_time_arg_val(13);
    constexpr uint32_t u_in1_per_core_w = get_compile_time_arg_val(14);
    constexpr uint32_t u_num_blocks = get_compile_time_arg_val(15);
    // Phase 4 (down)
    constexpr uint32_t d_in0_block_w = get_compile_time_arg_val(16);
    constexpr uint32_t d_in0_num_subblocks = get_compile_time_arg_val(17);
    constexpr uint32_t d_in0_block_num_tiles = get_compile_time_arg_val(18);
    constexpr uint32_t d_in0_subblock_num_tiles = get_compile_time_arg_val(19);
    constexpr uint32_t d_in1_num_subblocks = get_compile_time_arg_val(20);
    constexpr uint32_t d_in1_block_num_tiles = get_compile_time_arg_val(21);
    constexpr uint32_t d_in1_per_core_w = get_compile_time_arg_val(22);
    constexpr uint32_t d_num_blocks = get_compile_time_arg_val(23);
    // Subblock dims
    constexpr uint32_t gu_out_subblock_h = get_compile_time_arg_val(24);
    constexpr uint32_t gu_out_subblock_w = get_compile_time_arg_val(25);
    constexpr uint32_t gu_out_subblock_num_tiles = gu_out_subblock_h * gu_out_subblock_w;
    constexpr uint32_t gu_out_block_num_tiles = get_compile_time_arg_val(26);
    constexpr uint32_t d_out_subblock_h = get_compile_time_arg_val(27);
    constexpr uint32_t d_out_subblock_w = get_compile_time_arg_val(28);
    constexpr uint32_t d_out_subblock_num_tiles = d_out_subblock_h * d_out_subblock_w;
    constexpr uint32_t d_out_block_num_tiles = get_compile_time_arg_val(29);

    // CBs
    constexpr uint32_t cb_in0_x = get_named_compile_time_arg_val("cb_in0_x");
    constexpr uint32_t cb_in1_gate = get_named_compile_time_arg_val("cb_in1_gate");
    constexpr uint32_t cb_in1_up = get_named_compile_time_arg_val("cb_in1_up");
    constexpr uint32_t cb_in1_down = get_named_compile_time_arg_val("cb_in1_down");
    constexpr uint32_t cb_gate_intermed = get_named_compile_time_arg_val("cb_gate_intermed");
    constexpr uint32_t cb_up_intermed = get_named_compile_time_arg_val("cb_up_intermed");
    constexpr uint32_t cb_activated = get_named_compile_time_arg_val("cb_activated");
    constexpr uint32_t cb_in0_down_full = get_named_compile_time_arg_val("cb_in0_down_full");
    constexpr uint32_t cb_partials_gu = get_named_compile_time_arg_val("cb_mm_partials_gu");
    constexpr uint32_t cb_partials_d = get_named_compile_time_arg_val("cb_mm_partials_d");
    constexpr uint32_t cb_out = get_named_compile_time_arg_val("cb_out");

    DPRINT << "FSW: entering, init silu/mm" << ENDL();
    silu_tile_init_pack();
    mm_init(cb_in0_x, cb_in1_gate, cb_gate_intermed);
    DPRINT << "FSW: phase1 start" << ENDL();

    // Phase 1: gate matmul (silu fused on final pack)
    matmul_phase_v2<
        g_in0_block_w,
        g_in0_num_subblocks,
        g_in0_block_num_tiles,
        g_in0_subblock_num_tiles,
        g_in1_num_subblocks,
        g_in1_block_num_tiles,
        g_in1_per_core_w,
        g_num_blocks,
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_subblock_num_tiles,
        /*apply_silu_on_final=*/true>(cb_in0_x, cb_in1_gate, cb_partials_gu, cb_gate_intermed);

    // Phase 2: up matmul
    mm_init_short_with_dt(cb_in0_x, cb_in1_up, cb_in1_gate);
    matmul_phase_v2<
        u_in0_block_w,
        u_in0_num_subblocks,
        u_in0_block_num_tiles,
        u_in0_subblock_num_tiles,
        u_in1_num_subblocks,
        u_in1_block_num_tiles,
        u_in1_per_core_w,
        u_num_blocks,
        gu_out_subblock_h,
        gu_out_subblock_w,
        gu_out_subblock_num_tiles,
        /*apply_silu_on_final=*/false>(cb_in0_x, cb_in1_up, cb_partials_gu, cb_up_intermed);

    // Phase 3: elementwise multiply
    multiply_phase_v2<gu_out_block_num_tiles, gu_out_subblock_num_tiles>(
        cb_gate_intermed, cb_up_intermed, cb_activated);

    // Phase 4: down matmul
    mm_init(cb_in0_down_full, cb_in1_down, cb_out);
    matmul_phase_v2<
        d_in0_block_w,
        d_in0_num_subblocks,
        d_in0_block_num_tiles,
        d_in0_subblock_num_tiles,
        d_in1_num_subblocks,
        d_in1_block_num_tiles,
        d_in1_per_core_w,
        d_num_blocks,
        d_out_subblock_h,
        d_out_subblock_w,
        d_out_subblock_num_tiles,
        /*apply_silu_on_final=*/false>(cb_in0_down_full, cb_in1_down, cb_partials_d, cb_out);
}
