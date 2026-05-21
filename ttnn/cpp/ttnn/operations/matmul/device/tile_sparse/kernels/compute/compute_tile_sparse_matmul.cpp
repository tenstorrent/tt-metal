// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
//
// Tile-sparse matmul compute kernel with K-block skip.
//
// For each K-block, checks the runtime k_active_mask bitmask:
//   - Active  (bit=1): wait for A/B tiles from CBs, run matmul, accumulate.
//   - Inactive(bit=0): skip entirely — no CB wait/pop, no compute.
//
// Accumulation uses a software spill/reload through mm_partials_cb_id (cb_intermed0).
// Per-subblock pattern:
//   - First active block:  compute A*B → pack to mm_partials (or output if only 1 active)
//   - Middle active blocks: reload from mm_partials → compute and add → pack to mm_partials
//   - Last active block:   reload from mm_partials → compute and add → pack to output
//
// No PACKER_L1_ACC, no bias, no activation, no batch, no transpose.
//
// Compile-time args (same indices as bmm_large_block_zm_fused_bias_activation):
//   0  in0_block_w
//   1  in0_num_subblocks
//   2  in0_block_num_tiles
//   3  in0_subblock_num_tiles
//   4  in1_num_subblocks
//   5  in1_block_num_tiles
//   6  in1_per_core_w  (= out_subblock_w * in1_num_subblocks)
//   7  num_blocks_inner_dim  (= num_k_blocks)
//   8  out_num_blocks_x      (must be 1 for sparse path)
//   9  out_num_blocks_y
//  10  out_subblock_h
//  11  out_subblock_w
//  12  out_subblock_num_tiles
//  13  batch                 (must be 1 for sparse path)
//  14  out_block_num_tiles
//  15  untilize_out          (must be false)
//  16  get_batch_from_reader (must be false)
//  17  in0_transpose_tile    (must be false)
//
// Named compile-time args: cb_in0, cb_in1, cb_out, cb_intermed0
//
// Runtime args:
//   0  k_active_mask  -- bit k = 1 → K-block k should be processed

#include <cstdint>
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"

FORCE_INLINE void sparse_reload_from_partials(
    uint32_t in0_cb_id,
    uint32_t in1_cb_id,
    uint32_t mm_partials_cb_id,
    uint32_t out_subblock_num_tiles,
    uint32_t out_subblock_w,
    uint32_t out_subblock_h,
    uint32_t in0_block_w) {
    copy_tile_to_dst_init_short_with_dt(in1_cb_id, mm_partials_cb_id);
    cb_wait_front(mm_partials_cb_id, out_subblock_num_tiles);
    uint32_t start_tile_index = 0;
    uint32_t start_dst_index = 0;
    copy_block_matmul_partials(mm_partials_cb_id, start_tile_index, start_dst_index, out_subblock_num_tiles);
    cb_pop_front(mm_partials_cb_id, out_subblock_num_tiles);
    mm_block_init_short_with_dt(
        in0_cb_id, in1_cb_id, mm_partials_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);
}

void kernel_main() {
    // ---- Runtime args ----
    const uint32_t k_active_mask = get_arg_val<uint32_t>(0);

    // ---- Compile-time args ----
    constexpr uint32_t in0_block_w = get_compile_time_arg_val(0);
    constexpr uint32_t in0_num_subblocks = get_compile_time_arg_val(1);
    constexpr uint32_t in0_block_num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_subblock_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t in1_num_subblocks = get_compile_time_arg_val(4);
    constexpr uint32_t in1_block_num_tiles = get_compile_time_arg_val(5);
    constexpr uint32_t in1_block_w = get_compile_time_arg_val(6);  // in1_per_core_w
    constexpr uint32_t num_blocks = get_compile_time_arg_val(7);
    constexpr uint32_t out_num_blocks_x = get_compile_time_arg_val(8);
    constexpr uint32_t out_num_blocks_y = get_compile_time_arg_val(9);
    constexpr uint32_t out_subblock_h = get_compile_time_arg_val(10);
    constexpr uint32_t out_subblock_w = get_compile_time_arg_val(11);
    constexpr uint32_t out_subblock_num_tiles = get_compile_time_arg_val(12);
    // [13] batch, [14] out_block_num_tiles, [15-17] unused flags

    constexpr uint32_t in0_cb_id = get_named_compile_time_arg_val("cb_in0");
    constexpr uint32_t in1_cb_id = get_named_compile_time_arg_val("cb_in1");
    constexpr uint32_t out_cb_id = get_named_compile_time_arg_val("cb_out");
    constexpr uint32_t mm_partials_cb_id = get_named_compile_time_arg_val("cb_intermed0");

    // ---- Precompute last active block ----
    uint32_t last_active = 0;
    for (uint32_t b = 0; b < num_blocks; ++b) {
        if ((k_active_mask >> b) & 1u) {
            last_active = b;
        }
    }

    mm_block_init(in0_cb_id, in1_cb_id, mm_partials_cb_id, false, out_subblock_w, out_subblock_h, in0_block_w);

    for (uint32_t bh = 0; bh < out_num_blocks_y; ++bh) {
        for (uint32_t bw = 0; bw < out_num_blocks_x; ++bw) {
            uint32_t active_count = 0;

            for (uint32_t block = 0; block < num_blocks; ++block) {
                if (!((k_active_mask >> block) & 1u)) {
                    continue;  // Skip inactive K-block
                }

                bool is_first = (active_count == 0);
                bool is_last = (block == last_active);

                cb_wait_front(in0_cb_id, in0_block_num_tiles);
                cb_wait_front(in1_cb_id, in1_block_num_tiles);

                uint32_t in0_index_subblock_offset = 0;
                for (uint32_t in0_subblock = 0; in0_subblock < in0_num_subblocks; ++in0_subblock) {
                    uint32_t in1_index_subblock_offset = 0;
                    for (uint32_t in1_subblock = 0; in1_subblock < in1_num_subblocks; ++in1_subblock) {
                        tile_regs_acquire();

                        if (!is_first) {
                            // Load previous partial from mm_partials into DST registers
                            sparse_reload_from_partials(
                                in0_cb_id,
                                in1_cb_id,
                                mm_partials_cb_id,
                                out_subblock_num_tiles,
                                out_subblock_w,
                                out_subblock_h,
                                in0_block_w);
                        }

                        // Compute A[subblock] * B[subblock] and accumulate into DST
                        uint32_t dst_index = 0;
                        uint32_t in0_index = in0_index_subblock_offset;
                        uint32_t in1_index = in1_index_subblock_offset;
                        for (uint32_t inner = 0; inner < in0_block_w; ++inner) {
                            matmul_block(
                                in0_cb_id,
                                in1_cb_id,
                                in0_index,
                                in1_index,
                                dst_index,
                                false,  // in1_transpose
                                out_subblock_w,
                                out_subblock_h,
                                in0_block_w);
                            ++in0_index;
                            in1_index += in1_block_w;  // stride to next K-row of B block
                        }

                        if (is_last) {
                            // Pack result to output CB (consumed by writer)
                            tile_regs_commit();
                            cb_reserve_back(out_cb_id, out_subblock_num_tiles);
                            tile_regs_wait();
                            pack_tile_block(0, out_cb_id, out_subblock_num_tiles);
                            tile_regs_release();
                            cb_push_back(out_cb_id, out_subblock_num_tiles);
                        } else {
                            // Pack partial result to intermed CB (reload in next active block)
                            tile_regs_commit();
                            cb_reserve_back(mm_partials_cb_id, out_subblock_num_tiles);
                            tile_regs_wait();
                            pack_tile_block(0, mm_partials_cb_id, out_subblock_num_tiles);
                            tile_regs_release();
                            cb_push_back(mm_partials_cb_id, out_subblock_num_tiles);
                        }

                        in1_index_subblock_offset += out_subblock_w;
                    }
                    in0_index_subblock_offset += in0_subblock_num_tiles;
                }

                cb_pop_front(in0_cb_id, in0_block_num_tiles);
                cb_pop_front(in1_cb_id, in1_block_num_tiles);
                ++active_count;
            }
        }
    }
}
