// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The untilize kernel for fp8 tokens: the same batches as untilize_combine_fabric2d.cpp, but each row of every
// scale block (four tiles, 128 values) is quantized on its way out, as per_token_cast_to_fp8 does with 1x128 pow2
// scales:
//   scale = 2^ceil(log2(max(amax, clamp_min) / 448)),   e4m3 = rne(x / scale).
//
// Per scale block: copy its four tiles into DEST, take each row's |x| max and fold the four into the exact
// 1/scale (SFPU), and pack that 1/scale tile twice -- once for the multiply, once for the dataflow kernel, which
// writes the scales after the row's values. Then x * (1/scale) by column broadcast, rounded to the e4m3 grid,
// packed straight into the batch row at e4m3 width. The packer's row stride is the whole token, scales included,
// so a row's values and its scale tail are one run in cb_out; the packer only ever writes the values.
//
// fp32 DEST holds four tiles, one scale block: hence one block per acquire. pack_untilize goes in pairs of tiles,
// because its block must divide the token's width in fp8 tile columns (198 at emb 6144), scales included.

#include <cstdint>
#include "api/compute/common.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/bcast.h"
#include "api/compute/pack_untilize.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"
#include "api/dataflow/circular_buffer.h"
#include "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/per_token_cast_to_fp8/device/kernels/compute/pow2_inv_scale_sfpu.hpp"

void kernel_main() {
    constexpr uint32_t cb_in_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_out_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_batches_id = get_compile_time_arg_val(2);
    constexpr uint32_t data_ct_dim = get_compile_time_arg_val(3);   // tiles of values per token
    constexpr uint32_t window_tiles = get_compile_time_arg_val(4);  // tiles the dataflow kernel stages at a time
    constexpr uint32_t batch_rows = get_compile_time_arg_val(5);
    constexpr uint32_t cb_inv_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_scale_id = get_compile_time_arg_val(7);
    constexpr uint32_t row_ct_dim = get_compile_time_arg_val(8);  // token width in fp8 tile columns, scales included
    constexpr uint32_t clamp_min_bits = get_compile_time_arg_val(9);
    constexpr uint32_t inv_448_bits = get_compile_time_arg_val(10);

    constexpr uint32_t block_tiles = 4;  // one scale block
    constexpr uint32_t blocks_per_window = window_tiles / block_tiles;
    constexpr uint32_t pack_tiles = 2;
    static_assert(window_tiles % block_tiles == 0, "the input window must hold whole scale blocks");
    static_assert(data_ct_dim % window_tiles == 0, "a token must be whole input windows");
    static_assert(row_ct_dim % pack_tiles == 0, "pack_untilize needs a block width that divides the row");

    CircularBuffer cb_in(cb_in_id);
    CircularBuffer cb_out(cb_out_id);
    CircularBuffer cb_inv(cb_inv_id);
    CircularBuffer cb_scale(cb_scale_id);
    CircularBuffer cb_batches(cb_batches_id);

    compute_kernel_hw_startup(cb_in_id, cb_inv_id, cb_out_id);

    cb_batches.wait_front(1);
    const uint32_t num_batches = read_tile_value(cb_batches_id, 0, 0);

    for (uint32_t b = 0; b < num_batches; b++) {
        cb_out.reserve_back(batch_rows);
        for (uint32_t w = 0; w < data_ct_dim / window_tiles; w++) {
            cb_in.wait_front(window_tiles);

            // Each block's 1/scale, from its row maxima.
            cb_inv.reserve_back(blocks_per_window);
            cb_scale.reserve_back(blocks_per_window);
            reconfig_full_operand_srca(cb_in_id);
            pack_reconfig_data_format(cb_inv_id);
            for (uint32_t blk = 0; blk < blocks_per_window; blk++) {
                copy_init(cb_in_id);
                tile_regs_acquire();
                for (uint32_t k = 0; k < block_tiles; k++) {
                    copy_tile(cb_in_id, blk * block_tiles + k, k);
                }
                MATH((SFPU_UNARY_INIT(unused)));
                for (uint32_t k = 0; k < block_tiles; k++) {
                    MATH((amax_rows_tile_call(k)));
                }
                MATH((fold4_pow2_inv_scale_tile(0, clamp_min_bits, inv_448_bits)));
                tile_regs_commit();
                tile_regs_wait();
                pack_tile(0, cb_inv_id);
                pack_tile(0, cb_scale_id);
                tile_regs_release();
            }
            cb_inv.push_back(blocks_per_window);
            cb_scale.push_back(blocks_per_window);

            // x * (1/scale), rounded to e4m3, into the batch rows.
            cb_inv.wait_front(blocks_per_window);
            reconfig_data_format(cb_in_id, cb_inv_id);
            mul_bcast_cols_init(cb_in_id, cb_inv_id);
            pack_untilize_dest_init<pack_tiles, row_ct_dim>(cb_out_id);
            for (uint32_t blk = 0; blk < blocks_per_window; blk++) {
                tile_regs_acquire();
                for (uint32_t k = 0; k < block_tiles; k++) {
                    mul_tiles_bcast_cols(cb_in_id, cb_inv_id, blk * block_tiles + k, blk, k);
                }
                MATH((SFPU_UNARY_INIT(unused)));
                for (uint32_t k = 0; k < block_tiles; k++) {
                    MATH((round_to_e4m3_rne_tile(k)));
                }
                tile_regs_commit();
                tile_regs_wait();
                const uint32_t col = (w * blocks_per_window + blk) * (block_tiles / pack_tiles);
                pack_untilize_dest<pack_tiles, row_ct_dim>(cb_out_id, 1, col, 0);
                pack_untilize_dest<pack_tiles, row_ct_dim>(cb_out_id, 1, col + 1, pack_tiles);
                tile_regs_release();
            }
            pack_untilize_uninit(cb_out_id);
            cb_in.pop_front(window_tiles);
            cb_inv.pop_front(blocks_per_window);
        }
        cb_out.push_back(batch_rows);
    }
}
