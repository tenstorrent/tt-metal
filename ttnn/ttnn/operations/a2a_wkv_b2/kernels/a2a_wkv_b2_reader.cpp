// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 worker reader (NCRISC, NoC 0).
//
// Per head h owned by this core:
//   load_weight_block : W[h] = Kt x Nt bf8_b tiles -> cb_weight, one barrier per head (resident across mt)
//   load_attn_block   : mt_block*32 attn sticks (one token of head h each) -> cb_attn_rm, via
//                       read_sticks_for_tilize (one barrier per 32-stick tile row)
//
// load_weight_block is raw TensorAccessor reads: no kernel_lib helper reads already-tiled pages
// (read_sticks_for_tilize reads row-major sticks into a tilize CB) — see op_design.md "Helpers
// considered and rejected".

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_weight = get_compile_time_arg_val(0);
    constexpr uint32_t cb_attn_rm = get_compile_time_arg_val(1);
    constexpr uint32_t Kt = get_compile_time_arg_val(2);
    constexpr uint32_t Nt = get_compile_time_arg_val(3);
    constexpr uint32_t Mt = get_compile_time_arg_val(4);
    constexpr uint32_t mt_block = get_compile_time_arg_val(5);
    constexpr uint32_t tokens = get_compile_time_arg_val(6);  // T: sticks per head
    constexpr uint32_t stick_bytes = get_compile_time_arg_val(7);
    constexpr uint32_t attn_page_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t w_tile_bytes = get_compile_time_arg_val(9);
    constexpr auto attn_args = TensorAccessorArgs<10>();
    constexpr auto w_args = TensorAccessorArgs<attn_args.next_compile_time_args_offset()>();

    constexpr uint32_t tile_h = 32;
    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t w_block_tiles = Kt * Nt;

    const uint32_t attn_addr = get_arg_val<uint32_t>(0);
    const uint32_t w_addr = get_arg_val<uint32_t>(1);
    const uint32_t head_start = get_arg_val<uint32_t>(2);
    const uint32_t num_heads = get_arg_val<uint32_t>(3);

    const auto attn = TensorAccessor(attn_args, attn_addr, attn_page_bytes);
    const auto w = TensorAccessor(w_args, w_addr, w_tile_bytes);

    for (uint32_t head_idx = 0; head_idx < num_heads; ++head_idx) {
        const uint32_t h = head_start + head_idx;

        // load_weight_block: W[h] resident across all of the head's mt blocks.
        cb_reserve_back(cb_weight, w_block_tiles);
        uint32_t l1 = get_write_ptr(cb_weight);
        const uint32_t w_tile0 = h * w_block_tiles;
        for (uint32_t t = 0; t < w_block_tiles; ++t) {
            noc_async_read(w.get_noc_addr(w_tile0 + t), l1, w_tile_bytes);
            l1 += w_tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_weight, w_block_tiles);

        // load_attn_block: mt_block tile rows of this head's tokens per block (ragged last block).
        for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
            const uint32_t tok0 = mb * mt_block * tile_h;
            const uint32_t rows = (tokens - tok0) < mt_block * tile_h ? (tokens - tok0) : mt_block * tile_h;
            dataflow_kernel_lib::read_sticks_for_tilize<cb_attn_rm>(attn, rows, stick_bytes, h * tokens + tok0);
        }
    }
}
