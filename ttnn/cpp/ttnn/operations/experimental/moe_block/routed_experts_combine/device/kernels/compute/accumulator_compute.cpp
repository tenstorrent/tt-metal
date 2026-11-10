// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Accumulator compute: sums each token's routed rows, weighted by their gate weights, in fp32.
//
// For every row the reader hands over (the row, its gate weight, the token's local index) it computes
//
//     sum[token] += row * weight
//
// The multiply is a broadcast multiply: srcA is the row, srcB the gate weight (element 0 of its tile, broadcast to
// every element), the product lands in DEST. The add is done by the packer: packing with L1 accumulation on adds DEST
// onto the token's fp32 sum already in cb_acc (a token's first row packs without it, so stale L1 is overwritten).
// After the last row, every token's fp32 sum goes out as bf16 (cb_out); a token that got no rows is written as zeros.
//
// The rows are row-major data, but they are processed as if they were tiles: a W-wide bf16 row is W / 1024 "tiles".
// That is fine because everything here is elementwise - the order of the elements inside a tile does not matter, and
// they come back out in the same order.
//
// combine_snap (snap_offset_bits != 0): each product is first rounded to a multiple of q on the SFPU, as (v + C) - C
// with C = 1.5 x 2^23 q. Every fp32 add of such terms is exact while the sums stay below 2^24 q, so a token's sum is
// the same whatever order its rows land in - the combine is deterministic.
//
// The reader's meta pages (cb_meta, one uint32 each):
//
//   page        | value
//   ------------+--------------------------------------------------------------------------------------------
//   first       | 0x80000000 | this accumulator's tokens, or 0: no accumulator on this core on this chip (return)
//   one per row | the row's local token index (token / A: tokens are dealt to the accumulators round-robin)
//   last        | 0xFFFFFFFF: no more rows (the compute does not know the row count)

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/pack.h"
#include "api/compute/cb_api.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/bcast.h"
#include "api/compute/eltwise_unary/binop_with_scalar.h"
#include "api/compute/eltwise_unary/fill.h"
#include "ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp"

// fp32 tiles DEST holds at once (fp32 accumulation, half sync: 4)
constexpr uint32_t kDestTiles = compute_kernel_lib::DEST_AUTO_LIMIT;
constexpr uint32_t kMetaActive = 0x80000000u;
constexpr uint32_t kMetaEnd = 0xFFFFFFFFu;

void kernel_main() {
    constexpr uint32_t cb_row = get_compile_time_arg_val(0);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(1);
    constexpr uint32_t cb_acc = get_compile_time_arg_val(2);
    constexpr uint32_t cb_out = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(4);
    constexpr uint32_t max_tokens = get_compile_time_arg_val(5);
    constexpr uint32_t cb_w = get_compile_time_arg_val(6);
    constexpr uint32_t snap_offset_bits = get_compile_time_arg_val(7);  // (fp32 bits of C = 1.5 x 2^23 q; 0: off)
    compute_kernel_hw_startup(cb_row, cb_w, cb_acc);
    // (no runtime args: whether this core accumulates differs by chip, and one program for the mesh needs them alike)
    cb_wait_front(cb_meta, 1);
    const uint32_t first_page = read_tile_value(cb_meta, 0, 0);
    cb_pop_front(cb_meta, 1);
    if (first_page == 0) {
        return;
    }
    const uint32_t n_tokens = first_page & ~kMetaActive;
    mul_bcast_scalar_init(cb_row, cb_w);
    uint8_t has_rows[max_tokens];
    for (uint32_t i = 0; i < max_tokens; ++i) {
        has_rows[i] = 0;
    }
    cb_reserve_back(cb_acc, n_tokens * tiles_per_row);
    while (true) {
        cb_wait_front(cb_meta, 1);
        const uint32_t token = read_tile_value(cb_meta, 0, 0);
        cb_pop_front(cb_meta, 1);
        if (token == kMetaEnd) {
            break;
        }
        cb_wait_front(cb_row, tiles_per_row);
        cb_wait_front(cb_w, 1);
        pack_reconfig_l1_acc(has_rows[token] ? 1 : 0);
        for (uint32_t first = 0; first < tiles_per_row; first += kDestTiles) {
            const uint32_t count = tiles_per_row - first < kDestTiles ? tiles_per_row - first : kDestTiles;
            tile_regs_acquire();
            if constexpr (snap_offset_bits) {
                mul_bcast_scalar_init(cb_row, cb_w);
            }
            for (uint32_t j = 0; j < count; ++j) {
                mul_tiles_bcast_scalar(cb_row, cb_w, first + j, 0, j);
            }
            if constexpr (snap_offset_bits) {  // every term to a multiple of q: (v + C) - C
                binop_with_scalar_tile_init();
                for (uint32_t j = 0; j < count; ++j) {
                    add_unary_tile(j, snap_offset_bits);
                    sub_unary_tile(j, snap_offset_bits);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < count; ++j) {
                pack_tile<true>(j, cb_acc, token * tiles_per_row + first + j);
            }
            tile_regs_release();
        }
        has_rows[token] = 1;
        cb_pop_front(cb_row, tiles_per_row);
        cb_pop_front(cb_w, 1);
    }
    pack_reconfig_l1_acc(0);
    cb_push_back(cb_acc, n_tokens * tiles_per_row);
    cb_wait_front(cb_acc, n_tokens * tiles_per_row);
    reconfig_data_format_srca(cb_acc);
    pack_reconfig_data_format(cb_out);
    copy_init(cb_acc);
    for (uint32_t token = 0; token < n_tokens; ++token) {
        cb_reserve_back(cb_out, tiles_per_row);
        if (!has_rows[token]) {  // (a padding token: zeros, not the CB's stale L1)
            fill_tile_init();
            for (uint32_t first = 0; first < tiles_per_row; first += kDestTiles) {
                const uint32_t count = tiles_per_row - first < kDestTiles ? tiles_per_row - first : kDestTiles;
                tile_regs_acquire();
                for (uint32_t j = 0; j < count; ++j) {
                    fill_tile(j, 0.0f);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t j = 0; j < count; ++j) {
                    pack_tile(j, cb_out);
                }
                tile_regs_release();
            }
            cb_push_back(cb_out, tiles_per_row);
            copy_init(cb_acc);
            continue;
        }
        for (uint32_t first = 0; first < tiles_per_row; first += kDestTiles) {
            const uint32_t count = tiles_per_row - first < kDestTiles ? tiles_per_row - first : kDestTiles;
            tile_regs_acquire();
            for (uint32_t j = 0; j < count; ++j) {
                copy_tile(cb_acc, token * tiles_per_row + first + j, j);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t j = 0; j < count; ++j) {
                pack_tile(j, cb_out);
            }
            tile_regs_release();
        }
        cb_push_back(cb_out, tiles_per_row);
    }
}
