// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// DRAM DOWNLOAD FLOOR — reader half (NCRISC / NOC_0), the op's own split: W_gate + x.
//
// Issues EXACTLY the DRAM reads moe_fused_swiglu's reader issues for one M-block, at the same
// per-core slices, the same request lengths and through the same `TensorAccessor`, and then does
// NOTHING ELSE: no tilize, no multicast, no CB handshake with a consumer, no compute. All reads go
// in flight and ONE barrier closes them, so the measured time is the download itself rather than
// the op's per-chunk barrier schedule.
//
// WHY ONE BARRIER AND NOT THE OP'S SCHEDULE. The question this bench answers is "how long do these
// bytes take to arrive if nothing gets in the way", which is a CEILING on what any schedule could
// achieve. The op's real per-chunk barriers can only make it slower, and the gap between the two is
// exactly the quantity of interest.

#include <stdint.h>

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

constexpr uint32_t READ_WG = get_compile_time_arg_val(0);
constexpr uint32_t READ_X = get_compile_time_arg_val(1);
constexpr uint32_t HID_T = get_compile_time_arg_val(2);      // hidden tiles (W_gate row pitch)
constexpr uint32_t BFP4_TILE = get_compile_time_arg_val(3);  // 576
constexpr uint32_t X_ELEM = get_compile_time_arg_val(4);     // 2 = bf16 row-major, 0 = bfp8 tiles
constexpr uint32_t X_PAGE = get_compile_time_arg_val(5);     // accessor page: emb*2 (RM) or 1088 (tile)
constexpr uint32_t EMB_T = get_compile_time_arg_val(6);      // emb tiles (bfp8-tile x row pitch)
constexpr uint32_t TILE_H = 32;

constexpr uint32_t TA_BASE = 7;
constexpr auto wg_args = TensorAccessorArgs<TA_BASE>();
constexpr auto x_args = TensorAccessorArgs<wg_args.next_compile_time_args_offset()>();

constexpr uint32_t CB_WG = 0;
constexpr uint32_t CB_X = 3;

void kernel_main() {
    const uint32_t wg_addr = get_arg_val<uint32_t>(0);
    const uint32_t x_addr = get_arg_val<uint32_t>(1);
    const uint32_t kstart = get_arg_val<uint32_t>(2);  // my grid row's first emb K-tile
    const uint32_t kr = get_arg_val<uint32_t>(3);      // my K-tile count
    const uint32_t hstart = get_arg_val<uint32_t>(4);  // my grid column's first hidden tile
    const uint32_t hn = get_arg_val<uint32_t>(5);      // my hidden-tile count
    const uint32_t x_rows = get_arg_val<uint32_t>(6);  // tile-rows of x THIS core stages (0 = not an injector)
    const uint32_t x_row0 = get_arg_val<uint32_t>(7);  // first tile-row I stage

    if constexpr (READ_WG) {
        const auto wg_acc = TensorAccessor(wg_args, wg_addr, BFP4_TILE);
        cb_reserve_back(CB_WG, kr * hn);
        const uint32_t wp = get_write_ptr(CB_WG);
        // ONE request per K-row: the ND shard makes a K-row's `hn` tiles contiguous in one bank, so
        // this is the same transaction shape the op's coalesced path issues (hn*576 B).
        for (uint32_t k = 0; k < kr; ++k) {
            noc_async_read(wg_acc.get_noc_addr((kstart + k) * HID_T + hstart), wp + k * hn * BFP4_TILE, hn * BFP4_TILE);
        }
    }

    if constexpr (READ_X) {
        const auto x_acc = TensorAccessor(x_args, x_addr, X_PAGE);
        if (x_rows) {
            // The reserve COUNT is format-specific and must match the host's CB page count: bf16 lands
            // 32 stick-slices per tile-row, bfp8 lands `kr` whole tiles. Reserving 32 in the bfp8 case
            // asks for more pages than the CB holds, and `cb_reserve_back` then blocks forever.
            cb_reserve_back(CB_X, (X_ELEM == 2) ? (x_rows * TILE_H) : (x_rows * kr));
            const uint32_t xp = get_write_ptr(CB_X);
            if constexpr (X_ELEM == 2) {
                // bf16 ROW_MAJOR: one page is one `emb` stick; take my K-slice out of each of the 32
                // sticks of every tile-row I own. This is the sub-page read shape `reader_xstage` uses.
                const uint32_t slice = kr * TILE_H * 2;
                for (uint32_t r = 0; r < x_rows; ++r) {
                    for (uint32_t s = 0; s < TILE_H; ++s) {
                        noc_async_read(
                            x_acc.get_noc_addr((x_row0 + r) * TILE_H + s, kstart * TILE_H * 2),
                            xp + (r * TILE_H + s) * slice,
                            slice);
                    }
                }
            } else {
                // bfp8 TILE: whole tiles, no stick walk — the INPUT_FORMAT==1 twin.
                for (uint32_t r = 0; r < x_rows; ++r) {
                    for (uint32_t t = 0; t < kr; ++t) {
                        noc_async_read(
                            x_acc.get_noc_addr((x_row0 + r) * EMB_T + kstart + t), xp + (r * kr + t) * X_PAGE, X_PAGE);
                    }
                }
            }
        }
    }

    // ONE barrier for every stream this RISC issued — see the header.
    noc_async_read_barrier();
}
