// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

#include "api/compute/compute_kernel_api.h"
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/bcast.h"
#include "api/compute/pack.h"
#include "api/debug/kernel_profiler.hpp"

// persistent_moe_request with the hyper-connection post step fused (see dataflow/moe_partials_hcpost.cpp), on one
// stream-collapse core: this core's columns of the new streams,
//     new[h] = post[h] * x + sum_g comb[g, h] * old[g],     x = the sum of the op's chips' partials.
// Every row part (a slot, an old or a new stream row) is ONE 32x32 bf16 tile: its CB's pages are part_bytes apart and
// unpack/pack address tile i at i * page size, so a part's elements are the tile's first ones and the rest are
// don't-cares. Every step is elementwise or a scalar broadcast, so the don't-cares never reach a part's elements --
// tt-blaze's BcastMulAdd views its [1, 32] rows as [T, 32] tiles the same way -- and the FPU runs on whole tiles
// (its eltwise ops are wrong on the 4x32 tiles a part would otherwise take). post and comb are scalars at element 0
// of their pages. New rows are packed in order, each over the previous one's don't-care tail.
//   1. While the partials are on their way: mix[h] = sum_g comb[g, h] * old[g] into cb_mix (products through
//      cb_prod, then summed in DEST).
//   2. Once they are in, the critical path: x = their sum in slot (chip) order (dest-reuse adds, as
//      moe_partials_sum.cpp) into cb_x; then new[h] = post[h] * x + mix[h] into cb_out tile h.
//
// A several-row block (rows > 1, DSpark's verify block) has per-row post and comb, and its part (this core's columns
// of every row, as the block's (rows x 32) tiles) is tiles_per_part 32x32 views, page_bytes apart within a part of
// part_pages pages: a view's 32x32 row R holds block row R % rows (a 16-element face row is one block row's, and the
// block's face rows cycle through its rows), so a per-row scalar is a column broadcast of a tile whose column 0 holds
// coef[R % rows] (the reader stages these). Each view runs past its part's end by a don't-care tail, and new rows are
// packed in address order, each view over the previous one's tail.
//   1. mix[h] = sum_g comb[g, h] * old[g], every view, as column-broadcast MACs in DEST, into cb_mix (tile h * T + t);
//   2. x = the slots' sum, view by view, each slot as it lands; new[h] = post[h] * x + mix[h] into cb_out (page h *
//   part_pages + t * tile_pages).
//
// Compile-time args: 0: num_slots   1: num_streams   2: cb_slots   3: cb_s (old streams)   4: cb_post
//   5: cb_comb (comb[g, h] at page g * num_streams + h)   6: cb_mix   7: cb_x   8: cb_out   9: cb_prod
//   10: rows   11: tiles_per_part (views per part, rows > 1)   12: part_pages   13: tile_pages (a view's pages)
// Runtime args: 0: skip (a stop request)

namespace {

// Several rows: see above. DEST holds 8 bf16 tiles per half.
template <
    uint32_t num_slots,
    uint32_t streams,
    uint32_t cb_slots,
    uint32_t cb_s,
    uint32_t cb_post,
    uint32_t cb_comb,
    uint32_t cb_mix,
    uint32_t cb_x,
    uint32_t cb_out,
    uint32_t views,
    uint32_t part_pages,
    uint32_t tile_pages>
void block_post() {
    constexpr auto kReuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;
    constexpr uint32_t kDest = 8;
    static_assert(views <= kDest, "a part's views must fit DEST");
    constexpr uint32_t streams_per_pass = kDest / views;
    compute_kernel_hw_startup(cb_s, cb_comb, cb_mix);

    // 1. ELWMUL accumulates into DEST (which the packer clears on release), so the column-broadcast products of the
    // old streams sum in place.
    cb_wait_front(cb_s, streams * part_pages);
    cb_wait_front(cb_comb, streams * streams);
    cb_reserve_back(cb_mix, streams * views);
    {
        DeviceZoneScopedN("MOE_HC_MIX");
        mul_bcast_cols_init(cb_s, cb_comb, true /*acc_to_dest*/);
        for (uint32_t h = 0; h < streams; ++h) {
            tile_regs_acquire();
            for (uint32_t t = 0; t < views; ++t) {
                for (uint32_t g = 0; g < streams; ++g) {
                    mul_tiles_bcast_cols(cb_s, cb_comb, g * part_pages + t * tile_pages, g * streams + h, t);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t t = 0; t < views; ++t) {
                pack_tile<true>(t, cb_mix, h * views + t);
            }
            tile_regs_release();
        }
    }
    cb_push_back(cb_mix, streams * views);

    // 2. The reader publishes the slots one at a time, in chip order, as each chip's partial lands: x sums them
    // as they come (in that fixed order, so the sum's bits do not depend on the arrival order).
    DeviceZoneScopedN("MOE_HC_POST");
    cb_reserve_back(cb_x, views);
    tile_regs_acquire();
    cb_wait_front(cb_slots, part_pages);
    copy_init(cb_slots);
    for (uint32_t t = 0; t < views; ++t) {
        copy_tile(cb_slots, t * tile_pages, t);
    }
    add_reuse_dest_init<kReuse>(cb_slots);
    for (uint32_t m = 1; m < num_slots; ++m) {
        cb_wait_front(cb_slots, (m + 1) * part_pages);
        for (uint32_t t = 0; t < views; ++t) {
            add_reuse_dest_tiles<kReuse>(cb_slots, m * part_pages + t * tile_pages, t);
        }
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t t = 0; t < views; ++t) {
        pack_tile(t, cb_x);
    }
    tile_regs_release();
    cb_push_back(cb_x, views);

    cb_wait_front(cb_x, views);
    cb_wait_front(cb_post, streams);
    cb_wait_front(cb_mix, streams * views);
    cb_reserve_back(cb_out, streams * part_pages);
    for (uint32_t first = 0; first < streams; first += streams_per_pass) {
        const uint32_t last = first + streams_per_pass < streams ? first + streams_per_pass : streams;
        tile_regs_acquire();
        mul_bcast_cols_init(cb_x, cb_post);
        for (uint32_t h = first; h < last; ++h) {
            for (uint32_t t = 0; t < views; ++t) {
                mul_tiles_bcast_cols(cb_x, cb_post, t, h, (h - first) * views + t);
            }
        }
        add_reuse_dest_init<kReuse>(cb_mix);
        for (uint32_t h = first; h < last; ++h) {
            for (uint32_t t = 0; t < views; ++t) {
                add_reuse_dest_tiles<kReuse>(cb_mix, h * views + t, (h - first) * views + t);
            }
        }
        tile_regs_commit();
        tile_regs_wait();
        // In address order: a view's don't-care tail lies under the next one.
        for (uint32_t h = first; h < last; ++h) {
            for (uint32_t t = 0; t < views; ++t) {
                pack_tile<true>((h - first) * views + t, cb_out, h * part_pages + t * tile_pages);
            }
        }
        tile_regs_release();
    }
    cb_push_back(cb_out, streams * part_pages);
    cb_pop_front(cb_slots, num_slots * part_pages);
    cb_pop_front(cb_x, views);
    cb_pop_front(cb_mix, streams * views);
    cb_pop_front(cb_post, streams);
    cb_pop_front(cb_comb, streams * streams);
    cb_pop_front(cb_s, streams * part_pages);
}

}  // namespace

void kernel_main() {
    constexpr uint32_t num_slots = get_compile_time_arg_val(0);
    constexpr uint32_t streams = get_compile_time_arg_val(1);
    constexpr uint32_t cb_slots = get_compile_time_arg_val(2);
    constexpr uint32_t cb_s = get_compile_time_arg_val(3);
    constexpr uint32_t cb_post = get_compile_time_arg_val(4);
    constexpr uint32_t cb_comb = get_compile_time_arg_val(5);
    constexpr uint32_t cb_mix = get_compile_time_arg_val(6);
    constexpr uint32_t cb_x = get_compile_time_arg_val(7);
    constexpr uint32_t cb_out = get_compile_time_arg_val(8);
    constexpr uint32_t cb_prod = get_compile_time_arg_val(9);
    constexpr uint32_t rows = get_compile_time_arg_val(10);
    constexpr auto kReuse = EltwiseBinaryReuseDestType::DEST_TO_SRCA;

    if (get_arg_val<uint32_t>(0) != 0u) {
        return;
    }
    if constexpr (rows > 1) {
        block_post<
            num_slots,
            streams,
            cb_slots,
            cb_s,
            cb_post,
            cb_comb,
            cb_mix,
            cb_x,
            cb_out,
            get_compile_time_arg_val(11),
            get_compile_time_arg_val(12),
            get_compile_time_arg_val(13)>();
        return;
    }
    // Every CB is bf16 32x32: no data format reconfiguration anywhere.
    compute_kernel_hw_startup(cb_s, cb_comb, cb_prod);

    // 1.
    cb_wait_front(cb_s, streams);
    cb_wait_front(cb_comb, streams * streams);
    cb_reserve_back(cb_mix, streams);
    {
        DeviceZoneScopedN("MOE_HC_MIX");
        for (uint32_t h = 0; h < streams; ++h) {
            cb_reserve_back(cb_prod, streams);
            tile_regs_acquire();
            mul_bcast_scalar_init(cb_s, cb_comb);
            for (uint32_t g = 0; g < streams; ++g) {
                mul_tiles_bcast_scalar(cb_s, cb_comb, g, g * streams + h, g);
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t g = 0; g < streams; ++g) {
                pack_tile<true>(g, cb_prod, g);
            }
            tile_regs_release();
            cb_push_back(cb_prod, streams);

            cb_wait_front(cb_prod, streams);
            tile_regs_acquire();
            copy_init(cb_prod);
            copy_tile(cb_prod, 0, 0);
            add_reuse_dest_init<kReuse>(cb_prod);
            for (uint32_t g = 1; g < streams; ++g) {
                add_reuse_dest_tiles<kReuse>(cb_prod, g, 0);
            }
            tile_regs_commit();
            tile_regs_wait();
            pack_tile<true>(0, cb_mix, h);
            tile_regs_release();
            cb_pop_front(cb_prod, streams);
        }
    }
    cb_push_back(cb_mix, streams);

    // 2.
    cb_wait_front(cb_slots, num_slots);
    DeviceZoneScopedN("MOE_HC_POST");
    cb_reserve_back(cb_x, 1);
    tile_regs_acquire();
    copy_init(cb_slots);
    copy_tile(cb_slots, 0, 0);
    add_reuse_dest_init<kReuse>(cb_slots);
    for (uint32_t m = 1; m < num_slots; ++m) {
        add_reuse_dest_tiles<kReuse>(cb_slots, m, 0);
    }
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, cb_x);
    tile_regs_release();
    cb_push_back(cb_x, 1);

    cb_wait_front(cb_x, 1);
    cb_wait_front(cb_post, streams);
    cb_wait_front(cb_mix, streams);
    cb_reserve_back(cb_out, streams);
    tile_regs_acquire();
    mul_bcast_scalar_init(cb_x, cb_post);
    for (uint32_t h = 0; h < streams; ++h) {
        mul_tiles_bcast_scalar(cb_x, cb_post, 0, h, h);
    }
    add_reuse_dest_init<kReuse>(cb_mix);
    for (uint32_t h = 0; h < streams; ++h) {
        add_reuse_dest_tiles<kReuse>(cb_mix, h, h);
    }
    tile_regs_commit();
    tile_regs_wait();
    for (uint32_t h = 0; h < streams; ++h) {
        pack_tile<true>(h, cb_out, h);
    }
    tile_regs_release();
    cb_push_back(cb_out, streams);
    cb_pop_front(cb_slots, num_slots);
    cb_pop_front(cb_x, 1);
    cb_pop_front(cb_mix, streams);
    cb_pop_front(cb_post, streams);
    cb_pop_front(cb_comb, streams * streams);
    cb_pop_front(cb_s, streams);
}
