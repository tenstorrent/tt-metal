// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// matmul_reduce_scatter — transport-core TRISC: relay_add_block / final_add_block.
//
// cb_xport_sum = cb_xport_partial + arrival A [+ arrival B], accumulated in fp32 DEST (the transport kernels always
// run with fp32_dest_acc_en=True) and packed bf16. A relay port has (partial, A); a final core has (partial, A, B),
// or one of the two arrivals at a line end.
// A ring port's list starts with entries that have no upstream (the chip's own partial starts the block's chain):
// their tiles are copied through partial -> sum first (cb_xport_sum keeps exactly one producer), then the relay
// entries are added.
// The walk is segment by segment: every segment (the sender's unit) is waited for, added, pushed and popped on its
// own -- never held back for tiles of the next segment (a cross-segment lookahead chains into the upstream chip
// and can close a wait cycle on short block lists). Segments never straddle the CB wrap (capacity is a multiple of
// seg_tiles). Within a segment, DEST blocks are add_block (<= 4, the fp32 half-sync capacity) tiles.
//
// The add init is issued once for the whole walk. Three inputs are two DEST-accumulating adds under that one
// ELWADD(acc_to_dest) init -- dest += partial + A; dest += B + 0 (cb_xport_zero, one zero tile) -- so the running
// sum never leaves fp32 DEST and no init is re-issued per tile. acc_to_dest needs DEST == 0 at every acquire: each
// release zeroes the half it frees (pack dest-section-done), so two empty acquire/release rounds at start clear both
// halves, and the next round packs the zero tile from a cleared half.

#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/pack.h"
#include "api/compute/reg_api.h"
#include "api/compute/cb_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/perf_instrumentation.hpp"

// Stage zones (permanent; opt-in via KERNEL_PERF_ZONES): xadd_copy / xadd_add span the whole copy-through / add walk
// (the walk waits per segment on the reader and reserves on the sender / writer, so occupancy, not payload).

using namespace compute_kernel_lib;

void kernel_main() {
    constexpr uint32_t cb_xport_partial = get_compile_time_arg_val(0);
    constexpr uint32_t cb_arrival_a = get_compile_time_arg_val(1);
    constexpr uint32_t cb_arrival_b = get_compile_time_arg_val(2);
    constexpr uint32_t cb_xport_sum = get_compile_time_arg_val(3);
    constexpr uint32_t has_a = get_compile_time_arg_val(4);
    constexpr uint32_t has_b = get_compile_time_arg_val(5);
    constexpr uint32_t add_block = get_compile_time_arg_val(6);  // tiles per DEST batch (<= seg_tiles, <= 4)
    constexpr uint32_t seg_tiles = get_compile_time_arg_val(7);
    constexpr uint32_t cb_xport_zero = get_compile_time_arg_val(8);  // one zero tile (three-input finals only)
    const uint32_t num_copy_segs = get_arg_val<uint32_t>(0);  // upstream-less entries: copied through
    const uint32_t num_segs = get_arg_val<uint32_t>(1);       // relay entries (or the finals' block): added
    static_assert(add_block >= 1 && add_block <= 4, "fp32 half-sync DEST holds 4 tiles");

    constexpr uint32_t cb_second = has_a ? cb_arrival_a : cb_arrival_b;
    constexpr bool three = has_a && has_b;
    constexpr auto in_cfg = [](uint32_t cb) {
        return input(cb, WaitPolicy::PerBlockSize, PopPolicy::PerBlockSize, InputTileMapping::Block);
    };
    compute_kernel_hw_startup(cb_xport_partial, cb_second, cb_xport_sum);
    if (num_copy_segs > 0) {
        MaybeDeviceZoneScope("xadd_copy");
        eltwise_chain(
            IterationShape::grid(num_copy_segs, seg_tiles).block_size(add_block),
            CopyTile<in_cfg(cb_xport_partial)>{},
            PackTile<output(cb_xport_sum, ReservePolicy::PerBlockSize, PushPolicy::PerBlockSize)>{});
    }
    if (num_segs == 0) {
        return;
    }
    MaybeDeviceZoneScope("xadd_add");
    if constexpr (three) {
        for (uint32_t r = 0; r < 2; ++r) {  // zero both DEST halves (each release clears the half it frees)
            tile_regs_acquire();
            tile_regs_commit();
            tile_regs_wait();
            tile_regs_release();
        }
        cb_reserve_back(cb_xport_zero, 1);
        tile_regs_acquire();
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, cb_xport_zero);
        tile_regs_release();
        cb_push_back(cb_xport_zero, 1);
        cb_wait_front(cb_xport_zero, 1);  // never popped
    }
    add_init(cb_xport_partial, cb_second, /*acc_to_dest=*/three);

    for (uint32_t s = 0; s < num_segs; ++s) {
        cb_wait_front(cb_xport_partial, seg_tiles);
        cb_wait_front(cb_second, seg_tiles);
        if constexpr (three) {
            cb_wait_front(cb_arrival_b, seg_tiles);
        }
        cb_reserve_back(cb_xport_sum, seg_tiles);
        for (uint32_t off = 0; off < seg_tiles; off += add_block) {
            const uint32_t n = (seg_tiles - off) < add_block ? (seg_tiles - off) : add_block;
            tile_regs_acquire();
            for (uint32_t i = 0; i < n; ++i) {
                add_tiles(cb_xport_partial, cb_second, off + i, off + i, i);
            }
            if constexpr (three) {
                for (uint32_t i = 0; i < n; ++i) {
                    add_tiles(cb_arrival_b, cb_xport_zero, off + i, 0, i);
                }
            }
            tile_regs_commit();
            tile_regs_wait();
            for (uint32_t i = 0; i < n; ++i) {
                pack_tile(i, cb_xport_sum);
            }
            tile_regs_release();
        }
        cb_pop_front(cb_xport_partial, seg_tiles);
        cb_pop_front(cb_second, seg_tiles);
        if constexpr (three) {
            cb_pop_front(cb_arrival_b, seg_tiles);
        }
        cb_push_back(cb_xport_sum, seg_tiles);
    }
}
