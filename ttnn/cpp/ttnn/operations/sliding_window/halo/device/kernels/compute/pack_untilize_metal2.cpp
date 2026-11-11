// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// KEEP IN SYNC WITH: pack_untilize.cpp (this directory)
//
// This is the Metal 2.0 fork of that kernel. Same logic, expressed against the Metal 2.0
// named-binding APIs: the three CB-index CTAs became DFB bindings, the two remaining CTAs and the
// block-count RTA became named arguments. A behavioural change to either one must be mirrored in the
// other.
//
// The fork exists because the legacy original is bound by factories that cannot all convert at once
// (sliding_window/halo on the legacy host API, experimental/padded_slice on Metal 2.0); it lives
// alongside the original rather than replacing it. Once the last legacy consumer is ported, delete the
// original and rename this file over it.
//
// Binding vocabulary a Metal 2.0 KernelSpec must supply for this source:
//   dfb::src            — tiled input DFB, bound CONSUMER
//   dfb::out0           — untilized output DFB for even-numbered blocks, bound PRODUCER
//   dfb::out1           — untilized output DFB for odd-numbered blocks, bound PRODUCER; only when the
//                         factory defines SEPARATE_OUT1. Without the define, odd blocks are packed into
//                         dfb::out0 as well (one DFB under both of the kernel's output names).
//   args::tiles_per_row — compile-time arg
//   args::block_size    — compile-time arg
//   args::total_blocks  — runtime arg

#include <cstdint>

#include "ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/pack_untilize.h"
#include "experimental/kernel_args.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#ifdef PROLOGUE_AFFINE
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#endif

void kernel_main() {
    constexpr uint32_t src_dfb = dfb::src;
    constexpr uint32_t out_dfb0 = dfb::out0;
#ifdef SEPARATE_OUT1
    constexpr uint32_t out_dfb1 = dfb::out1;
#else
    constexpr uint32_t out_dfb1 = out_dfb0;
#endif
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);  // number of tiles along width of shard
    constexpr uint32_t block_size = get_arg(args::block_size);  // number of tiles along height that make up a block

    const uint32_t total_blocks = get_arg(args::total_blocks);

    compute_kernel_hw_startup(src_dfb, out_dfb0);
#ifdef PROLOGUE_AFFINE
    constexpr uint32_t scale_cb_id = dfb::scale;
    constexpr uint32_t shift_cb_id = dfb::shift;
    cb_wait_front(scale_cb_id, tiles_per_row);
    cb_wait_front(shift_cb_id, tiles_per_row);
    pack_untilize_init<1, tiles_per_row>(src_dfb, out_dfb0);
    add_binary_tile_init();
    silu_tile_init();
    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        const uint32_t out_cb_id = (block_idx % 2 == 0) ? out_dfb0 : out_dfb1;
        for (uint32_t r = 0; r < block_size; r++) {
            cb_wait_front(src_dfb, tiles_per_row);
            cb_reserve_back(out_cb_id, tiles_per_row);
            for (uint32_t c = 0; c < tiles_per_row; c++) {
                tile_regs_acquire();
                mul_bcast_rows_init(src_dfb, scale_cb_id);
                mul_tiles_bcast<BroadcastType::ROW>(src_dfb, scale_cb_id, c, c, 0);
                unary_bcast_init<BroadcastType::ROW>(shift_cb_id);
                unary_bcast<BroadcastType::ROW>(shift_cb_id, c, 1);
                add_binary_tile(0, 1, 0);
                silu_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<1, tiles_per_row>(out_cb_id, 1, c);
                tile_regs_release();
            }
            cb_pop_front(src_dfb, tiles_per_row);
            cb_push_back(out_cb_id, tiles_per_row);
        }
    }
    pack_untilize_uninit(out_dfb0);
    return;
#endif

    // Initialize once before the loop
    compute_kernel_lib::untilize_init<tiles_per_row, src_dfb, out_dfb0>();

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        // Use unified untilize with Neither mode since we handle init/uninit outside the loop
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_dfb,
                out_dfb0,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_dfb,
                out_dfb1,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        }
    }

    // Uninit after loop
    compute_kernel_lib::untilize_uninit<tiles_per_row, src_dfb, out_dfb0>();
}
