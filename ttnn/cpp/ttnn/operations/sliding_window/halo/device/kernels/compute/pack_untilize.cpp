// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: A Metal 2.0 fork of this kernel lives beside it, as
// pack_untilize_metal2.cpp. Ops ported to Metal 2.0 bind the fork; this file serves
// the consumers still on the legacy API. Until the last of them migrates and
// this file is retired, changes here likely belong in the fork too.

#include <cstdint>

#include "ttnn/kernel_lib/untilize_helpers.hpp"
#include "api/compute/pack_untilize.h"

constexpr uint32_t MAX_PACK_UNTILIZE_WIDTH = 8;
constexpr uint32_t NUM_RISCV_DATA_MOVEMENT_CORES = 2;
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#ifdef PROLOGUE_AFFINE
#include "api/compute/bcast.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"
#endif

void kernel_main() {
    constexpr uint32_t src_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t out_cb_id0 = get_compile_time_arg_val(1);
    constexpr uint32_t out_cb_id1 = get_compile_time_arg_val(2);
    constexpr uint32_t tiles_per_row = get_compile_time_arg_val(3);  // number of tiles along width of shard
    constexpr uint32_t block_size = get_compile_time_arg_val(4);  // number of tiles along height that make up a block

    const uint32_t total_blocks = get_arg_val<uint32_t>(0);

    compute_kernel_hw_startup(src_cb_id, out_cb_id0);
#ifdef PROLOGUE_AFFINE
    constexpr uint32_t scale_cb_id = get_compile_time_arg_val(5);
    constexpr uint32_t shift_cb_id = get_compile_time_arg_val(6);
    cb_wait_front(scale_cb_id, tiles_per_row);
    cb_wait_front(shift_cb_id, tiles_per_row);
    pack_untilize_init<1, tiles_per_row>(src_cb_id, out_cb_id0);
    add_binary_tile_init();
    silu_tile_init();
    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        const uint32_t out_cb_id = (block_idx % 2 == 0) ? out_cb_id0 : out_cb_id1;
        for (uint32_t r = 0; r < block_size; r++) {
            cb_wait_front(src_cb_id, tiles_per_row);
            cb_reserve_back(out_cb_id, tiles_per_row);
            for (uint32_t c = 0; c < tiles_per_row; c++) {
                tile_regs_acquire();
                mul_bcast_rows_init(src_cb_id, scale_cb_id);
                mul_tiles_bcast<BroadcastType::ROW>(src_cb_id, scale_cb_id, c, c, 0);
                unary_bcast_init<BroadcastType::ROW>(shift_cb_id);
                unary_bcast<BroadcastType::ROW>(shift_cb_id, c, 1);
                add_binary_tile(0, 1, 0);
                silu_tile(0);
                tile_regs_commit();
                tile_regs_wait();
                pack_untilize_dest<1, tiles_per_row>(out_cb_id, 1, c);
                tile_regs_release();
            }
            cb_pop_front(src_cb_id, tiles_per_row);
            cb_push_back(out_cb_id, tiles_per_row);
        }
    }
    pack_untilize_uninit(out_cb_id0);
    return;
#endif

    // Initialize once before the loop
    compute_kernel_lib::untilize_init<tiles_per_row, src_cb_id, out_cb_id0>();

    for (uint32_t block_idx = 0; block_idx < total_blocks; block_idx++) {
        // Use unified untilize with Neither mode since we handle init/uninit outside the loop
        if (block_idx % 2 == 0) {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id0,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        } else {
            compute_kernel_lib::untilize<
                tiles_per_row,
                src_cb_id,
                out_cb_id1,
                compute_kernel_lib::untilize_config::InitUninitMode::Neither,
                compute_kernel_lib::untilize_config::WaitMode::WaitBlock,
                compute_kernel_lib::untilize_config::ReconfigureRegisterDatatypeMode::NoReconfigure>(block_size);
        }
    }

    // Uninit after loop
    compute_kernel_lib::untilize_uninit<tiles_per_row, src_cb_id, out_cb_id0>();
}
