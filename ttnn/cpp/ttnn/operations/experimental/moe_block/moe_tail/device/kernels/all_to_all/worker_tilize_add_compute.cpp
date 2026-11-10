// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// worker_tilize_add_compute: an all-to-all worker's compute - each owned row tile's worker_columns tiles tilized
// (CB_ROWS -> CB_TILIZED, or straight to CB_OUTPUT without a shared expert), then + the shared expert's tiles into
// CB_OUTPUT, for each row block in turn. skip_padding: only the live row tiles (the reader's mask, CB_LIVE_ROW_TILES).

#include <cstdint>
#include "api/compute/compute_kernel_api.h"
#include "api/compute/cb_api.h"
#include "api/compute/pack.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/reconfig_data_format.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/moe_block/moe_tail/device/kernels/all_to_all/padded_row_tiles.hpp"

// (moe_tail_program_factory's all_to_all_cb)
constexpr uint32_t CB_ROWS = 0, CB_TILIZED = 1, CB_SHARED_EXPERT = 2, CB_OUTPUT = 3;

void kernel_main() {
    constexpr uint32_t worker_columns = get_compile_time_arg_val(0);
    constexpr bool has_shared_expert = get_compile_time_arg_val(1) != 0;
    constexpr uint32_t row_blocks = get_compile_time_arg_val(2);
    constexpr bool skip_padding = get_named_compile_time_arg_val("skip_padding") != 0;
    // (without a shared expert the tilized tiles are the output)
    constexpr uint32_t tilize_output_cb = has_shared_expert ? CB_TILIZED : CB_OUTPUT;
    const uint32_t owned_row_tiles = get_arg_val<uint32_t>(0);
    compute_kernel_hw_startup(CB_ROWS, CB_SHARED_EXPERT, CB_OUTPUT);
    for (uint32_t block = 0; block < row_blocks; ++block) {
        uint64_t live_mask = 0;
        if constexpr (skip_padding) {
            cb_wait_front(CB_LIVE_ROW_TILES, 1);  // (the reader's live row tiles of this block: every thread the same)
            live_mask =
                read_tile_value(CB_LIVE_ROW_TILES, 0, 0) | (uint64_t(read_tile_value(CB_LIVE_ROW_TILES, 0, 1)) << 32);
        }
        for (uint32_t row_tile = 0; row_tile < owned_row_tiles; ++row_tile) {
            if (!row_tile_is_live<skip_padding>(live_mask, row_tile)) {
                continue;
            }
            // one row tile: worker_columns tiles wide, one block (waits for / pops CB_ROWS, reserves / pushes the
            // output, reconfigures unpack and pack to the two CBs' formats)
            compute_kernel_lib::tilize<worker_columns, CB_ROWS, tilize_output_cb>(1);
            if constexpr (has_shared_expert) {
                cb_wait_front(CB_TILIZED, worker_columns);
                cb_wait_front(CB_SHARED_EXPERT, worker_columns);
                cb_reserve_back(CB_OUTPUT, worker_columns);
                reconfig_data_format(CB_TILIZED, CB_SHARED_EXPERT);
                pack_reconfig_data_format(CB_OUTPUT);
                add_init(CB_TILIZED, CB_SHARED_EXPERT);
                tile_regs_acquire();
                for (uint32_t column = 0; column < worker_columns; ++column) {
                    add_tiles(CB_TILIZED, CB_SHARED_EXPERT, column, column, column);
                }
                tile_regs_commit();
                tile_regs_wait();
                for (uint32_t column = 0; column < worker_columns; ++column) {
                    pack_tile(column, CB_OUTPUT);
                }
                tile_regs_release();
                cb_push_back(CB_OUTPUT, worker_columns);
                cb_pop_front(CB_TILIZED, worker_columns);
                cb_pop_front(CB_SHARED_EXPERT, worker_columns);
            }
        }
        if constexpr (skip_padding) {
            cb_pop_front(CB_LIVE_ROW_TILES, 1);
        }
    }
}
