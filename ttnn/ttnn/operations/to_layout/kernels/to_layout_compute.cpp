// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

// to_layout compute: tilize_block / untilize_block (op_design.md, Block schedule).
//
// One compute_kernel_hw_startup and ONE helper call per kernel over all of this NEO's tile-rows, so the
// helper's init / reconfig / uninit happen once per kernel; per tile-row the helper waits bw entries,
// converts the whole bw-tile row, pushes bw and pops bw.
//   regime TILIZE:   staged_input = cb_row_block (32 rows x bw tiles) -> staged_output = cb_tile_block
//   regime UNTILIZE: staged_input = cb_tile_block_in (bw tiles) -> staged_output = cb_row_block_out
// bfloat16 through the 16-bit DEST: unpack -> datacopy -> pack, bit-exact for the values moved.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/kernel_thread_globals.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"

namespace ckl = compute_kernel_lib;

namespace {
constexpr uint32_t REGIME_TILIZE = 0;
constexpr uint32_t REGIME_UNTILIZE = 1;
}  // namespace

void kernel_main() {
    // This NEO's tile-rows. With C = 1 (the plan) a NEO holds the whole cluster's share.
    const uint32_t num_tile_rows = get_arg(args::num_tile_rows);
    constexpr uint32_t regime = get_arg(args::regime);
    static_assert(regime == REGIME_TILIZE || regime == REGIME_UNTILIZE, "to_layout_compute: unknown regime");
    constexpr uint32_t bw = get_arg(args::block_width_tiles);
    constexpr uint32_t cb_in = dfb::staged_input;
    constexpr uint32_t cb_out = dfb::staged_output;

    DataflowBuffer staged_input(dfb::staged_input);
    DataflowBuffer staged_output(dfb::staged_output);
    compute_kernel_hw_startup(dfb::staged_input, dfb::staged_output);

    if constexpr (regime == REGIME_TILIZE) {
        ckl::tilize<bw, cb_in, cb_out>(staged_input, staged_output, num_tile_rows);
    } else {
        ckl::untilize<bw, cb_in, cb_out>(staged_input, staged_output, num_tile_rows);
    }
}
