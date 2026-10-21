// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// transpose_wh over this core's K head tiles; the tile count follows the reader's row split, derived
// from the static core index and the common sequence length instead of a per-core compile-time arg.

#include <cstdint>

#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/transpose.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

void kernel_main() {
    compute_kernel_hw_startup(dfb::in, dfb::out);
    transpose_init(dfb::in);

    const uint32_t core_idx = get_arg(args::core_idx);
    const uint32_t h_tiles = get_arg(args::h_tiles);
    constexpr uint32_t kv_num_tiles = get_arg(args::kv_num_tiles);
    constexpr uint32_t batch = get_arg(args::batch);
    constexpr uint32_t num_cores = get_arg(args::num_cores);

    const uint32_t total_rows = batch * h_tiles;
    const uint32_t num_blocks = ttnn::runtime_length::balanced_end(core_idx, total_rows, num_cores) -
                                ttnn::runtime_length::balanced_start(core_idx, total_rows, num_cores);
    const uint32_t NHtWt = num_blocks * kv_num_tiles;

    DataflowBuffer dfb_in(dfb::in);
    DataflowBuffer dfb_out(dfb::out);

    for (uint32_t n = 0; n < NHtWt; n++) {
        dfb_in.wait_front(1);
        dfb_out.reserve_back(1);

        tile_regs_acquire();
        transpose_tile(dfb::in, 0, 0);
        tile_regs_commit();
        tile_regs_wait();
        pack_tile(0, dfb::out);
        tile_regs_release();

        dfb_out.push_back(1);
        dfb_in.pop_front(1);
    }
}
