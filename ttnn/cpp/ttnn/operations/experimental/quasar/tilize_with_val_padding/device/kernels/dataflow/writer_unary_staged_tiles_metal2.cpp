// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc.h"
#include "api/kernel_thread_globals.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

// Writes tilize's output tiles to the interleaved output with implicit sync. Output entry k is tile
// k / num_lanes of lane k % num_lanes (the strided DFB interleaves the lanes' tile counters), and thread
// t of N drains the entries t, t + N, .... With COLUMN_LANES a lane's tiles are its lane_tiles-wide
// column slice of each block; otherwise they are whole blocks.
void kernel_main() {
    constexpr uint32_t tiles_per_row = get_arg(args::tiles_per_row);
    constexpr uint32_t lane_tiles = get_arg(args::lane_tiles);
    constexpr uint32_t num_lanes = get_arg(args::num_lanes);

    const uint32_t first_block = get_arg(args::first_block);
    const uint32_t num_lane_blocks = get_arg(args::num_lane_blocks);

    Noc noc;
    DataflowBuffer cb_out(dfb::out);
    const auto s = TensorAccessor(tensor::output);

    for (uint32_t k = get_my_thread_id(); k < num_lanes * num_lane_blocks * lane_tiles; k += get_num_threads()) {
        const uint32_t lane = k % num_lanes;
        const uint32_t q = k / num_lanes;
        const uint32_t m = q / lane_tiles;
        const uint32_t block = COLUMN_LANES ? first_block + m : first_block + lane + num_lanes * m;
        const uint32_t tile = block * tiles_per_row + (COLUMN_LANES ? lane * lane_tiles : 0) + q % lane_tiles;
        noc.async_write<NocOptions::TXN_ID>(cb_out, s, {}, {.page_id = tile});
    }
}
