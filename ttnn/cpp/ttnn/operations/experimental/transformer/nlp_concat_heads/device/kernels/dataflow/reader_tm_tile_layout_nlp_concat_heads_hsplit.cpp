// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Head-split variant of reader_tm_tile_layout_nlp_concat_heads.cpp (head_split=true, interleaved in/out):
// the unit of work is one (tile row, head) pair instead of one tile row, so short sequences use more cores. Unit u
// (global row R = u / in0_c, head h = u % in0_c) reads the in0_w_tiles tiles of head h at row R and pushes them in
// order; the output tiles of unit u are u * in0_w_tiles .. + in0_w_tiles - 1, so a core's units form one contiguous
// output range (written by the unchanged interleaved writer). All tiles of a unit are read with one barrier.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    Noc noc;

    uint32_t num_units = get_arg(args::num_units);
    uint32_t unit_start = get_arg(args::unit_start);

    constexpr uint32_t in0_h_tiles = get_arg(args::in0_h_tiles);
    constexpr uint32_t in0_w_tiles = get_arg(args::in0_w_tiles);
    constexpr uint32_t in0_c = get_arg(args::in0_c);
    constexpr uint32_t in0_HtWt = get_arg(args::in0_HtWt);
    constexpr uint32_t in0_CHtWt = in0_c * in0_HtWt;

    DataflowBuffer dfb_in0(dfb::in0);
    const uint32_t single_tile_size_bytes = dfb_in0.get_entry_size();
    const auto s0 = TensorAccessor(tensor::src);

    for (uint32_t u = unit_start; u < unit_start + num_units; u++) {
        const uint32_t R = u / in0_c;
        const uint32_t h = u - R * in0_c;
        const uint32_t b = R / in0_h_tiles;
        const uint32_t rr = R - b * in0_h_tiles;
        uint32_t tile_id = b * in0_CHtWt + h * in0_HtWt + rr * in0_w_tiles;

        dfb_in0.reserve_back(in0_w_tiles);
        uint32_t l1_write_addr = dfb_in0.get_write_ptr();
        for (uint32_t w = 0; w < in0_w_tiles; w++) {
            noc.async_read(
                s0, CoreLocalMem<uint32_t>(l1_write_addr), single_tile_size_bytes, {.page_id = tile_id + w}, {});
            l1_write_addr += single_tile_size_bytes;
        }
        noc.async_read_barrier();
        dfb_in0.push_back(in0_w_tiles);
    }
}
