// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/tile_constants.h"
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "experimental/kernel_args.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

using namespace tt;

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t num_pairs = num_tiles >> 1;

    const uint32_t is_odd = num_tiles & 1;

    const auto output_addrg = TensorAccessor(tensor::output);

    Noc noc;
    DataflowBuffer dfb_dst(dfb::dst);
    const uint32_t dst_tile_bytes = dfb_dst.get_tile_size();

    for (uint32_t p = 0; p < num_pairs; p++) {
        uint32_t i = start_id + (p << 1);
        dfb_dst.wait_front(2);

        uint32_t dst_dfb_read_base = dfb_dst.get_read_ptr();
        uint32_t dst_dfb_read0_ptr = dst_dfb_read_base;
        uint32_t dst_dfb_read1_ptr = dst_dfb_read_base + dst_tile_bytes;

        noc.async_write(CoreLocalMem<uint32_t>(dst_dfb_read0_ptr), output_addrg, dst_tile_bytes, {}, {.page_id = i});
        noc.async_write(
            CoreLocalMem<uint32_t>(dst_dfb_read1_ptr), output_addrg, dst_tile_bytes, {}, {.page_id = i + 1});
        noc.async_write_barrier();
        dfb_dst.pop_front(2);
    }

    if (is_odd) {
        uint32_t i = start_id + (num_pairs << 1);
        dfb_dst.wait_front(1);

        uint32_t dst_dfb_read0_ptr = dfb_dst.get_read_ptr();

        noc.async_write(CoreLocalMem<uint32_t>(dst_dfb_read0_ptr), output_addrg, dst_tile_bytes, {}, {.page_id = i});
        noc.async_write_barrier();
        dfb_dst.pop_front(1);
    }
}
