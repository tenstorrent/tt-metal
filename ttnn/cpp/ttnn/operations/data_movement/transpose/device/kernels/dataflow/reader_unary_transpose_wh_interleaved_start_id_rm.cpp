// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks_per_core);

    constexpr auto Ht = get_arg(args::Ht);
    constexpr auto H_per_tile = get_arg(args::H_per_tile);
    constexpr auto H_per_tile_last = get_arg(args::H_per_tile_last);
    constexpr auto Wt = get_arg(args::Wt);
    constexpr auto W_size_bytes = get_arg(args::W_size_bytes);
    constexpr auto l1_write_offset_bytes = get_arg(args::l1_write_offset_bytes);

    const uint32_t stick_size_bytes = W_size_bytes;

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer dfb(dfb::in0);

    uint32_t i_stick = start_id;

    // this reader will read a NHW tensor in NWH order
    // Uses tt::data_movement::common::noc_async_read_sharded to restore the multi-page
    // split that BLOCK/WIDTH-sharded RM buffers need (a logical row can span multiple
    // shards laterally). PR #42130 had replaced these helpers with the
    // single-NOC-transfer experimental::Noc::async_read primitive, which silently
    // dropped the split logic for BLOCK/WIDTH-sharded RM inputs (24+ test cases).
    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            dfb.reserve_back(Wt);
            const uint32_t dfb_write_ptr = dfb.get_write_ptr();
            uint32_t l1_write_offset = 0;
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                tt::data_movement::common::noc_async_read_sharded(
                    noc, dfb_write_ptr + l1_write_offset, s, i_stick, 0, stick_size_bytes);
                l1_write_offset += l1_write_offset_bytes;
                i_stick += 1;
            }
            noc.async_read_barrier();
            dfb.push_back(Wt);
        }
    }
}
