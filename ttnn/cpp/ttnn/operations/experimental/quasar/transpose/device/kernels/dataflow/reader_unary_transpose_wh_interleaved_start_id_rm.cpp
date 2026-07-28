// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/tensor/tensor_accessor.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    uint32_t start_id = get_arg(args::start_id);
    uint32_t num_hw_blocks_per_core = get_arg(args::num_hw_blocks);

    constexpr uint32_t Ht = get_arg(args::Ht);
    constexpr uint32_t H_per_tile = get_arg(args::H_per_tile);
    constexpr uint32_t H_per_tile_last = get_arg(args::H_per_tile_last);
    constexpr uint32_t Wt = get_arg(args::Wt);
    constexpr uint32_t W = get_arg(args::W);
    constexpr uint32_t HtWt = get_arg(args::HtWt);
    constexpr uint32_t W_size_bytes = get_arg(args::W_size_bytes);
    constexpr uint32_t l1_write_offset_bytes = get_arg(args::l1_write_offset_bytes);

    const uint32_t stick_size_bytes = W_size_bytes;

    const auto s = TensorAccessor(tensor::src);

    Noc noc;
    DataflowBuffer cb(dfb::cb_in0);

    uint32_t i_stick = start_id;

    // this reader will read a NHW tensor in NWH order
    // Uses tt::data_movement::common::noc_async_read_sharded to restore the multi-page
    // split that BLOCK/WIDTH-sharded RM buffers need (a logical row can span multiple
    // shards laterally). PR #42130 had replaced these helpers with the
    // single-NOC-transfer experimental::Noc::async_read primitive, which silently
    // dropped the split logic for BLOCK/WIDTH-sharded RM inputs (24+ test cases).
    for (uint32_t n = 0; n < num_hw_blocks_per_core; n++) {
        for (uint32_t h = 0; h < Ht; ++h) {
            cb.reserve_back(Wt);
            const uint32_t cb_write_ptr = cb.get_write_ptr();
            uint32_t l1_write_offset = 0;
            uint32_t H_curr = h == Ht - 1 ? H_per_tile_last : H_per_tile;
            for (uint32_t h_datum = 0; h_datum < H_curr; ++h_datum) {
                tt::data_movement::common::noc_async_read_sharded(
                    noc, cb_write_ptr + l1_write_offset, s, i_stick, 0, stick_size_bytes);
                l1_write_offset += l1_write_offset_bytes;
                i_stick += 1;
            }
            noc.async_read_barrier();
            cb.push_back(Wt);
        }
    }
}
