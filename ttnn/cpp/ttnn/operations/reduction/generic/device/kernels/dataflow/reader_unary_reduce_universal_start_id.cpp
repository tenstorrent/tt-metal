// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    auto num_tiles = get_arg(args::num_tiles);
    auto start_id = get_arg(args::start_id);
    constexpr auto scaler_bits = get_arg(args::scaler_bits);

    float scaler_f = __builtin_bit_cast(float, scaler_bits);
    dataflow_kernel_lib::prepare_reduce_scaler<dfb::scaler_dfb, REDUCE_OP, REDUCE_DIM>(scaler_f);

    constexpr uint32_t onetile = 1;
    DataflowBuffer cb_in0(dfb::in_dfb);
    uint32_t tile_bytes = cb_in0.get_tile_size();

    auto tensor_accessor = TensorAccessor(ta::input);

    Noc noc;

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_in0.reserve_back(onetile);
        noc.async_read(tensor_accessor, cb_in0, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        cb_in0.push_back(onetile);
    }
}
