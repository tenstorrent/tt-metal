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
#include "ttnn/cpp/ttnn/kernel_lib/reduce_plan_args.hpp"

void kernel_main() {
    uint32_t num_tiles = get_arg(args::num_tiles);
    uint32_t start_id = get_arg(args::start_id);

    using Auxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();

    constexpr uint32_t onetile = 1;

    auto tensor_accessor = TensorAccessor(tensor::src);

    Noc noc;
    // dfb::in0 is the reduce input pipe: this kernel fills it, the compute kernel drains it.
    DataflowBuffer dfb_in0(dfb::in0);
    uint32_t tile_bytes = dfb_in0.get_tile_size();

    // read a ublock of tiles from src to the buffer, and then push the ublock to unpacker
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        dfb_in0.reserve_back(onetile);
        noc.async_read(tensor_accessor, dfb_in0, tile_bytes, {.page_id = i}, {.offset_bytes = 0});
        noc.async_read_barrier();
        dfb_in0.push_back(onetile);
    }
}
