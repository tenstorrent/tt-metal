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
    const uint32_t num_tiles = get_arg(args::num_tiles);
    const uint32_t start_id = get_arg(args::start_id);
    constexpr uint32_t tiles_per_batch = get_arg(args::tiles_per_batch);

    using Auxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();

    auto tensor_accessor = TensorAccessor(tensor::src);

    const Noc noc;
    // dfb::in0 is the reduce input pipe: this kernel fills it, the compute kernel drains it.
    DataflowBuffer dfb_in0(dfb::in0);
    const uint32_t tile_bytes = dfb_in0.get_tile_size();

    // One barrier per batch; a short final batch uses the same path.
    const uint32_t end_id = start_id + num_tiles;
    for (uint32_t i = start_id; i < end_id;) {
        const uint32_t batch = std::min(tiles_per_batch, end_id - i);
        dfb_in0.reserve_back(batch);
        for (uint32_t k = 0; k < batch; ++k) {
            noc.async_read(tensor_accessor, dfb_in0, tile_bytes, {.page_id = i + k}, {.offset_bytes = k * tile_bytes});
        }
        noc.async_read_barrier();
        dfb_in0.push_back(batch);
        i += batch;
    }
}
