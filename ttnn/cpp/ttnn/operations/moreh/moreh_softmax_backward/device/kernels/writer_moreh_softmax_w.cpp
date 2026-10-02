// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    using Auxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();

    auto N = get_arg(args::N);
    auto tile_offset = get_arg(args::tile_offset);
    auto Wt = get_arg(args::Wt);

    constexpr uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dx);

    uint32_t blk = 1;

    Noc noc;
    DataflowBuffer dfb_out_obj(dfb::out);
    const auto out_tile_bytes = dfb_out_obj.get_tile_size();

    uint32_t tile_id = tile_offset;
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t w = 0; w < Wt; w++) {
            dfb_out_obj.wait_front(blk);
            noc.async_write(dfb_out_obj, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_write_barrier();
            dfb_out_obj.pop_front(blk);
            tile_id++;
        }
    }
}
