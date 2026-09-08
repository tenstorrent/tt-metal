// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    const uint32_t batch_num = get_arg(args::batch_num);
    const uint32_t Wt = get_arg(args::Wt);
    const uint32_t Wt_per_core = get_arg(args::Wt_per_core);
    const uint32_t start_id = get_arg(args::start_id);
    using Auxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<Auxiliary>();

    const auto s0 = TensorAccessor(tensor::src0);

    Noc noc;
    DataflowBuffer dfb_in0(dfb::in0);
    const auto in0_tile_bytes = dfb_in0.get_tile_size();

    constexpr uint32_t onetile = 1;
    for (uint32_t wt = 0; wt < Wt_per_core; ++wt) {
        uint32_t read_tile_id = start_id + wt;
        for (uint32_t b = 0; b < batch_num; ++b) {
            dfb_in0.reserve_back(onetile);
            noc.async_read(s0, dfb_in0, in0_tile_bytes, {.page_id = read_tile_id}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in0.push_back(onetile);
            read_tile_id += Wt;
        }
    }
}
