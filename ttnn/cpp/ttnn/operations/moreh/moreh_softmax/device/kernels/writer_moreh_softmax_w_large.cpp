// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    using MaxAuxiliary =
        ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::max_scaler>;
    using SumAuxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<
        ttnn::kernel_lib::ReduceAuxiliaryArgs<MaxAuxiliary::next_compile_time_args_offset()>,
        dfb::sum_scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<MaxAuxiliary>();
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<SumAuxiliary>();

    std::uint32_t N = get_arg(args::num_rows);
    std::uint32_t tile_offset = get_arg(args::tile_offset);
    std::uint32_t Wt = get_arg(args::Wt);

    constexpr auto dfb_id_out = dfb::out;
    constexpr std::uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    Noc noc;
    DataflowBuffer dfb_out_obj(dfb_id_out);
    const auto out_tile_bytes = dfb_out_obj.get_entry_size();

    std::uint32_t blk = 1;

    std::uint32_t tile_id = tile_offset;
    for (std::uint32_t i = 0; i < N; i++) {
        for (std::uint32_t w = 0; w < Wt; w++) {
            dfb_out_obj.wait_front(blk);
            noc.async_write(dfb_out_obj, s, out_tile_bytes, {.offset_bytes = 0}, {.page_id = tile_id});
            noc.async_write_barrier();
            dfb_out_obj.pop_front(blk);
            tile_id++;
        }
    }
}
