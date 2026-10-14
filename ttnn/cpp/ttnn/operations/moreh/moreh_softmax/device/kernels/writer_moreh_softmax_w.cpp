// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"

void kernel_main() {
    using MaxAuxiliary =
        ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb::max_scaler>;
    using SumAuxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<
        ttnn::kernel_lib::ReduceAuxiliaryArgs<MaxAuxiliary::next_compile_time_args_offset()>,
        dfb::sum_scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<MaxAuxiliary>();
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<SumAuxiliary>();

    const std::uint32_t N = get_arg(args::num_rows);
    const std::uint32_t tile_offset = get_arg(args::tile_offset);
    const std::uint32_t Wt = get_arg(args::Wt);

    constexpr auto dfb_id_out = dfb::out;
    constexpr std::uint32_t onetile = 1;

    const auto s = TensorAccessor(tensor::dst);

    const Noc noc;
    DataflowBuffer dfb_out_obj(dfb_id_out);
    const std::uint32_t tile_bytes = dfb_out_obj.get_entry_size();

    std::uint32_t tile_id = tile_offset;
    for (std::uint32_t i = 0; i < N; i++) {
        dfb_out_obj.wait_front(static_cast<uint16_t>(Wt));
        for (std::uint32_t w = 0; w < Wt; w++) {
            noc.async_write(dfb_out_obj, s, tile_bytes, {.offset_bytes = w * tile_bytes}, {.page_id = tile_id});
            tile_id++;
        }
        noc.async_write_barrier();
        dfb_out_obj.pop_front(static_cast<uint16_t>(Wt));
    }
}
