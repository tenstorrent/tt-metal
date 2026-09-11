// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

void kernel_main() {
    // Runtime args
    const std::uint32_t N = get_arg(args::num_rows);
    const std::uint32_t tile_offset = get_arg(args::tile_offset);
    const std::uint32_t Ht = get_arg(args::Ht);
    const std::uint32_t Wt = get_arg(args::Wt);

    // Constants
    constexpr auto dfb_in = dfb::in;
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;

    // Ublocks size defined in tiles
    constexpr std::uint32_t onetile = 1;

    // Input tensor
    const auto src_in = TensorAccessor(tensor::src);

    using MaxAuxiliary =
        ttnn::kernel_lib::BoundReduceAuxiliaryArgs<ttnn::kernel_lib::ReduceAuxiliaryArgs<0>, dfb_max_scaler>;
    using SumAuxiliary = ttnn::kernel_lib::BoundReduceAuxiliaryArgs<
        ttnn::kernel_lib::ReduceAuxiliaryArgs<MaxAuxiliary::next_compile_time_args_offset()>,
        dfb_sum_scaler>;
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<MaxAuxiliary>();
    dataflow_kernel_lib::prepare_reduce_auxiliary_tiles<SumAuxiliary>();

    Noc noc;
    DataflowBuffer dfb_in_obj(dfb_in);
    const auto in_tile_bytes = dfb_in_obj.get_entry_size();

    std::uint32_t curr_tile = tile_offset;
    for (std::uint32_t i = 0; i < N; i += onetile) {
        std::uint32_t w_idx = curr_tile % Wt;
        std::uint32_t nc_idx = curr_tile / Wt;
        std::uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;
        for (std::uint32_t h = 0; h < Ht; h++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += Wt;
        }

        w_idx = curr_tile % Wt;
        nc_idx = curr_tile / Wt;
        tile_idx = nc_idx * Ht * Wt + w_idx;
        for (std::uint32_t h = 0; h < Ht; h++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += Wt;
        }

        w_idx = curr_tile % Wt;
        nc_idx = curr_tile / Wt;
        tile_idx = nc_idx * Ht * Wt + w_idx;
        for (std::uint32_t h = 0; h < Ht; h++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            tile_idx += Wt;
        }

        curr_tile += 1;
    }
}
