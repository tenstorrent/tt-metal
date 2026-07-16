// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t N = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);
    const uint32_t mask_w = get_arg_val<uint32_t>(4);

    // Constants
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;

    // Ublocks size defined in tiles
    constexpr uint32_t onetile = 1;

    // Input tensor
    constexpr bool is_fp32 = get_compile_time_arg_val(0) == 1;
    constexpr auto in_args = TensorAccessorArgs<1>();
    const auto src_in = TensorAccessor(in_args, src_addr);

    // Generate scaler tiles: MAX needs row-0 fill (reduce LLK), SUM needs col-0 fill (matmul)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_ROW>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_ROW>();

    // Generate mask tile
    DataflowBuffer dfb_mask_obj(cb_mask);
    if (is_fp32) {
        generate_mask_w<uint32_t>(dfb_mask_obj, mask_w);
    } else {
        generate_mask_w<uint16_t>(dfb_mask_obj, mask_w);
    }

    Noc noc;
    DataflowBuffer dfb_in_obj(cb_in);
    const auto in_tile_bytes = get_tile_size(cb_in);

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i += onetile) {
        uint32_t curr_offset_i = curr_tile;
        for (uint32_t w = 0; w < Wt; w++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            curr_tile++;
        }

        curr_tile = curr_offset_i;
        for (uint32_t w = 0; w < Wt; w++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            curr_tile++;
        }

        curr_tile = curr_offset_i;
        for (uint32_t w = 0; w < Wt; w++) {
            dfb_in_obj.reserve_back(onetile);
            noc.async_read(src_in, dfb_in_obj, in_tile_bytes, {.page_id = curr_tile}, {.offset_bytes = 0});
            noc.async_read_barrier();
            dfb_in_obj.push_back(onetile);
            curr_tile++;
        }
    }
}
