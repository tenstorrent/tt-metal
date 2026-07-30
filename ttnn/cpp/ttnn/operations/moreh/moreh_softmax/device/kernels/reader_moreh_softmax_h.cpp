// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/kernel/dataflow/moreh_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    // Runtime arguments
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t N = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t Ht = get_arg_val<uint32_t>(3);
    const uint32_t Wt = get_arg_val<uint32_t>(4);
    const uint32_t mask_h = get_arg_val<uint32_t>(5);

    // Constants
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_max_scaler = tt::CBIndex::c_2;
    constexpr auto cb_sum_scaler = tt::CBIndex::c_3;

    // Ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src_in_tile_bytes = get_tile_size(cb_in);

    // Input tensor
    constexpr bool is_fp32 = get_compile_time_arg_val(0) == 1;
    constexpr auto in_args = TensorAccessorArgs<1>();
    const auto src_in = TensorAccessor(in_args, src_addr);

    // Generate scaler tiles: MAX needs row-0 fill (reduce LLK), SUM needs col-0 fill (matmul)
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_max_scaler, ckernel::PoolType::MAX, ckernel::ReduceDim::REDUCE_COL>();
    dataflow_kernel_lib::
        calculate_and_prepare_reduce_scaler<cb_sum_scaler, ckernel::PoolType::SUM, ckernel::ReduceDim::REDUCE_COL>();

    // Generate mask tile
    DataflowBuffer dfb_mask_obj(cb_mask);
    if (is_fp32) {
        generate_mask_h<uint32_t>(dfb_mask_obj, mask_h);
    } else {
        generate_mask_h<uint16_t>(dfb_mask_obj, mask_h);
    }

    Noc noc;
    DataflowBuffer dfb_in_obj(cb_in);

    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i += onetile) {
        uint32_t w_idx = curr_tile % Wt;
        uint32_t nc_idx = curr_tile / Wt;
        uint32_t tile_idx = nc_idx * Ht * Wt + w_idx;
        dfb_in_obj.reserve_back(Ht);
        for (uint32_t h = 0; h < Ht; h++) {
            noc.async_read(
                src_in, dfb_in_obj, src_in_tile_bytes, {.page_id = tile_idx}, {.offset_bytes = h * src_in_tile_bytes});
            tile_idx += Wt;
        }
        noc.async_read_barrier();
        dfb_in_obj.push_back(Ht);
        curr_tile += 1;
    }
}
