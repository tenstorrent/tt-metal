// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"

#include <cstdint>

void kernel_main() {
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;

    constexpr std::uint32_t block_wt = get_arg(args::block_w);

    const std::uint32_t mask_start_tile_id = get_arg(args::mask_start_tile_id);
    std::uint32_t mask_num_tiles = get_arg(args::mask_num_tiles);

    constexpr auto dfb_attn = dfb::fused_attn;
    DataflowBuffer dfb_attn_obj(dfb_attn);
    std::uint32_t mask_tile_bytes = dfb_attn_obj.get_entry_size();
    std::uint32_t mask_id = mask_start_tile_id;

    const auto addr_mask = TensorAccessor(tensor::mask);

    Noc noc;

    constexpr auto dfb_fused_scale = dfb::fused_scale;
    const std::uint32_t pre_scale = get_arg(args::pre_scale);
    generate_bcast_unary_scalar(CircularBuffer(dfb_fused_scale), pre_scale);

    constexpr std::uint32_t block_ht = get_arg(args::block_ht);
    for (std::uint32_t h = 0; h < block_ht; h++) {
        dfb_attn_obj.reserve_back(block_wt);
        std::uint32_t write_offset = 0;
        for (std::uint32_t w = 0; w < block_wt; w++) {
            noc.async_read(
                addr_mask, dfb_attn_obj, mask_tile_bytes, {.page_id = mask_id}, {.offset_bytes = write_offset});
            write_offset += mask_tile_bytes;
            ++mask_id;

            if (h == 0 && w == 0) {
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    dfb_max_scaler,
                    ckernel::PoolType::MAX,
                    ckernel::ReduceDim::REDUCE_ROW>();
                dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
                    dfb_sum_scaler,
                    ckernel::PoolType::SUM,
                    ckernel::ReduceDim::REDUCE_ROW>();
            }
        }
        noc.async_read_barrier();

        dfb_attn_obj.push_back(block_wt);
        if (mask_id == mask_num_tiles) {
            mask_id = 0;
        }
    }
}
