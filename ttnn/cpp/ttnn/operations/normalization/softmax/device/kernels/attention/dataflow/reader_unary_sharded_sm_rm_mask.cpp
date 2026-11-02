// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "mask_padded_tile.hpp"

#include <cstdint>

void kernel_main() {
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;

#ifdef MASK_PADDED_DATA
    // The last dimension is not a multiple of the tile width: the compute adds this -inf tile to the last tile of
    // every row so the padding columns stay out of the row max and sum.
    constexpr std::uint32_t num_datum_padded = get_arg(args::num_datum_padded);
    constexpr std::uint32_t tile_hw = get_arg(args::tile_hw);
    DataflowBuffer dfb_mask_padded_obj(dfb::mask_padded);
    push_mask_padded_tile<num_datum_padded, tile_hw>(dfb_mask_padded_obj);
#endif

#ifdef FUSED_SCALE_MASK
    constexpr std::uint32_t block_wt = get_arg(args::block_w);
    const std::uint32_t mask_start_tile_id = get_arg(args::mask_start_tile_id);

    constexpr auto dfb_attn = dfb::fused_attn;
    DataflowBuffer dfb_attn_obj(dfb_attn);
    std::uint32_t mask_tile_bytes = dfb_attn_obj.get_entry_size();

    const auto addr_mask = TensorAccessor(tensor::mask);

    Noc noc;

    constexpr auto dfb_fused_scale = dfb::fused_scale;
    const std::uint32_t pre_scale = get_arg(args::pre_scale);
    generate_bcast_unary_scalar(CircularBuffer(dfb_fused_scale), pre_scale);

    constexpr std::uint32_t FLOAT32_DTYPE = get_arg(args::mask_float32);
    constexpr std::uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE ? 64 : 32;
    constexpr std::uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE ? 1024 : 512;

    dfb_attn_obj.reserve_back(block_wt);
    std::uint32_t local_noc_x = my_x[noc.get_noc_id()];
    std::uint32_t local_noc_y = my_y[noc.get_noc_id()];
    std::uint32_t write_offset = 0;
    for (std::uint32_t w = 0; w < block_wt; w++) {
        noc.async_read(
            addr_mask,
            dfb_attn_obj,
            mask_read_tile_face_bytes * 2,
            {.page_id = mask_start_tile_id + w},
            {.offset_bytes = write_offset});
        noc.async_read_barrier();
        std::uint32_t src_addr = dfb_attn_obj.get_write_ptr() + write_offset + mask_read_tile_face_bytes;
        noc.async_read(
            UnicastEndpoint{},
            dfb_attn_obj,
            mask_read_tile_face_bytes,
            {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = src_addr},
            {.offset_bytes = write_offset + mask_read_tile_offset_bytes});
        write_offset += mask_tile_bytes;
    }
    noc.async_read_barrier();
    dfb_attn_obj.push_back(block_wt);
#endif

    {
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
