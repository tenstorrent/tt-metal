// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar_metal2.hpp"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/tensor/noc_traits.h"
#include "api/dataflow/endpoints.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/kernel_helper_functions/local_l1_copy.hpp"

#include <cstdint>

void kernel_main() {
    constexpr auto dfb_max_scaler = dfb::max_scaler;
    constexpr auto dfb_sum_scaler = dfb::sum_scaler;

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
    DataflowBuffer dfb_fused_scale_obj(dfb_fused_scale);
    generate_bcast_unary_scalar(dfb_fused_scale_obj, pre_scale);

    constexpr std::uint32_t FLOAT32_DTYPE = get_arg(args::mask_float32);
    constexpr std::uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE ? 64 : 32;
    constexpr std::uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE ? 1024 : 512;

    dfb_attn_obj.reserve_back(block_wt);
#ifndef ARCH_QUASAR
    std::uint32_t local_noc_x = my_x[noc.get_noc_id()];  // Gen1 loopback source; Quasar copies with the RISC below
    std::uint32_t local_noc_y = my_y[noc.get_noc_id()];
#endif
    std::uint32_t write_offset = 0;
    for (std::uint32_t w = 0; w < block_wt; w++) {
        noc.async_read(
            addr_mask,
            dfb_attn_obj,
            mask_read_tile_face_bytes * 2,
            {.page_id = mask_start_tile_id + w},
            {.offset_bytes = write_offset});
        noc.async_read_barrier();
#ifdef ARCH_QUASAR
        // Relocate the second half-row (cols 16..31) into face 1 with a scalar copy: the bytes are already
        // resident from the barriered read above, and a NoC self-loopback (src coords == dst coords) spins
        // on can_post or silently drops on Quasar (see local_l1_copy).
        {
            const std::uint32_t base = dfb_attn_obj.get_write_ptr() + write_offset;
            local_l1_copy(
                base + mask_read_tile_offset_bytes, base + mask_read_tile_face_bytes, mask_read_tile_face_bytes);
        }
#else
        std::uint32_t src_addr = dfb_attn_obj.get_write_ptr() + write_offset + mask_read_tile_face_bytes;
        noc.async_read(
            UnicastEndpoint{},
            dfb_attn_obj,
            mask_read_tile_face_bytes,
            {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = src_addr},
            {.offset_bytes = write_offset + mask_read_tile_offset_bytes});
#endif
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
