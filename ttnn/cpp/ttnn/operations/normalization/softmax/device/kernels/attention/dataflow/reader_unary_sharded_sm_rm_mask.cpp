// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/kernel/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "experimental/noc.h"
#include "experimental/circular_buffer.h"
#include "experimental/tensor.h"
#include "experimental/endpoints.h"

void kernel_main() {
#if FUSED_SCALE_MASK
    constexpr uint32_t block_wt = get_compile_time_arg_val(0);
    constexpr auto mask_args = TensorAccessorArgs<1>();
    constexpr uint32_t size = get_compile_time_arg_val(mask_args.next_compile_time_args_offset());
    const uint32_t mask_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_start_tile_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_attn = tt::CBIndex::c_3;
    uint32_t mask_tile_bytes = get_tile_size(cb_attn);

    const auto addr_mask = TensorAccessor(mask_args, mask_addr, size);

    experimental::Noc noc;
    experimental::CircularBuffer cb_attn_obj(cb_attn);

    constexpr auto cb_fused_scale = tt::CBIndex::c_2;
    const uint32_t pre_scale = get_arg_val<uint32_t>(1);
    generate_bcast_unary_scalar(cb_fused_scale, pre_scale);

    constexpr uint32_t FLOAT32_DTYPE = get_compile_time_arg_val(mask_args.next_compile_time_args_offset() + 1);
    constexpr uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE ? 64 : 32;
    constexpr uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE ? 1024 : 512;

    cb_attn_obj.reserve_back(block_wt);
    uint32_t local_noc_x = my_x[noc.get_noc_id()];
    uint32_t local_noc_y = my_y[noc.get_noc_id()];
    uint32_t write_offset = 0;
    for (uint32_t w = 0; w < block_wt; w++) {
        noc.async_read(
            addr_mask,
            cb_attn_obj,
            mask_read_tile_face_bytes * 2,
            {.page_id = mask_start_tile_id + w},
            {.offset_bytes = write_offset});
        noc.async_read_barrier();
        uint32_t src_addr = cb_attn_obj.get_write_ptr() + write_offset + mask_read_tile_face_bytes;
        noc.async_read(
            experimental::UnicastEndpoint{},
            cb_attn_obj,
            mask_read_tile_face_bytes,
            {.noc_x = local_noc_x, .noc_y = local_noc_y, .addr = src_addr},
            {.offset_bytes = write_offset + mask_read_tile_offset_bytes});
        write_offset += mask_tile_bytes;
    }
    noc.async_read_barrier();
    cb_attn_obj.push_back(block_wt);
#endif

    {
        constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_1;
        const uint32_t reduce_scaler = get_arg_val<uint32_t>(0);
        generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
    }
}
