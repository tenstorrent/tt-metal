// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {
#if FUSED_SCALE_MASK
    constexpr uint32_t block_wt = get_compile_time_arg_val(0);
    constexpr bool is_dram_mask = get_compile_time_arg_val(1) == 1;
    const uint32_t mask_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_start_tile_id = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_attn = tt::CBIndex::c_3;
    uint32_t mask_tile_bytes = get_tile_size(cb_attn);

    constexpr bool stick_size_is_pow2 = get_compile_time_arg_val(2) == 1;
    constexpr uint32_t size = get_compile_time_arg_val(3);

    const auto addr_mask = get_interleaved_addr_gen<is_dram_mask, stick_size_is_pow2>(mask_addr, size);

    constexpr auto cb_fused_scale = tt::CBIndex::c_2;
    const uint32_t pre_scale = get_arg_val<uint32_t>(1);
    generate_bcast_unary_scalar(cb_fused_scale, pre_scale);

    constexpr uint32_t FLOAT32_DTYPE = get_compile_time_arg_val(4);
    constexpr uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE ? 64 : 32;
    constexpr uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE ? 1024 : 512;

    cb_reserve_back(cb_attn, block_wt);
    uint32_t l1_write_addr = get_write_ptr(cb_attn);
    for (uint32_t w = 0; w < block_wt; w++) {
        uint64_t mask_noc_addr = get_noc_addr(mask_start_tile_id + w, addr_mask);
        noc_async_read(mask_noc_addr, l1_write_addr, mask_read_tile_face_bytes * 2);
        mask_noc_addr = get_noc_addr(l1_write_addr + mask_read_tile_face_bytes);
        noc_async_read_barrier();
        noc_async_read(mask_noc_addr, l1_write_addr + mask_read_tile_offset_bytes, mask_read_tile_face_bytes);
        l1_write_addr += mask_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_attn, block_wt);
#endif

    {
        constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_1;
        const uint32_t reduce_scaler = get_arg_val<uint32_t>(0);
        generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
    }
}
