// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// #include "debug/dprint.h"
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

void kernel_main() {

    #if FUSED_SCALE_MASK
    constexpr uint32_t block_wt = get_compile_time_arg_val(0);
    constexpr bool is_dram_mask = get_compile_time_arg_val(1) == 1;
    const uint32_t mask_addr  = get_arg_val<uint32_t>(2);
    const uint32_t mask_start_tile_id  = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_attn = tt::CB::c_in3;
    uint32_t mask_tile_bytes = get_tile_size(cb_attn);

    #define stick_size_is_pow2 get_compile_time_arg_val(2) == 1
    #if (stick_size_is_pow2)
    constexpr uint32_t log_base_2_of_page_size = get_compile_time_arg_val(3);
    #else
    constexpr uint32_t page_size = get_compile_time_arg_val(3);
    #endif
    #if (stick_size_is_pow2)
    const InterleavedPow2AddrGen<is_dram_mask> addr_mask = {
        .bank_base_address = mask_addr,
        .log_base_2_of_page_size = log_base_2_of_page_size
    };
    #else
    const InterleavedAddrGen<is_dram_mask> addr_mask = {
        .bank_base_address = mask_addr,
        .page_size = page_size
    };
    #endif

    constexpr auto cb_fused_scale = tt::CB::c_in2;
    const uint32_t pre_scale = get_arg_val<uint32_t>(1);
    generate_bcast_unary_scalar(cb_fused_scale, pre_scale);

    constexpr uint32_t FLOAT32_DTYPE = get_compile_time_arg_val(4);
    uint32_t mask_read_tile_face_bytes = FLOAT32_DTYPE ? 64 : 32;
    uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE ? 1024 : 512;

    cb_reserve_back(cb_attn, block_wt);
    uint32_t l1_write_addr = get_write_ptr(cb_attn);
    for (uint32_t w = 0; w<block_wt; w++) {
        uint64_t mask_noc_addr = get_noc_addr(mask_start_tile_id + w, addr_mask);
        noc_async_read(mask_noc_addr, l1_write_addr, mask_read_tile_face_bytes);
        mask_noc_addr += mask_read_tile_face_bytes;
        noc_async_read(mask_noc_addr, l1_write_addr + mask_read_tile_offset_bytes, mask_read_tile_face_bytes);
        l1_write_addr += mask_tile_bytes;
    }
    noc_async_read_barrier();
    cb_push_back(cb_attn, block_wt);
    #endif

    {
        constexpr uint32_t cb_reduce_scaler = tt::CB::c_in1;
        const uint32_t reduce_scaler = get_arg_val<uint32_t>(0);
        generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
    }
}
