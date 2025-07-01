// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"

// HW-bcast scale for fused scale-attn-softmax
FORCE_INLINE void generate_inv_sqrt_hw_bcast_tile() {
    constexpr auto cb_fused_scale = tt::CBIndex::c_2;
    uint32_t u = get_arg_val<uint32_t>(1);
    cb_reserve_back(cb_fused_scale, 1);
    auto ptr = reinterpret_cast<uint16_t*>(get_write_ptr(cb_fused_scale));
    ptr[0] = u >> 16;
    cb_push_back(cb_fused_scale, 1);
}

void kernel_main() {
    constexpr uint32_t cb_reduce_scaler = tt::CBIndex::c_1;
    const uint32_t reduce_scaler = get_arg_val<uint32_t>(0);

    constexpr uint32_t block_wt = get_compile_time_arg_val(0);
    constexpr bool is_dram_mask = get_compile_time_arg_val(1) == 1;

    const uint32_t mask_addr = get_arg_val<uint32_t>(2);
    const uint32_t mask_start_tile_id = get_arg_val<uint32_t>(3);
    uint32_t mask_num_tiles = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_attn = tt::CBIndex::c_3;
    uint32_t mask_tile_bytes = get_tile_size(cb_attn);
    const DataFormat mask_data_format = get_dataformat(cb_attn);
    uint32_t mask_id = mask_start_tile_id;

    const InterleavedAddrGenFast<is_dram_mask> addr_mask = {
        .bank_base_address = mask_addr, .page_size = mask_tile_bytes, .data_format = mask_data_format};

    constexpr auto cb_fused_scale = tt::CBIndex::c_2;
    const uint32_t pre_scale = get_arg_val<uint32_t>(1);
    generate_bcast_unary_scalar(cb_fused_scale, pre_scale);

    constexpr uint32_t block_ht = get_compile_time_arg_val(4);
    for (uint32_t h = 0; h < block_ht; h++) {
        cb_reserve_back(cb_attn, block_wt);
        uint32_t l1_write_addr = get_write_ptr(cb_attn);
        for (uint32_t w = 0; w < block_wt; w++) {
            noc_async_read_tile(mask_id, addr_mask, l1_write_addr);
            l1_write_addr += mask_tile_bytes;
            ++mask_id;

            if (h == 0 && w == 0) {
                generate_reduce_scaler(cb_reduce_scaler, reduce_scaler);
            }
        }
        noc_async_read_barrier();

        cb_push_back(cb_attn, block_wt);
        if (mask_id == mask_num_tiles) {
            mask_id = 0;
        }
    }
}
