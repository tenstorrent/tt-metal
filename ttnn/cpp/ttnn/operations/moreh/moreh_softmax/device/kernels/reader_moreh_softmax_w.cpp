// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

#include <cstdint>

void kernel_main() {
    // Runtime args
    const uint32_t src_addr = get_arg_val<uint32_t>(0);
    const uint32_t N = get_arg_val<uint32_t>(1);
    const uint32_t tile_offset = get_arg_val<uint32_t>(2);
    const uint32_t Wt = get_arg_val<uint32_t>(3);
    const uint32_t scaler = get_arg_val<uint32_t>(4);
    const uint32_t mask_w = get_arg_val<uint32_t>(5);

    // Constants
    constexpr auto cb_in = tt::CBIndex::c_0;
    constexpr auto cb_mask = tt::CBIndex::c_1;
    constexpr auto cb_scaler = tt::CBIndex::c_2;

    // Ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t src_in_tile_bytes = get_tile_size(cb_in);

    // Input tensor
    constexpr bool is_fp32 = get_compile_time_arg_val(0) == 1;
    constexpr auto in_args = TensorAccessorArgs<1>();
    const auto src_in = TensorAccessor(in_args, src_addr, src_in_tile_bytes);

    // Generate scaler and mask tiles
    if (is_fp32) {
        generate_bcast_scaler<uint32_t>(cb_scaler, scaler);
        generate_mask_w<uint32_t>(cb_mask, mask_w);
    } else {
        generate_bcast_scaler<uint16_t>(cb_scaler, scaler);
        generate_mask_w<uint16_t>(cb_mask, mask_w);
    }

    // Read ublocks from src0 to CB0, then push ublocks to compute kernel
    uint32_t l1_write_addr_in = 0;
    uint32_t curr_tile = tile_offset;
    for (uint32_t i = 0; i < N; i += onetile) {
        cb_reserve_back(cb_in, Wt);
        l1_write_addr_in = get_write_ptr(cb_in);
        for (uint32_t w = 0; w < Wt; w++) {
            noc_async_read_tile(curr_tile, src_in, l1_write_addr_in);
            l1_write_addr_in += src_in_tile_bytes;
            curr_tile++;
        }
        noc_async_read_barrier();
        cb_push_back(cb_in, Wt);
    }
}
