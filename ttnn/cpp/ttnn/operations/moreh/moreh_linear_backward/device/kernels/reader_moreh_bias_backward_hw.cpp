// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"

void kernel_main() {
    ArgFetcher arg_fetcher;
    const uint32_t src_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t num_tiles = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t mask_h = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t mask_w = arg_fetcher.get_next_arg_val<uint32_t>();
    const bool do_mask_h = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    const bool do_mask_w = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_mask_h_w = 2;

    union {
        float f;
        uint32_t u;
    } scaler;
    scaler.f = 1.0f;
    fill_cb_with_value(cb_id_scaler, scaler.u);

    if (do_mask_h || do_mask_w) {
        generate_mask_h_w(cb_id_mask_h_w, mask_h, mask_w);
    }

    uint32_t l1_write_addr_in0;
    uint32_t src_tile_bytes = get_tile_size(cb_id_in0);
    auto src_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src_is_dram> s0 = {
        .bank_base_address = src_addr, .page_size = src_tile_bytes, .data_format = src_data_format};

    constexpr uint32_t onetile = 1;
    for (uint32_t i = start_id; i < start_id + num_tiles; i++) {
        cb_reserve_back(cb_id_in0, onetile);
        l1_write_addr_in0 = get_write_ptr(cb_id_in0);
        noc_async_read_tile(i, s0, l1_write_addr_in0);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, onetile);
    }
}
