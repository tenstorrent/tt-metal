// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/moreh_common.hpp"
void kernel_main() {
    ArgFetcher arg_fetcher;
    const uint32_t src0_addr = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t batch_num = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Wt = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t Wt_per_core = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t start_id = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t mask_h = arg_fetcher.get_next_arg_val<uint32_t>();
    const uint32_t mask_w = arg_fetcher.get_next_arg_val<uint32_t>();
    const bool do_mask_h = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);
    const bool do_mask_w = (arg_fetcher.get_next_arg_val<uint32_t>() == 1);

    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t scaler = get_compile_time_arg_val(1);
    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_scaler = 1;
    constexpr uint32_t cb_id_mask_h_w = 2;

    generate_reduce_scaler(cb_id_scaler, scaler);

    if (do_mask_h || do_mask_w) {
        generate_mask_h_w(cb_id_mask_h_w, mask_h, mask_w);
    }

    uint32_t l1_write_addr_in0;
    uint32_t src0_tile_bytes = get_tile_size(cb_id_in0);
    auto src0_data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src0_is_dram> s0 = {
        .bank_base_address = src0_addr, .page_size = src0_tile_bytes, .data_format = src0_data_format};

    constexpr uint32_t onetile = 1;
    for (uint32_t wt = 0; wt < Wt_per_core; ++wt) {
        uint32_t read_tile_id = start_id + wt;
        for (uint32_t b = 0; b < batch_num; ++b) {
            cb_reserve_back(cb_id_in0, onetile);
            l1_write_addr_in0 = get_write_ptr(cb_id_in0);
            noc_async_read_tile(read_tile_id, s0, l1_write_addr_in0);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
            read_tile_id += Wt;
        }
    }
}
