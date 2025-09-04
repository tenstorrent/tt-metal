// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t N = get_arg_val<uint32_t>(1);
    uint32_t C = get_arg_val<uint32_t>(2);
    uint32_t HtWt = get_arg_val<uint32_t>(3);
    uint32_t batch_step = get_arg_val<uint32_t>(4);    // CHtWt - HtWt
    uint32_t channel_step = get_arg_val<uint32_t>(5);  // NCHtWt - HtWt

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    uint32_t i = 0;
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t n = 0; n < N; n++) {
            for (uint32_t hw = 0; hw < HtWt; hw++) {
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read_tile(i, s, l1_write_addr);
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
                i++;
            }
            i += batch_step;
        }
        i -= channel_step;
    }
}
