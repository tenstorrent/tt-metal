// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t input_num_blocks_w_per_core = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t input_num_blocks_h = get_arg_val<uint32_t>(3);
    uint32_t input_total_blocks_w = get_arg_val<uint32_t>(4);

    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t scaler = get_compile_time_arg_val(1);

    constexpr uint32_t cb_id_in2 = 2;
    generate_reduce_scaler(cb_id_in2, scaler);

    constexpr uint32_t cb_id_in0 = 0;

    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    // read a ublock of tiles from src to CB, and then push the ublock to unpacker
    for (uint32_t block_h_id = 0; block_h_id < input_num_blocks_h; block_h_id++) {
        uint32_t end_id = start_id + input_num_blocks_w_per_core;
        for (uint32_t i = start_id; i < end_id; i++) {
            cb_reserve_back(cb_id_in0, onetile);
            uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
            noc_async_read_tile((block_h_id * input_total_blocks_w) + i, s, l1_write_addr);
            noc_async_read_barrier();
            cb_push_back(cb_id_in0, onetile);
        }
    }
}
