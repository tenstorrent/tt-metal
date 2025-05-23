// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t ntiles_per_row = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);

    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_block_id = get_arg_val<uint32_t>(1);
    uint32_t num_blocks = get_arg_val<uint32_t>(2);

    constexpr bool src_is_dram = true;

    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    uint32_t end_block_id = start_block_id + num_blocks;
    for (uint32_t i = start_block_id; i < end_block_id; ++i) {
        cb_reserve_back(cb_id_in0, ntiles_per_row);
        uint64_t l1_write_addr = get_write_ptr(cb_id_in0);
        for (uint32_t j = 0; j < ntiles_per_row; ++j) {
            noc_async_read_tile(ntiles_per_row * i + j, s, l1_write_addr);
            l1_write_addr += tile_bytes;
        }
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, ntiles_per_row);
    }
}
