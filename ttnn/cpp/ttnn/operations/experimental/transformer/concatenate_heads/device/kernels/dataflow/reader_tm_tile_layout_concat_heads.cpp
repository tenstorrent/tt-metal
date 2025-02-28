// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // READER RUNTIME ARGS
    uint32_t in0_tensor_addr = get_arg_val<uint32_t>(0);
    uint32_t in0_tensor_tile_id = get_arg_val<uint32_t>(1);

    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t in0_is_dram = get_compile_time_arg_val(1);
    // READER COMPILE TIME ARGS
    constexpr uint32_t in0_w_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t in0_c = get_compile_time_arg_val(3);
    constexpr uint32_t in0_HtWt = get_compile_time_arg_val(4);

    constexpr uint32_t cb_id_in0 = 0;
    uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    constexpr bool in0_is_dram_bool = in0_is_dram == 1;
    constexpr bool tile_dtype_is_bfloat16 = get_compile_time_arg_val(0) == 1;

    DataFormat data_format = DataFormat::Invalid;
    if constexpr (tile_dtype_is_bfloat16) {
        data_format = DataFormat::Float16;
    } else {
        data_format = DataFormat::Bfp8_b;
    }
    const InterleavedAddrGenFast<in0_is_dram_bool> s0 = {
        .bank_base_address = in0_tensor_addr, .page_size = single_tile_size_bytes, .data_format = data_format};

    uint32_t l1_write_addr_in0 = get_write_ptr(cb_id_in0);
    uint32_t in0_tensor_current_tile_id = in0_tensor_tile_id;

    for (uint32_t c_dim = 0; c_dim < in0_c; c_dim++) {
        cb_reserve_back(cb_id_in0, in0_w_tiles);

        in0_tensor_current_tile_id = in0_tensor_tile_id;
        for (uint32_t w_dim = 0; w_dim < in0_w_tiles; w_dim++) {
            noc_async_read_tile(in0_tensor_current_tile_id, s0, l1_write_addr_in0);
            l1_write_addr_in0 += single_tile_size_bytes;
            in0_tensor_current_tile_id++;
        }
        in0_tensor_tile_id += in0_HtWt;
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, in0_w_tiles);
    }
}
