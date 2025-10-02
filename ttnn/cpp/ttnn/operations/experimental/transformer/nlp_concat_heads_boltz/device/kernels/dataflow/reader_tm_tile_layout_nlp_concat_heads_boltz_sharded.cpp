// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // WRITER RUNTIME ARGS
    uint32_t nheads = get_arg_val<uint32_t>(0);                    // This is per core per risc
    uint32_t start_read_offset_bytes = get_arg_val<uint32_t>(1);   // offset by nheads * in0_HtWt
    uint32_t start_write_offset_bytes = get_arg_val<uint32_t>(2);  // offset by nheads * in0_Wt

    // DEBUG: Print all runtime arg_vals
    DPRINT << "=== RUNTIME ARG_VALS ===" << ENDL();
    DPRINT << "arg_val[0] (nheads): " << nheads << ENDL();
    DPRINT << "arg_val[1] (start_read_offset_bytes): " << start_read_offset_bytes << ENDL();
    DPRINT << "arg_val[2] (start_write_offset_bytes): " << start_write_offset_bytes << ENDL();
    // COMPILE TIME ARGS
    // interleaved accessor args
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out0 = get_compile_time_arg_val(1);
    constexpr uint32_t in0_h_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t head_dim_size_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t out_row_size_bytes =
        get_compile_time_arg_val(4);  // total nheads per core * in0_w_tiles * single_tile_size_bytes
    constexpr uint32_t block_size = get_compile_time_arg_val(5);  // total nheads per core * in0_HtWt

    const uint32_t single_tile_size_bytes = get_tile_size(cb_id_in0);

    // DEBUG: Print all compile-time args
    DPRINT << "=== COMPILE TIME ARGS ===" << ENDL();
    DPRINT << "compile_time_arg[0] (cb_id_in0): " << cb_id_in0 << ENDL();
    DPRINT << "compile_time_arg[1] (cb_id_out0): " << cb_id_out0 << ENDL();
    DPRINT << "compile_time_arg[2] (in0_h_tiles): " << in0_h_tiles << ENDL();
    DPRINT << "compile_time_arg[3] (head_dim_size_bytes): " << head_dim_size_bytes << ENDL();
    DPRINT << "compile_time_arg[4] (out_row_size_bytes): " << out_row_size_bytes << ENDL();
    DPRINT << "compile_time_arg[5] (block_size): " << block_size << ENDL();
    DPRINT << "single_tile_size_bytes: " << single_tile_size_bytes << ENDL();

    cb_reserve_back(cb_id_in0, block_size);  // Redundant
    cb_reserve_back(cb_id_out0, block_size);

    uint64_t noc_l1_read_addr = get_noc_addr(get_read_ptr(cb_id_in0)) + start_read_offset_bytes;
    uint32_t l1_write_addr = get_write_ptr(cb_id_out0) + start_write_offset_bytes;

    for (uint32_t i = 0; i < nheads; ++i) {
        uint32_t curr_l1_write_addr = l1_write_addr;
        for (uint32_t j = 0; j < in0_h_tiles; ++j) {
            noc_async_read(noc_l1_read_addr, curr_l1_write_addr, head_dim_size_bytes);
            noc_l1_read_addr += head_dim_size_bytes;
            curr_l1_write_addr += out_row_size_bytes;
        }
        l1_write_addr += head_dim_size_bytes;
    }

    noc_async_read_barrier();
    // cb_push_back(cb_id_out0, block_size);
}
