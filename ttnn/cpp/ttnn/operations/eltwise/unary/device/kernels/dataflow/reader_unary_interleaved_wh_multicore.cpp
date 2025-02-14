
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t start_id = get_arg_val<uint32_t>(1);
    uint32_t single_block_size_row_arg = get_arg_val<uint32_t>(2);
    uint32_t single_block_size_col_arg = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t num_tiles_per_2d = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);
    uint32_t total_tiles_per_row = get_compile_time_arg_val(3);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

#ifdef BACKWARDS
    for (uint32_t dim = 0; dim > -third_dim; dim--) {
        for (uint32_t c = 0; c > -single_block_size_col_arg; c--) {
            for (uint32_t r = 0; r > -single_block_size_row_arg; r--) {
                uint32_t tile = -start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
#else
    for (uint32_t dim = 0; dim < third_dim; dim++) {
        for (uint32_t c = 0; c < single_block_size_col_arg; c++) {
            for (uint32_t r = 0; r < single_block_size_row_arg; r++) {
                uint32_t tile = start_id + dim * num_tiles_per_2d + c * total_tiles_per_row + r;
#endif
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read_tile(tile, s, l1_write_addr);

                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
            }
        }
    }
}
