
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t core_number = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t num_tiles_per_2d = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);
    const uint32_t number_blocks_per_core = get_compile_time_arg_val(3);

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_in0, onetile);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_in0);
    const DataFormat data_format = get_dataformat(cb_id_in0);

    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

#ifdef BACKWARDS
    uint32_t end_id = -num_tiles_per_2d;
    for (uint32_t dim = 0; dim > -third_dim; dim--) {
        for (uint32_t k = 0; k > -num_blocks; k--) {
            for (uint32_t i = num_tiles_per_2d * dim - number_blocks_per_core * core_number;
                 i > end_id + num_tiles_per_2d * dim;
                 i = i - tiles_per_row) {
#else
    uint32_t end_id = num_tiles_per_2d;
    for (uint32_t dim = 0; dim < third_dim; dim++) {
        for (uint32_t k = 0; k < num_blocks; k++) {
            for (uint32_t i = num_tiles_per_2d * dim + number_blocks_per_core * core_number;
                 i < end_id + num_tiles_per_2d * dim;
                 i = i + tiles_per_row) {
#endif
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read_tile(i + k, s, l1_write_addr);

                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
            }
        }
    }
#endif
}
