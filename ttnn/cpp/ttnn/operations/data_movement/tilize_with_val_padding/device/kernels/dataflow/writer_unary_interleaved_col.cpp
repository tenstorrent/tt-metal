// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t total_num_tiles = get_arg_val<uint32_t>(1);
    uint32_t core_number = get_arg_val<uint32_t>(2);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(3);
    uint32_t number_tiles_per_core = get_arg_val<uint32_t>(4);
    uint32_t num_blocks = get_arg_val<uint32_t>(5);

    // DPRINT << "total_num_tiles: " << total_num_tiles <<ENDL();
    // DPRINT << "tiles_per_row: " << tiles_per_row <<ENDL();
    // DPRINT << "number_tiles_per_core: " << number_tiles_per_core <<ENDL();

    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = get_compile_time_arg_val(1) == 1;
    const uint32_t num_tiles_per_2d = get_compile_time_arg_val(2);
    const uint32_t third_dim = get_compile_time_arg_val(3);
    const uint32_t number_blocks_per_core = get_compile_time_arg_val(4);
    // DPRINT << "num_tiles_per_2d: " << num_tiles_per_2d <<ENDL();

#ifdef OUT_SHARDED
    cb_wait_front(cb_id_out, onetile);
#else

    // single-tile ublocks
    constexpr uint32_t onetile = 1;
    const uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);

    const InterleavedAddrGenFast<dst_is_dram> s = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

#ifdef BACKWARDS
    uint32_t end_id = total_num_tiles;
    // DPRINT<< "BACKWARDS "<< ENDL();
    uint32_t core_multiply = -1;
    for (uint32_t dim = third_dim - 1; dim >= 0; dim--) {
        for (uint32_t k = num_blocks - 1; k >= 0; k--) {
            for (uint32_t i = num_tiles_per_2d * dim + number_blocks_per_core * core_number;
                 i != end_id - num_tiles_per_2d * dim;
                 i = i - tiles_per_row) {
#else
    // DPRINT<< "FORWARD "<< ENDL();
    uint32_t end_id = total_num_tiles;
    uint32_t core_multiply = 1;
    for (uint32_t dim = 0; dim < third_dim; dim++) {
        // DPRINT << "DIM = " << dim << ENDL();
        for (uint32_t k = 0; k < num_blocks; k++) {
            for (uint32_t i = num_tiles_per_2d * dim + number_blocks_per_core * core_number;
                 i < end_id + num_tiles_per_2d * dim;
                 i = i + tiles_per_row) {
#endif
                // DPRINT << "writer drpint for i=" << i + k << ENDL();
                cb_wait_front(cb_id_out, onetile);
                // DPRINT << "IT IS THE read" << ENDL();
                uint32_t l1_read_addr = get_read_ptr(cb_id_out);

                // DPRINT << "WRITING NOW TILE NUMBER "<< i + k<< ENDL();
                noc_async_write_tile(i + k, s, l1_read_addr);
                // auto* ptr0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_read_addr);
                // for (uint32_t i0 = 0; i0 < 1024; i0 = i0+1) {
                //     DPRINT << "IN THE WRITER VALUE AT i0 = " << (uint32_t)i0 <<  " is: " << BF16((uint16_t)ptr0[i0])
                //     << ENDL();
                // }

                noc_async_write_barrier();
                cb_pop_front(cb_id_out, onetile);
            }
        }
    }
#endif
}
