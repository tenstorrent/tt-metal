
// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t core_number = get_arg_val<uint32_t>(1);
    uint32_t tiles_per_row = get_arg_val<uint32_t>(2);
    uint32_t num_blocks = get_arg_val<uint32_t>(3);

    DPRINT << "src_addr: " << src_addr << ENDL();
    DPRINT << "core_number: " << core_number << ENDL();
    DPRINT << "tiles_per_row: " << tiles_per_row << ENDL();
    DPRINT << "num_blocks: " << num_blocks << ENDL();
    constexpr uint32_t cb_id_in0 = 0;
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    const uint32_t num_tiles_per_2d = get_compile_time_arg_val(1);
    const uint32_t third_dim = get_compile_time_arg_val(2);
    const uint32_t number_blocks_per_core = get_compile_time_arg_val(3);

    DPRINT << "num_tiles_per_2d: " << num_tiles_per_2d << ENDL();
    DPRINT << "third_dim: " << third_dim << ENDL();
    DPRINT << "number_blocks_per_core: " << number_blocks_per_core << ENDL();

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
        DPRINT << "FOR DIM =" << dim << ENDL();
        for (uint32_t k = 0; k < num_blocks; k++) {
            DPRINT << "for k = : " << k << ENDL();
            DPRINT << "start i is: " << num_tiles_per_2d * dim + number_blocks_per_core * core_number << ENDL();
            DPRINT << "end i is:" << end_id + num_tiles_per_2d * dim << ENDL();
            DPRINT << "increment i by :" << tiles_per_row << ENDL();
            for (uint32_t i = num_tiles_per_2d * dim + number_blocks_per_core * core_number;
                 i < end_id + num_tiles_per_2d * dim;
                 i = i + tiles_per_row) {
#endif
                DPRINT << "Reading for i=: " << i << ENDL();
                cb_reserve_back(cb_id_in0, onetile);
                uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
                noc_async_read_tile(i + k, s, l1_write_addr);

                auto* ptr0 = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(l1_write_addr);
                for (uint32_t i0 = 0; i0 < 1024; i0 = i0 + 1) {
                    DPRINT << "IN THE WRITER VALUE AT i0 = " << (uint32_t)i0 << " is: " << BF16((uint16_t)ptr0[i0])
                           << ENDL();
                }
                noc_async_read_barrier();
                cb_push_back(cb_id_in0, onetile);
            }
        }
    }
#endif
}
