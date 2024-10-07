// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

constexpr uint32_t onetile = 1;

void kernel_main() {
    constexpr bool src0_is_dram = get_compile_time_arg_val(0) == 1;

    // uint32_t src_addr = get_arg_val<uint32_t>(0);
    // uint32_t value = get_arg_val<uint32_t>(1);
    // uint32_t num_tiles = get_arg_val<uint32_t>(2);
    uint32_t fill_value = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_value = tt::CB::c_intermed0;
    constexpr uint32_t onetile = 1;

    const uint32_t src_tile_bytes = get_tile_size(cb_value);
    const DataFormat src_data_format = get_dataformat(cb_value);
    DPRINT << "Ditmemay1" << ENDL();

    cb_reserve_back(cb_value, onetile);


    uint32_t write_addr = get_write_ptr(cb_value);
    DPRINT << "Ditmemay2" << ENDL();
    auto ptr = reinterpret_cast<uint32_t *>(write_addr);
    DPRINT << fill_value << ENDL();

    for (uint32_t i = 0; i < 1024; ++i) {
        ptr[i] = fill_value;
        // DPRINT << "------------- " << ptr+i << ENDL();
    }
    // DPRINT << TSLICE(cb_value, 0, SliceRange::hw0_w0_32()) << ENDL();

    DPRINT << "Ditmemay4" << ENDL();

    cb_push_back(cb_value, 1);

    DPRINT << "Ditmemay" << ENDL();

}
