// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include <cstdint>
#include "dataflow_api.h"
#include "hw/inc/dataflow_api.h"
#include "hw/inc/dataflow_api_addrgen.h"
#include "debug/dprint.h"
void kernel_main() {
    const uint32_t num_units = get_arg_val<uint32_t>(0);
    const uint32_t num_elements_per_row = get_arg_val<uint32_t>(1);
    const uint32_t unpadded_row_size_bytes = get_arg_val<uint32_t>(2);
    const uint32_t padded_row_size_bytes = get_arg_val<uint32_t>(3);
    const uint32_t pad_size_bytes = padded_row_size_bytes - unpadded_row_size_bytes;
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);
    constexpr uint32_t cb_temp_pad = get_compile_time_arg_val(1);

#ifdef DEBUG
    DPRINT << "num_units: " << num_units << ", num_elements_per_row: " << num_elements_per_row
           << ", unpadded_row_size_bytes: " << unpadded_row_size_bytes
           << ", padded_row_size_bytes: " << padded_row_size_bytes << ", pad_size_bytes: " << pad_size_bytes << ENDL();
#endif
    const uint32_t pad_addr = get_read_ptr(cb_temp_pad);
    const uint32_t out_addr = get_read_ptr(cb_id_out);
    volatile tt_l1_ptr uint16_t* pad_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(pad_addr);
    for (uint32_t i = 0; i < num_elements_per_row; ++i) {
        pad_ptr[i] = 0;  // Changing this to any other value causes a hang. Figure out why.
    }
    uint32_t write_addr = out_addr + unpadded_row_size_bytes;
    uint64_t pad_noc_addr = get_noc_addr(pad_addr + unpadded_row_size_bytes);
    for (uint32_t i = 0; i < num_units; ++i) {
        noc_async_read(pad_noc_addr, write_addr, pad_size_bytes);
        write_addr += padded_row_size_bytes;
    }
    noc_async_read_barrier();
    cb_wait_front(cb_id_out, num_units);
}
