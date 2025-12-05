// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint.h"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t input_cb_index = get_compile_time_arg_val(0);
    constexpr uint32_t output_cb_index = get_compile_time_arg_val(1);
    constexpr uint32_t config_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t pixel_size = get_compile_time_arg_val(3);
    constexpr uint32_t aligned_pixel_size = get_compile_time_arg_val(4);
    constexpr uint32_t num_transfers = get_compile_time_arg_val(5);

    // Get CB addresses
    const uint32_t input_l1_addr = get_read_ptr(input_cb_index);
    const uint32_t output_l1_addr = get_write_ptr(output_cb_index);
    const uint32_t config_l1_addr = get_read_ptr(config_cb_index);

    DPRINT << "fold_with_config: num_transfers=" << num_transfers << " pixel_size=" << pixel_size
           << " aligned_pixel_size=" << aligned_pixel_size << ENDL();
    DPRINT << "  input_l1=" << input_l1_addr << " output_l1=" << output_l1_addr << " config_l1=" << config_l1_addr
           << ENDL();

    // Cast config to uint16_t pointer
    volatile tt_l1_ptr uint16_t* config_data = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(config_l1_addr);

    // Process each transfer entry from config tensor
    uint32_t dst_offset = 0;
    for (uint32_t i = 0; i < num_transfers; ++i) {
        // Read config entry: src_noc_x, src_noc_y, src_local_idx, length
        uint16_t src_noc_x = config_data[i * 4 + 0];
        uint16_t src_noc_y = config_data[i * 4 + 1];
        uint16_t src_local_idx = config_data[i * 4 + 2];
        uint16_t length = config_data[i * 4 + 3];

        // Debug: print first few transfers
        // if (i < 4) {
        //     DPRINT << "  Transfer[" << i << "]: noc=(" << src_noc_x << "," << src_noc_y
        //            << ") local_idx=" << src_local_idx << " len=" << length << ENDL();
        // }

        // Skip if length is 0 (padding entry)
        if (length == 0) {
            continue;
        }

        // Calculate source NOC address (use pixel_size for ROW_MAJOR memory layout)
        uint32_t src_offset = src_local_idx * pixel_size;
        uint64_t src_noc_addr = get_noc_addr(src_noc_x, src_noc_y, input_l1_addr + src_offset);

        // Calculate destination L1 address
        uint32_t dst_l1_addr = output_l1_addr + dst_offset;

        // Perform NOC read (read from remote core to local output buffer)
        uint32_t read_size = length * pixel_size;
        noc_async_read(src_noc_addr, dst_l1_addr, read_size);

        // Advance destination offset
        dst_offset += length * pixel_size;
    }

    // Wait for all reads to complete
    noc_async_read_barrier();
}
