// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "debug/dprint_tile.h"

void kernel_main() {
    // Runtime args
    // ------------
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(0);
    const uint32_t dst_page_offset = get_arg_val<uint32_t>(1);

    // Compile time args
    // -----------------
    constexpr uint32_t dst_cb_idx = get_compile_time_arg_val(0);
    constexpr bool dst_is_dram = (bool)get_compile_time_arg_val(1);
    constexpr uint32_t tiles_per_page = get_compile_time_arg_val(2);
    constexpr uint32_t dst_page_size = get_compile_time_arg_val(3);
    constexpr uint32_t pages_per_group = get_compile_time_arg_val(4);
    constexpr uint32_t pages_per_batch = get_compile_time_arg_val(5);
    constexpr uint32_t num_batches = get_compile_time_arg_val(6);
    constexpr uint32_t scaler_cb_idx = get_compile_time_arg_val(7);

    //-------------------------------------------------------------------------
    // Write scaler to scaler cb
    cb_reserve_back(scaler_cb_idx, 1);
    const uint32_t scaler_cb_addr = get_write_ptr(scaler_cb_idx);
    const uint64_t scaler_noc_addr = get_noc_addr(scaler_cb_addr);

    volatile tt_l1_ptr uint32_t* ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(scaler_cb_addr);
    for (int j = 0; j < 128; ++j) {
        ptr[j] = 0x3f803f80;  // 1.0f packed twice
    }

    noc_async_read_one_packet_set_state(scaler_noc_addr, 512);
    noc_async_read_one_packet_with_state(scaler_noc_addr, scaler_cb_addr + (1 << 9));
    noc_async_read_one_packet_with_state(scaler_noc_addr, scaler_cb_addr + (2 << 9));
    noc_async_read_one_packet_with_state(scaler_noc_addr, scaler_cb_addr + (3 << 9));
    noc_async_read_barrier();

    cb_push_back(scaler_cb_idx, 1);

    //-------------------------------------------------------------------------
    const auto s_dst = get_interleaved_addr_gen<dst_is_dram, dst_page_size>(dst_base_addr);
    for (uint32_t batch = 0; batch < num_batches; ++batch) {
        const uint32_t batch_offset = batch * pages_per_batch;

        for (uint32_t page = 0; page < pages_per_group; ++page) {
            const uint64_t dst_noc_addr = s_dst.get_noc_addr(dst_page_offset + batch_offset + page);

            cb_wait_front(dst_cb_idx, tiles_per_page);
            const uint32_t dst_cb_addr = get_read_ptr(dst_cb_idx);
            noc_async_write(dst_cb_addr, dst_noc_addr, dst_page_size);
            cb_pop_front(dst_cb_idx, tiles_per_page);
        }
    }
    noc_async_write_barrier();
}
