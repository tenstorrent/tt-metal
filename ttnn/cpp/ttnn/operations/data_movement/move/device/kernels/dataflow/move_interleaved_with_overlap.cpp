// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t num_tiles = get_arg_val<uint32_t>(3);
    uint32_t semaphore_addr = get_semaphore(get_arg_val<uint32_t>(4));
    uint32_t controller_noc_x = get_arg_val<uint32_t>(5);
    uint32_t controller_noc_y = get_arg_val<uint32_t>(6);
    uint32_t control_value = get_arg_val<uint32_t>(7);
    bool is_controller = get_arg_val<uint32_t>(8) == 1;
    uint32_t range_0_start_noc_x = get_arg_val<uint32_t>(9);
    uint32_t range_0_start_noc_y = get_arg_val<uint32_t>(10);
    uint32_t range_0_end_noc_x = get_arg_val<uint32_t>(11);
    uint32_t range_0_end_noc_y = get_arg_val<uint32_t>(12);
    uint32_t range_0_size = get_arg_val<uint32_t>(13);
    uint32_t range_1_start_noc_x = get_arg_val<uint32_t>(14);
    uint32_t range_1_start_noc_y = get_arg_val<uint32_t>(15);
    uint32_t range_1_end_noc_x = get_arg_val<uint32_t>(16);
    uint32_t range_1_end_noc_y = get_arg_val<uint32_t>(17);
    uint32_t range_1_size = get_arg_val<uint32_t>(18);
    uint32_t range_2_start_noc_x = get_arg_val<uint32_t>(19);
    uint32_t range_2_start_noc_y = get_arg_val<uint32_t>(20);
    uint32_t range_2_end_noc_x = get_arg_val<uint32_t>(21);
    uint32_t range_2_end_noc_y = get_arg_val<uint32_t>(22);
    uint32_t range_2_size = get_arg_val<uint32_t>(23);
    bool do_third_multicast = get_arg_val<uint32_t>(24) == 1;

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr bool src_is_dram = get_compile_time_arg_val(1) == 1;
    constexpr bool dst_is_dram = get_compile_time_arg_val(2) == 1;

    const DataFormat data_format = get_dataformat(cb_id);

    // if controller core then this local address will be incremented by remote cores,
    // otherwise controller core will set this to signal that write to dst can be done once controller core sees
    // control_value locally
    volatile uint32_t *semaphore_addr_ptr = reinterpret_cast<volatile uint32_t *>(semaphore_addr);

    // ublocks size defined in tiles
    constexpr uint32_t ublock_size_tiles = 1;
    uint32_t tile_bytes = get_tile_size(cb_id);

    const InterleavedAddrGenFast<src_is_dram> src_addrgen = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};

    const InterleavedAddrGenFast<dst_is_dram> dst_addrgen = {
        .bank_base_address = dst_addr, .page_size = tile_bytes, .data_format = data_format};

    // read a ublock of tiles from src to CB
    cb_reserve_back(cb_id, num_tiles);
    uint32_t l1_write_addr = get_write_ptr(cb_id);
    for (uint32_t i = start_id; i < start_id + num_tiles; i += ublock_size_tiles) {
        noc_async_read_tile(i, src_addrgen, l1_write_addr);
        noc_async_read_barrier();
        l1_write_addr += tile_bytes;
    }
    cb_push_back(cb_id, num_tiles);

    if (is_controller) {
        noc_semaphore_wait(semaphore_addr_ptr, control_value);

        // signal to cores that write to dst can begin
        uint64_t range0_multicast_semaphore_addr = get_noc_multicast_addr(
            range_0_start_noc_x, range_0_start_noc_y, range_0_end_noc_x, range_0_end_noc_y, semaphore_addr);
        noc_semaphore_set_multicast(semaphore_addr, range0_multicast_semaphore_addr, range_0_size);
        uint64_t range1_multicast_semaphore_addr = get_noc_multicast_addr(
            range_1_start_noc_x, range_1_start_noc_y, range_1_end_noc_x, range_1_end_noc_y, semaphore_addr);
        noc_semaphore_set_multicast(semaphore_addr, range1_multicast_semaphore_addr, range_1_size);
        if (do_third_multicast) {
            uint64_t range2_multicast_semaphore_addr = get_noc_multicast_addr(
                range_2_start_noc_x, range_2_start_noc_y, range_2_end_noc_x, range_2_end_noc_y, semaphore_addr);
            noc_semaphore_set_multicast(semaphore_addr, range2_multicast_semaphore_addr, range_2_size);
        }
    } else {
        // increment controller core semaphore
        uint64_t controller_noc_address = get_noc_addr(controller_noc_x, controller_noc_y, semaphore_addr);
        noc_semaphore_inc(controller_noc_address, 1);
        // wait for controller to signal write
        noc_semaphore_wait(semaphore_addr_ptr, control_value);
    }

    cb_wait_front(cb_id, num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_id);
    for (uint32_t i = start_id; i < start_id + num_tiles; i += ublock_size_tiles) {
        noc_async_write_tile(i, dst_addrgen, l1_read_addr);
        noc_async_write_barrier();
        l1_read_addr += tile_bytes;
    }
    cb_pop_front(cb_id, num_tiles);
}
