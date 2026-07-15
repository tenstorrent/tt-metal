// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

void kernel_main() {
    uint32_t src_addr = get_arg_val<uint32_t>(0);
    uint32_t dst_addr = get_arg_val<uint32_t>(1);
    uint32_t start_id = get_arg_val<uint32_t>(2);
    uint32_t num_pages = get_arg_val<uint32_t>(3);
    uint32_t semaphore_arg = get_arg_val<uint32_t>(4);
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
    uint32_t aligned_page_size = get_arg_val<uint32_t>(25);

    constexpr uint32_t cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t page_size = get_compile_time_arg_val(1);
    constexpr auto src_args = TensorAccessorArgs<2>();
    constexpr auto dst_args = TensorAccessorArgs<src_args.next_compile_time_args_offset()>();

    Noc noc;
    CircularBuffer cb(cb_id);

    const auto src_addrgen = TensorAccessor(src_args, src_addr);
    const auto dst_addrgen = TensorAccessor(dst_args, dst_addr);

    // if controller core then this local address will be incremented by remote cores,
    // otherwise controller core will set this to signal that write to dst can be done once controller core sees
    // control_value locally
    Semaphore<> sem(semaphore_arg);

    // read a ublock of tiles from src to CB
    cb.reserve_back(num_pages);
    uint32_t l1_write_addr = cb.get_write_ptr();
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        CoreLocalMem<uint32_t> dst(l1_write_addr);
        noc.async_read(src_addrgen, dst, page_size, {.page_id = i, .offset_bytes = 0}, {.offset_bytes = 0});
        noc.async_read_barrier();
        l1_write_addr += aligned_page_size;
    }
    cb.push_back(num_pages);

    if (is_controller) {
        sem.wait(control_value);

        // signal to cores that write to dst can begin
        sem.set_multicast<NocOptions::DEFAULT>(
            noc, range_0_start_noc_x, range_0_start_noc_y, range_0_end_noc_x, range_0_end_noc_y, range_0_size);
        sem.set_multicast<NocOptions::DEFAULT>(
            noc, range_1_start_noc_x, range_1_start_noc_y, range_1_end_noc_x, range_1_end_noc_y, range_1_size);
        if (do_third_multicast) {
            sem.set_multicast<NocOptions::DEFAULT>(
                noc, range_2_start_noc_x, range_2_start_noc_y, range_2_end_noc_x, range_2_end_noc_y, range_2_size);
        }
    } else {
        // increment controller core semaphore
        sem.up(noc, controller_noc_x, controller_noc_y, 1);
        // wait for controller to signal write
        sem.wait(control_value);
    }

    cb.wait_front(num_pages);
    uint32_t l1_read_addr = cb.get_read_ptr();
    for (uint32_t i = start_id; i < start_id + num_pages; ++i) {
        CoreLocalMem<uint32_t> src(l1_read_addr);
        noc.async_write(src, dst_addrgen, page_size, {.offset_bytes = 0}, {.page_id = i, .offset_bytes = 0});
        noc.async_write_barrier();
        l1_read_addr += aligned_page_size;
    }
    cb.pop_front(num_pages);
}
