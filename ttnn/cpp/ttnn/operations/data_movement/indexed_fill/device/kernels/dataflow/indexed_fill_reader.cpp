// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

#include "debug/dprint.h"

void kernel_main() {
    uint32_t batch_ids_addr = get_arg_val<uint32_t>(0);
    uint32_t batch_id_size = get_arg_val<uint32_t>(1);
    uint32_t input_addr_a = get_arg_val<uint32_t>(2);
    uint32_t input_addr_b = get_arg_val<uint32_t>(3);
    uint32_t batch_size_in_sticks = get_arg_val<uint32_t>(4);
    uint32_t my_batch_id = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(0);
    constexpr uint32_t batch_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t stick_size = get_compile_time_arg_val(2);
    constexpr auto src0_args = TensorAccessorArgs<3>();
    constexpr auto src1_args = TensorAccessorArgs<src0_args.next_compile_time_args_offset()>();
    constexpr auto batch_ids_args = TensorAccessorArgs<src1_args.next_compile_time_args_offset()>();

    const auto s0 = TensorAccessor(src0_args, input_addr_a, stick_size);
    const auto s1 = TensorAccessor(src1_args, input_addr_b, stick_size);

    const auto batchAddr = TensorAccessor(batch_ids_args, batch_ids_addr, batch_id_size << 2);

    bool replace_batch = false;
    uint32_t batch_to_replace_id = 0;
    // first go through batch id

    volatile tt_l1_ptr int* addr_ptr;

    if (batch_id_size > 0) {
        uint64_t src_noc_addr = get_noc_addr(0, batchAddr);
        uint32_t l1_write_addr = get_write_ptr(batch_cb_id);
        noc_async_read(src_noc_addr, l1_write_addr, (batch_id_size << 2));
        noc_async_read_barrier();
        addr_ptr = reinterpret_cast<volatile tt_l1_ptr int*>(l1_write_addr);
    }
    for (uint32_t i = 0; i < batch_id_size; i++) {
        uint32_t batch_id_to_replace = addr_ptr[i];
        if (batch_id_to_replace == my_batch_id) {
            replace_batch = true;
            batch_to_replace_id = i;
        }
    }

    uint32_t start_id;
    if (replace_batch) {
        start_id = batch_to_replace_id;
    } else {
        start_id = my_batch_id;
    }

    uint32_t end_id = start_id + batch_size_in_sticks;
    for (uint32_t i = start_id; i < end_id; ++i) {
        cb_reserve_back(cb_id_in0, 1);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        uint64_t src_noc_addr;
        if (replace_batch) {
            src_noc_addr = get_noc_addr(i, s1);
        } else {
            src_noc_addr = get_noc_addr(i, s0);
        }
        noc_async_read(src_noc_addr, l1_write_addr, stick_size);
        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
    }
}
