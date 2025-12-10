// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

inline void write_data(
    uint32_t dst_addr_l,
    uint32_t dst_addr_s,
    uint32_t dst_addr_m,
    uint32_t page_bytes,
    uint32_t core_noc_x,
    uint32_t core_noc_y,
    uint32_t cb_int_cb_l,
    uint32_t cb_int_cb_s,
    uint32_t cb_int_cb_m,
    uint32_t onetile,
    uint32_t input_num_tiles) {
    cb_wait_front(cb_int_cb_l, input_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_int_cb_l);
    uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_l, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, input_num_tiles * page_bytes);
    cb_pop_front(cb_int_cb_l, input_num_tiles);

    //  for tensor s
    cb_wait_front(cb_int_cb_s, onetile);
    l1_read_addr = get_read_ptr(cb_int_cb_s);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_s, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    cb_pop_front(cb_int_cb_s, onetile);

    // for tensor m
    cb_wait_front(cb_int_cb_m, onetile);
    l1_read_addr = get_read_ptr(cb_int_cb_m);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_m, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_int_cb_m, onetile);
}
void kernel_main() {
    uint32_t final_dst_addr_l = get_arg_val<uint32_t>(0);
    uint32_t final_dst_addr_s = get_arg_val<uint32_t>(1);
    uint32_t final_dst_addr_m = get_arg_val<uint32_t>(2);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(3);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(4);

    constexpr uint32_t cb_int_cb_l = get_compile_time_arg_val(0);
    constexpr uint32_t cb_int_cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_int_cb_m = get_compile_time_arg_val(2);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);

    // receives l, m, s tensors from the compute kernel and writes to the final output buffers
    constexpr uint32_t onetile = 1;
    write_data(
        final_dst_addr_l,
        final_dst_addr_s,
        final_dst_addr_m,
        page_bytes,
        core_noc_x,
        core_noc_y,
        cb_int_cb_l,
        cb_int_cb_s,
        cb_int_cb_m,
        onetile,
        input_num_tiles);
}
