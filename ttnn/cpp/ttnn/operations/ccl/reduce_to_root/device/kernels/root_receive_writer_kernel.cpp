// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"

// receives l, m, s tensors from the compute kernel and writes to intermediate buffers
//  this is going to happen twice:
//  first from device 0 where we write to the intermediate buffers
//  second from device 1 where we write to the final output buffers

inline write_data(
    uint32_t dst_addr_l,
    uint32_t num_tiles_l,
    uint32_t dst_addr_s,
    uint32_t dst_addr_m,
    uint32_t page_bytes,
    uint32_t core_noc_x,
    uint32_t core_noc_y,
    uint32_t cb_int_cb_l,
    uint32_t cb_int_cb_s,
    uint32_t cb_int_cb_m,
    uint32_t onetile) {
    uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_l, 0);
    uint32_t chunk_size = 8;
    for (uint32_t i = 0; i < num_tiles_l / chunk_size; ++i) {
        cb_wait_front(cb_int_cb_l, chunk_size);
        uint32_t l1_read_addr = get_read_ptr(cb_int_cb_l);
        noc_async_write(l1_read_addr, dst_noc_addr, chunk_size * page_bytes);
        dst_noc_addr += chunk_size * page_bytes;
        noc_async_write_barrier();
        cb_pop_front(cb_int_cb_l, chunk_size);
    }

    // for tensor s
    cb_wait_front(cb_int_cb_s, onetile);
    uint32_t l1_read_addr = get_read_ptr(cb_int_cb_s);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_s, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    noc_async_write_barrier();
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
    uint32_t inter_dst_addr_l = get_arg_val<uint32_t>(0);
    uint32_t num_tiles_l = get_arg_val<uint32_t>(1);
    uint32_t inter_dst_addr_s = get_arg_val<uint32_t>(2);
    uint32_t inter_dst_addr_m = get_arg_val<uint32_t>(3);
    uint32_t final_dst_addr_l = get_arg_val<uint32_t>(4);
    uint32_t final_dst_addr_s = get_arg_val<uint32_t>(5);
    uint32_t final_dst_addr_m = get_arg_val<uint32_t>(6);

    constexpr core_noc_x = get_compile_time_arg_val(0);
    constexpr core_noc_y = get_compile_time_arg_val(1);
    constexpr uint32_t cb_int_cb_l = get_compile_time_arg_val(2);
    constexpr uint32_t cb_int_cb_s = get_compile_time_arg_val(3);
    constexpr uint32_t cb_int_cb_m = get_compile_time_arg_val(4);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    const uint32_t page_bytes = get_arg_val<uint32_t>(7);

    // write l, s, m from device 0 to intermediate buffers
    write_data(
        get_write_ptr(inter_dst_addr_l),
        num_tiles_l,
        get_write_ptr(inter_dst_addr_s),
        get_write_ptr(inter_dst_addr_m),
        page_bytes,
        core_noc_x,
        core_noc_y,
        cb_int_cb_l,
        cb_int_cb_s,
        cb_int_cb_m,
        onetile);
    // write l, s, m from device 2 to final buffers
    write_data(
        dst_addr_l,
        num_tiles_l,
        dst_addr_s,
        dst_addr_m,
        page_bytes,
        core_noc_x,
        core_noc_y,
        cb_int_cb_l,
        cb_int_cb_s,
        cb_int_cb_m,
        onetile);
}
