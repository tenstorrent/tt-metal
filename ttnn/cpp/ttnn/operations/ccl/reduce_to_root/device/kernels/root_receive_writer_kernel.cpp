// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

// receives l, m, s tensors from the compute kernel and writes to intermediate buffers
//  this is going to happen twice:
//  first from device 0 where we write to the intermediate buffers
//  second from device 1 where we write to the final output buffers

using tt::data_movement::common::tt_memmove;

inline void write_intermediate_data(
    uint32_t cb_int_l,
    uint32_t cb_int_s,
    uint32_t cb_int_m,
    uint32_t compute_cb_l,
    uint32_t compute_cb_s,
    uint32_t compute_cb_m,
    uint32_t onetile,
    uint32_t input_num_tiles,
    uint32_t page_bytes) {
    // mmove from intermediate cbs to compute cbs
    // DPRINT << "moving from intermediate cbs to compute cbs\n";
    // DPRINT << "waiting front on cbs: l, s, m " << (uint32_t)cb_int_l << ", " << (uint32_t)cb_int_s << ", "
    //       << (uint32_t)cb_int_m << "\n";
    // DPRINT << "PUSHING TO INTERMEDIATE CBS: l, s, m " << (uint32_t)compute_cb_l << ", " << (uint32_t)compute_cb_s
    //       << ", " << (uint32_t)compute_cb_m << "\n";

    // for tensor l
    cb_wait_front(cb_int_l, input_num_tiles);
    // DPRINT << "printing  l before moving it\n";
    // print_full_tile(cb_int_l, 1, false);
    uint32_t l1_read_addr = get_read_ptr(cb_int_l);
    cb_reserve_back(compute_cb_l, input_num_tiles);
    uint32_t l1_write_addr = get_write_ptr(compute_cb_l);
    tt_memmove<false, false, false, 0>(l1_write_addr, l1_read_addr, input_num_tiles * page_bytes);
    // DPRINT << "printing moved l from compute cb l\n";
    // print_full_tile(compute_cb_l, 1, false);
    cb_push_back(compute_cb_l, input_num_tiles);
    // DPRINT << "AFTER push back\n";
    cb_pop_front(cb_int_l, input_num_tiles);
    // DPRINT << "AFTER pop front\n";

    // for tensor s
    cb_wait_front(cb_int_s, onetile);
    // DPRINT << "printing  s before moving it\n";
    // print_full_tile(cb_int_s, 0, false);
    // DPRINT << "waiting front for s tensor\n";
    l1_read_addr = get_read_ptr(cb_int_s);
    cb_reserve_back(compute_cb_s, onetile);
    l1_write_addr = get_write_ptr(compute_cb_s);
    tt_memmove<false, false, false, 0>(l1_write_addr, l1_read_addr, onetile * page_bytes);
    // DPRINT << "printing moved s from compute cb s\n";
    // print_full_tile(compute_cb_s, 0, false);
    cb_push_back(compute_cb_s, onetile);
    cb_pop_front(cb_int_s, onetile);

    // for tensor m
    cb_wait_front(cb_int_m, onetile);
    // DPRINT << "printing m before moving it\n";
    // print_full_tile(cb_int_m, 0, false);
    // DPRINT << "waiting front for m tensor\n";
    l1_read_addr = get_read_ptr(cb_int_m);
    cb_reserve_back(compute_cb_m, onetile);
    l1_write_addr = get_write_ptr(compute_cb_m);
    tt_memmove<false, false, false, 0>(l1_write_addr, l1_read_addr, onetile * page_bytes);
    // DPRINT << "printing moved m from compute cb m\n";
    // print_full_tile(compute_cb_m, 0, false);
    cb_push_back(compute_cb_m, onetile);
    cb_pop_front(cb_int_m, onetile);
}

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
    // DPRINT << "start writing data\n";

    cb_wait_front(cb_int_cb_l, input_num_tiles);
    // DPRINT << "printing final l tensor\n";
    // print_full_tile(cb_int_cb_l, 0, false);
    // DPRINT << "waiting front for s tensor\n";
    uint32_t l1_read_addr = get_read_ptr(cb_int_cb_l);
    uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_l, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, input_num_tiles * page_bytes);
    noc_async_write_barrier();
    // DPRINT << "after noc write for s tensor\n";
    cb_pop_front(cb_int_cb_l, input_num_tiles);

    // DPRINT << "finished writing l tensor\n";
    //  for tensor s
    cb_wait_front(cb_int_cb_s, onetile);
    // DPRINT << "printing final s tensor\n";
    // print_full_tile(cb_int_cb_s, 0, false);
    // DPRINT << "waiting front for s tensor\n";
    l1_read_addr = get_read_ptr(cb_int_cb_s);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_s, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    noc_async_write_barrier();
    // DPRINT << "after noc write for s tensor\n";
    cb_pop_front(cb_int_cb_s, onetile);

    // DPRINT << "finished writing s tensor\n";

    // for tensor m
    cb_wait_front(cb_int_cb_m, onetile);
    // DPRINT << "printing final m tensor\n";
    // print_full_tile(cb_int_cb_m, 0, false);
    l1_read_addr = get_read_ptr(cb_int_cb_m);
    dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr_m, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, onetile * page_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_int_cb_m, onetile);
    // DPRINT << "finished writing m tensor\n";
}
void kernel_main() {
    // DPRINT << "root writer kernel started\n";
    uint32_t inter_dst_addr_l = get_arg_val<uint32_t>(0);
    uint32_t inter_dst_addr_s = get_arg_val<uint32_t>(1);
    uint32_t inter_dst_addr_m = get_arg_val<uint32_t>(2);
    uint32_t final_dst_addr_l = get_arg_val<uint32_t>(3);
    uint32_t final_dst_addr_s = get_arg_val<uint32_t>(4);
    uint32_t final_dst_addr_m = get_arg_val<uint32_t>(5);
    const uint32_t core_noc_x = get_arg_val<uint32_t>(6);
    const uint32_t core_noc_y = get_arg_val<uint32_t>(7);

    constexpr uint32_t cb_int_cb_l = get_compile_time_arg_val(0);
    constexpr uint32_t cb_int_cb_s = get_compile_time_arg_val(1);
    constexpr uint32_t cb_int_cb_m = get_compile_time_arg_val(2);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(3);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(4);

    // single-tile ublocks
    constexpr uint32_t onetile = 1;

    // DPRINT << "BEFORE writing first output to intermediate buffer\n";
    /*
    write_intermediate_data(
        cb_int_cb_l,
        cb_int_cb_s,
        cb_int_cb_m,
        inter_dst_addr_l,
        inter_dst_addr_s,
        inter_dst_addr_m,
        onetile,
        input_num_tiles,
        page_bytes);
    */
    // DPRINT << "after writing first output to intermediate buffer\n";
    //  write l, s, m from device 2 to final buffers
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
    // DPRINT << "root writer kernel completed\n";
}
