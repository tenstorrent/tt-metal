// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include "api/dataflow/dataflow_api.h"
#include "cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);

    constexpr uint32_t cb_out = get_compile_time_arg_val(0);
    constexpr uint32_t input_num_tiles = get_compile_time_arg_val(1);
    constexpr uint32_t page_bytes = get_compile_time_arg_val(2);
    constexpr uint32_t core_noc_x = get_compile_time_arg_val(3);
    constexpr uint32_t core_noc_y = get_compile_time_arg_val(4);
    DPRINT << "compile time args:\n";
    DPRINT << " cb_out: " << (uint32_t)cb_out << "\n";
    DPRINT << " input_num_tiles: " << (uint32_t)input_num_tiles << "\n";
    DPRINT << " page_bytes: " << (uint32_t)page_bytes << "\n";
    DPRINT << " core_noc_x: " << (uint32_t)core_noc_x << "\n";
    DPRINT << " core_noc_y: " << (uint32_t)core_noc_y << "\n";

    DPRINT << "arg vals:\n";
    DPRINT << " dst_addr: " << (uint32_t)dst_addr << "\n";

    DPRINT << "start of receiver writer kernel\n";
    cb_wait_front(cb_out, input_num_tiles);
    uint32_t l1_read_addr = get_read_ptr(cb_out);
    uint64_t dst_noc_addr = get_noc_addr(core_noc_x, core_noc_y, dst_addr, 0);
    noc_async_write(l1_read_addr, dst_noc_addr, input_num_tiles * page_bytes);
    noc_async_write_barrier();
    cb_pop_front(cb_out, input_num_tiles);
    DPRINT << "end of receiver writer kernel\n";
}
