// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t num_tiles = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);
constexpr uint32_t core_noc_x = get_compile_time_arg_val(3);
constexpr uint32_t core_noc_y = get_compile_time_arg_val(4);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // read local data to cb
    DPRINT << "compile args\n";
    DPRINT << " cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << " num_tiles: " << (uint32_t)num_tiles << "\n";
    DPRINT << " tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << " core_noc_x: " << (uint32_t)core_noc_x << "\n";
    DPRINT << " core_noc_y: " << (uint32_t)core_noc_y << "\n";

    size_t arg_idx = 0;
    uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);

    DPRINT << "arg vals\n";
    DPRINT << " tensor_address0: " << (uint32_t)tensor_address0 << "\n";

    DPRINT << "start of sender reader kernel\n";
    cb_reserve_back(cb0_id, num_tiles);
    const uint32_t l1_write_addr = get_write_ptr(cb0_id);
    uint64_t base_src_addr = get_noc_addr(core_noc_x, core_noc_y, tensor_address0);
    noc_async_read(base_src_addr, l1_write_addr, num_tiles * tensor0_page_size);
    noc_async_read_barrier();
    cb_push_back(cb0_id, num_tiles);
    DPRINT << "end of sender reader kernel\n";
}
