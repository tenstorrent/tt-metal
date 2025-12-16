// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

using address_t = uint32_t;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t cb0_id = get_compile_time_arg_val(0);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(1);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(2);
constexpr bool is_sender = get_compile_time_arg_val(3);
constexpr uint32_t core_noc_x = get_compile_time_arg_val(4);
constexpr uint32_t core_noc_y = get_compile_time_arg_val(5);

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    if (is_sender) {
        size_t arg_idx = 0;
        // Load the input tensor spec
        uint32_t tensor_address0 = get_arg_val<uint32_t>(arg_idx++);
        uint32_t tile_id_start = get_arg_val<uint32_t>(arg_idx++);
        uint32_t tile_id_end = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_pages_to_read = std::min(tile_id_end - tile_id_start, packet_size_in_pages);
        cb_reserve_back(cb0_id, packet_size_in_pages);
        const uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t base_src_addr = get_noc_addr(core_noc_x, core_noc_y, tensor_address0);
        uint64_t read_addr = base_src_addr + (tile_id_start * tensor0_page_size);
        noc_async_read(read_addr, l1_write_addr, num_pages_to_read * tensor0_page_size);
        noc_async_read_barrier();
        cb_push_back(cb0_id, packet_size_in_pages);
    }
}
