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

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    uint32_t tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_end = get_arg_val<uint32_t>(arg_idx++);

#ifdef SHARDED
    typedef ShardedInfo<
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9)>
        tensor_shard_info;
    // Sharded addrgen
    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx++));
    experimental::ShardedAddrGen<tensor_shard_info> tensor0_addrgen = {
        .bank_base_address = tensor_address0, .shard_array = mapping_table};
#else
    constexpr auto tensor0_args = TensorAccessorArgs<3>();
    auto tensor0_addrgen = TensorAccessor(tensor0_args, tensor_address0, tensor0_page_size);
#endif

    uint32_t tile_id = tile_id_start;
    while (tile_id < tile_id_end) {
        cb_reserve_back(cb0_id, packet_size_in_pages);
        const uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
        uint32_t l1_write_addr = l1_write_addr_base;

        uint32_t num_pages_to_read = std::min(tile_id_end - tile_id, packet_size_in_pages);
        for (uint32_t j = 0; j < num_pages_to_read; j++) {
#ifdef SHARDED
            noc_async_read_page(tile_id, tensor0_addrgen, l1_write_addr);
#else
            noc_async_read_tile(tile_id, tensor0_addrgen, l1_write_addr);
#endif
            l1_write_addr += tensor0_page_size;
            tile_id++;
        }

        noc_async_read_barrier();
        cb_push_back(cb0_id, packet_size_in_pages);
    }
}
