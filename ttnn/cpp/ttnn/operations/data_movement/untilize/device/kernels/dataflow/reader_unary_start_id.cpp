// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // run-time args
    uint32_t src_addr = get_arg_val<uint32_t>(0);

    // compile-time args
    constexpr bool src_is_dram = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(1);
    constexpr uint32_t num_tiles = get_compile_time_arg_val(2);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(3);
    constexpr uint32_t start_page_id = get_compile_time_arg_val(4);

#ifdef SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(5),    // Memory layout
        get_compile_time_arg_val(6),    // The number of sharding cores
        get_compile_time_arg_val(7),    // The page size we offset each write to
        get_compile_time_arg_val(8),    // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(9),    // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(10),   // pages_per_shard_x
        get_compile_time_arg_val(11)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(1));
    experimental::ShardedAddrGen<tensor_shard_info> s = {.bank_base_address = src_addr, .shard_array = mapping_table};
#else
    const DataFormat data_format = get_dataformat(cb_id_in0);
    const InterleavedAddrGenFast<src_is_dram> s = {
        .bank_base_address = src_addr, .page_size = tile_bytes, .data_format = data_format};
#endif

    uint32_t end_page_id = start_page_id + num_tiles;
    for (uint32_t page_id = start_page_id; page_id < end_page_id; ++page_id) {
        cb_reserve_back(cb_id_in0, 1);

        uint64_t noc_read_addr = get_noc_addr(page_id, s);
        uint32_t l1_write_addr = get_write_ptr(cb_id_in0);
        noc_async_read(noc_read_addr, l1_write_addr, tile_bytes);

        noc_async_read_barrier();
        cb_push_back(cb_id_in0, 1);
    }
}
