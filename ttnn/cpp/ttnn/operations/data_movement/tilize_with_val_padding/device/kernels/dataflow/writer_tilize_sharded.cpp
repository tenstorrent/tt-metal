// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    // Compile-time args
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    // Runtime args
    uint32_t rt_arg_ind = 0;
    uint32_t dst_addr = get_arg_val<uint32_t>(rt_arg_ind++);  // Buffer base address for ShardedAddrGen
    uint32_t num_tiles = get_arg_val<uint32_t>(rt_arg_ind++);

#ifdef SHARDED
    // ShardedAddrGen setup for output tensor
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(1),   // Memory layout
        get_compile_time_arg_val(2),   // Number of sharding cores
        get_compile_time_arg_val(3),   // Page size
        get_compile_time_arg_val(4),   // Pages per shard row
        get_compile_time_arg_val(5),   // Contiguous pages flag
        get_compile_time_arg_val(6),   // pages_per_shard_x
        get_compile_time_arg_val(7)>;  // pages_per_shard_y

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt_arg_ind));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {.bank_base_address = dst_addr, .shard_array = mapping_table};
#else
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out);
    constexpr auto dst_args = TensorAccessorArgs<1>();
    const auto s0 = TensorAccessor(dst_args, dst_addr, tile_bytes);
#endif

    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out);

    // Write tiles from CB to sharded output
    for (uint32_t tile_id = 0; tile_id < num_tiles; tile_id++) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read_addr = get_read_ptr(cb_id_out);

        uint64_t dst_noc_addr = get_noc_addr(tile_id, s0);
        noc_async_write(l1_read_addr, dst_noc_addr, tile_bytes);
        noc_async_write_barrier();  // Barrier before pop to ensure write completes

        cb_pop_front(cb_id_out, 1);
    }
}
