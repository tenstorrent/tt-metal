// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

void kernel_main() {
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(0);

    uint32_t rt = 0;
    const uint32_t dst_base_addr = get_arg_val<uint32_t>(rt++);
    const uint32_t num_tiles_core = get_arg_val<uint32_t>(rt++);
    const uint32_t shard_start_tile = get_arg_val<uint32_t>(rt++);
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out);

#ifdef SHARDED
    using tensor_shard_info = ShardedInfo<
        get_compile_time_arg_val(1),
        get_compile_time_arg_val(2),
        get_compile_time_arg_val(3),
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7)>;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {.bank_base_address = dst_base_addr, .shard_array = mapping_table};
#else
    constexpr auto dst_args = TensorAccessorArgs<1>();
    constexpr auto s0 = TensorAccessor(dst_args, dst_base_addr, tile_bytes);
#endif

    // Each core writes its own global tile range [shard_start_tile ... shard_start_tile + num_tiles_core]
    for (uint32_t t = 0; t < num_tiles_core; ++t) {
        cb_wait_front(cb_id_out, 1);
        uint32_t l1_read = get_read_ptr(cb_id_out);

        const uint32_t global_tile = shard_start_tile + t;
        const uint64_t dst_noc_addr = get_noc_addr(global_tile, s0);

        noc_async_write(l1_read, dst_noc_addr, tile_bytes);
        noc_async_write_barrier();
        cb_pop_front(cb_id_out, 1);
    }

}
