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
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t row_size = get_compile_time_arg_val(2);
constexpr uint32_t num_rows_per_packet = get_compile_time_arg_val(3);

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
    uint32_t row_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t row_id_end = get_arg_val<uint32_t>(arg_idx++);

#ifdef SHARDED
    typedef ShardedInfo<
        get_compile_time_arg_val(4),
        get_compile_time_arg_val(5),
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10)>
        tensor_shard_info;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx++));
    experimental::ShardedAddrGen<tensor_shard_info> tensor0_addrgen = {
        .bank_base_address = tensor_address0, .shard_array = mapping_table};
#else
    constexpr auto tensor0_args = TensorAccessorArgs<4>();
    auto tensor0_addrgen = TensorAccessor(tensor0_args, tensor_address0, row_size);
#endif

    uint32_t row_id = row_id_start;
    while (row_id < row_id_end) {
        cb_reserve_back(cb0_id, num_rows_per_packet);
        uint32_t l1_write_addr = get_write_ptr(cb0_id);
        for (uint32_t i = 0; i < num_rows_per_packet && row_id < row_id_end; ++i) {
            uint64_t noc_src_addr = get_noc_addr(row_id, tensor0_addrgen);
            noc_async_read(noc_src_addr, l1_write_addr, page_size);
            l1_write_addr += page_size;
            row_id++;
        }
        noc_async_read_barrier();
        cb_push_back(cb0_id, num_rows_per_packet);
    }
}
