// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(0));
constexpr uint32_t cb0_id = get_compile_time_arg_val(1);
constexpr uint32_t page_size = get_compile_time_arg_val(2);
constexpr uint32_t row_size = get_compile_time_arg_val(3);
constexpr uint32_t num_packets_per_row = get_compile_time_arg_val(4);
constexpr uint32_t max_packet_size = get_compile_time_arg_val(5);

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

    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;

#ifdef SHARDED
    typedef ShardedInfo<
        get_compile_time_arg_val(6),
        get_compile_time_arg_val(7),
        get_compile_time_arg_val(8),
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12)>
        tensor_shard_info;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx++));
    experimental::ShardedAddrGen<tensor_shard_info> tensor0_addrgen = {
        .bank_base_address = tensor_address0, .shard_array = mapping_table};
#else
    // interleaved addrgen
    const auto tensor0_addrgen = get_interleaved_addr_gen<is_dram, row_size>(tensor_address0);
#endif

    uint32_t row_id = row_id_start;
    while (row_id < row_id_end) {
        cb_reserve_back(cb0_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t noc_src_addr = get_noc_addr(row_id, tensor0_addrgen);

        for (uint32_t j = 0; j < num_packets_per_row; j++) {
            uint32_t packet_size = std::min(max_packet_size, page_size);
            packet_size = std::min(packet_size, page_size - max_packet_size * j);
            noc_async_read(noc_src_addr, l1_write_addr, packet_size);

            l1_write_addr += packet_size;
            noc_src_addr += packet_size;  // advance the noc address for the next packet
        }
        row_id++;
        noc_async_read_barrier();
        cb_push_back(cb0_id, 1);
    }
}
