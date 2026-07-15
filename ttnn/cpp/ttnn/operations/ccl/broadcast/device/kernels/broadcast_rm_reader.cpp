// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
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
constexpr uint32_t page_size = get_compile_time_arg_val(1);
constexpr uint32_t num_rows_per_packet = get_compile_time_arg_val(3);
constexpr uint32_t num_packets_per_page = get_compile_time_arg_val(4);
constexpr uint32_t max_packet_size = get_compile_time_arg_val(5);
constexpr uint32_t is_sender = get_compile_time_arg_val(6);

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
        address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
        uint32_t row_id_start = get_arg_val<uint32_t>(arg_idx++);
        uint32_t row_id_end = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_rows_per_batch = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t output_rows_per_batch = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t input_height = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t output_height = get_arg_val<uint32_t>(arg_idx++);
        const uint32_t batch_begin = get_arg_val<uint32_t>(arg_idx++);

#ifdef SHARDED
        typedef ShardedInfo<
            get_compile_time_arg_val(7),
            get_compile_time_arg_val(8),
            get_compile_time_arg_val(9),
            get_compile_time_arg_val(10),
            get_compile_time_arg_val(11),
            get_compile_time_arg_val(12),
            get_compile_time_arg_val(13)>
            tensor_shard_info;

        const auto [mapping_table, rt_increment] =
            experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx++));
        experimental::ShardedAddrGen<tensor_shard_info> tensor0_addrgen = {
            .bank_base_address = tensor_address0, .shard_array = mapping_table};
#else
        constexpr auto tensor0_args = TensorAccessorArgs<7>();
        auto tensor0_addrgen = TensorAccessor(tensor0_args, tensor_address0);
#endif

        uint32_t row_id = row_id_start;
        while (row_id < row_id_end) {
            cb_reserve_back(cb0_id, num_rows_per_packet);
            uint32_t l1_write_addr = get_write_ptr(cb0_id);
            for (uint32_t i = 0; i < num_rows_per_packet && row_id < row_id_end; ++i) {
                uint32_t input_row_id = row_id;
#ifdef SELECT_INPUT_ROWS
                const uint32_t output_batch = row_id / output_rows_per_batch;
                const uint32_t row_in_output_batch = row_id % output_rows_per_batch;
                const uint32_t middle_index = row_in_output_batch / output_height;
                const uint32_t height_index = row_in_output_batch % output_height;
                input_row_id =
                    (output_batch + batch_begin) * input_rows_per_batch + middle_index * input_height + height_index;
#endif
#ifdef ND_FULL_WIDTH_ROW_BLOCKS
                constexpr uint32_t rows_per_shard = ND_ROWS_PER_SHARD;
                uint64_t noc_src_addr = tensor0_addrgen.get_shard_noc_addr(
                    input_row_id / rows_per_shard, (input_row_id % rows_per_shard) * page_size);
#else
                uint64_t noc_src_addr = tensor0_addrgen.get_noc_addr(input_row_id);
#endif
                uint32_t bytes_remaining = page_size;
                uint32_t offset = 0;
                for (uint32_t pkt = 0; pkt < num_packets_per_page && bytes_remaining > 0; ++pkt) {
                    uint32_t packet_size = std::min(max_packet_size, bytes_remaining);
                    noc_async_read(noc_src_addr + offset, l1_write_addr + offset, packet_size);
                    offset += packet_size;
                    bytes_remaining -= packet_size;
                }
                l1_write_addr += page_size;
                row_id++;
            }
            noc_async_read_barrier();
            cb_push_back(cb0_id, num_rows_per_packet);
        }
    }
}
