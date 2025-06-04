// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include <cstdint>
#include <utility>
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(1));
constexpr uint32_t cb0_id = get_compile_time_arg_val(2);
constexpr uint32_t page_size = get_compile_time_arg_val(3);
constexpr uint32_t row_size = get_compile_time_arg_val(4);
constexpr uint32_t num_packets_per_row = get_compile_time_arg_val(5);
constexpr uint32_t max_packet_size = get_compile_time_arg_val(6);
constexpr bool src_stick_size_is_pow2 = get_compile_time_arg_val(7) == 1;
constexpr uint32_t src_log_base_2_of_page_size = get_compile_time_arg_val(8);

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

    // print every compile and runtime arg in uint32_t
    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "page_size: " << (uint32_t)page_size << "\n";
    DPRINT << "num_packets_per_row: " << (uint32_t)num_packets_per_row << "\n";
    DPRINT << "max_packet_size: " << (uint32_t)max_packet_size << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "row_id_start: " << (uint32_t)row_id_start << "\n";
    DPRINT << "row_id_end: " << (uint32_t)row_id_end << "\n";
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;

#ifdef SHARDED
    typedef ShardedInfo<
        get_compile_time_arg_val(9),
        get_compile_time_arg_val(10),
        get_compile_time_arg_val(11),
        get_compile_time_arg_val(12),
        get_compile_time_arg_val(13),
        get_compile_time_arg_val(14),
        get_compile_time_arg_val(15)>
        tensor_shard_info;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(arg_idx++));
    experimental::ShardedAddrGen<tensor_shard_info> tensor0_addrgen = {
        .bank_base_address = tensor_address0, .shard_array = mapping_table};
#else
    // interleaved addrgen
    const auto tensor0_addrgen = get_interleaved_addr_gen<is_dram, src_stick_size_is_pow2>(
        tensor_address0, row_size, src_log_base_2_of_page_size);
#endif

    uint32_t row_id = row_id_start;
    // uint32_t l1_write_addr_base = get_write_ptr(cb0_id);
    while (row_id < row_id_end) {
        DPRINT << "row_id: " << row_id << "\n";
        cb_reserve_back(cb0_id, 1);
        uint32_t l1_write_addr = get_write_ptr(cb0_id);
        uint64_t noc_src_addr = get_noc_addr(row_id, tensor0_addrgen);
        DPRINT << "noc_src_addr: " << (uint64_t)noc_src_addr << "\n";

        // uint32_t num_pages_to_read = std::min(tile_id_end - tile_id, packet_size_in_pages);
        for (uint32_t j = 0; j < num_packets_per_row; j++) {
            DPRINT << "l1_write_addr : " << (uint32_t)l1_write_addr << " for j " << (uint32_t)j << "\n";
            uint32_t packet_size = std::min(max_packet_size, page_size);
            packet_size = std::min(packet_size, page_size - max_packet_size * j);
            DPRINT << "packet_size: " << (uint32_t)packet_size << "\n";
            noc_async_read(noc_src_addr, l1_write_addr, packet_size);
            noc_async_read_barrier();
            // if (j >=0) {
            //     volatile tt_l1_ptr uint16_t* dst_noc2 = reinterpret_cast<volatile tt_l1_ptr
            //     uint16_t*>(l1_write_addr); for (uint16_t value = 0; value < 32; value++) {
            //         DPRINT << "value at " << (uint16_t)value << " is: " << BF16((uint16_t)dst_noc2[value]) << ENDL();
            //     }
            // }

            l1_write_addr += packet_size;
            noc_src_addr += packet_size;  // advance the noc address for the next packet
        }
        row_id++;
        noc_async_read_barrier();
        cb_push_back(cb0_id, 1);
    }

    DPRINT << "DONE \n";
}
