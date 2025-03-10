// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

inline void print_tiles(uint32_t cb_id, uint32_t tile_start = 0, uint32_t num_tiles = 1, bool untilize = false) {
    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
        print_full_tile(cb_id, tile_start + tile_idx, untilize);
    }
}

void kernel_main() {
    DPRINT << "Starting kernel_main for writer" << ENDL();
    size_t ct_arg_idx = 0, rt_arg_idx = 0;
    constexpr uint32_t output_tensor_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t fabric_sender_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t fabric_receiver_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t packet_header_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t chip_id = get_compile_time_arg_val(4);
    constexpr uint32_t tiles_per_core_width = get_compile_time_arg_val(5);
    constexpr uint32_t cores_per_device = get_compile_time_arg_val(6);
    constexpr uint32_t num_devices = get_compile_time_arg_val(7);
    constexpr uint32_t page_size_bytes = get_compile_time_arg_val(8);

    constexpr uint32_t device_order[num_devices - 1] =
        DEVICE_ORDER;  // this is code gen'd in the program factory using the defines

    typedef ShardedInfo<
        get_compile_time_arg_val(9),   // Memory layout
        get_compile_time_arg_val(10),  // The number of sharding cores
        get_compile_time_arg_val(11),  // The page size we offset each write to
        get_compile_time_arg_val(12),  // The number of pages in each sharding row not including padding pages
        get_compile_time_arg_val(13),  // This defines times when contiguous pages can't be calculated
        get_compile_time_arg_val(14),  // pages_per_shard_x
        get_compile_time_arg_val(15)>  // pages_per_shard_y
        tensor_shard_info;

    const auto [mapping_table, rt_increment] =
        experimental::shard_addr_gen_utils::get_shard_map<tensor_shard_info>(get_arg_addr(rt_arg_idx));
    experimental::ShardedAddrGen<tensor_shard_info> s0 = {
        .bank_base_address = get_read_ptr(output_tensor_cb_id), .shard_array = mapping_table};
    rt_arg_idx += rt_increment;

    auto packet_header_buffer_addr_forward = get_write_ptr(packet_header_cb_id);
    auto* unicast_packet_header = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    auto* sem_inc_packet_header =
        reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward + sizeof(PACKET_HEADER_TYPE));

    auto fabric_connection = FabricConnectionManager::build_from_args(rt_arg_idx);
    DPRINT << "Fabric connection is logically connected: " << (uint32_t)fabric_connection.is_logically_connected()
           << ENDL();
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }
    DPRINT << "Fabric connection opened " << ENDL();
    for (auto target_device_id : device_order) {
        if (target_device_id == chip_id) {
            break;
        }
        uint32_t num_hops = std::abs(int(target_device_id) - int(chip_id));
        unicast_packet_header->to_chip_unicast(static_cast<uint8_t>(num_hops));
        auto& fabric_conn = target_device_id > chip_id ? fabric_connection.get_forward_connection()
                                                       : fabric_connection.get_backward_connection();

        for (uint32_t core_id = 0; core_id < cores_per_device; ++core_id) {
            DPRINT << "Waiting for " << tiles_per_core_width << " tiles from fabric sender cb" << ENDL();
            cb_wait_front(fabric_sender_cb_id, tiles_per_core_width);
            DPRINT << "Got " << tiles_per_core_width << " tiles from fabric sender cb" << ENDL();
            auto base_l1_read_addr = get_write_ptr(fabric_sender_cb_id);
            print_tiles(fabric_sender_cb_id, 0, tiles_per_core_width, true);
            // iterate through all the tiles_per_core_width

            for (uint32_t page_id = 0; page_id < tiles_per_core_width; ++page_id) {
                uint32_t tile_id = page_id + core_id * tiles_per_core_width;
                uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, s0, 0 /*offset*/, 0 /*noc_id*/);
                unicast_packet_header->to_noc_unicast_write(
                    tt::fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, page_size_bytes);

                uint32_t l1_read_addr = base_l1_read_addr + page_id * page_size_bytes;
                fabric_conn.wait_for_empty_write_slot();
                fabric_conn.send_payload_without_header_non_blocking_from_address(l1_read_addr, page_size_bytes);
                fabric_conn.send_payload_flush_blocking_from_address(
                    (uint32_t)unicast_packet_header, sizeof(PACKET_HEADER_TYPE));
                noc_async_writes_flushed();  // flushed because cross chip?
            }

            cb_pop_front(fabric_sender_cb_id, tiles_per_core_width);
            // increment semaphore here?
        }

        if (fabric_connection.is_logically_connected()) {
            fabric_connection.close();
        }
    }
}
