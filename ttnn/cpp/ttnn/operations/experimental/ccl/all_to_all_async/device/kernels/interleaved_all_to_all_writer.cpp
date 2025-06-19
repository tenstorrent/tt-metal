// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/minimal_ccl_common.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_ring_id = get_compile_time_arg_val(0);
constexpr uint32_t ring_size = get_compile_time_arg_val(1);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(2);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(3);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(4));
constexpr uint32_t cb0_id = get_compile_time_arg_val(5);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(6);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(8);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(9);
constexpr bool dynamic_alternate = get_compile_time_arg_val(10);
constexpr uint32_t chunk_granularity = get_compile_time_arg_val(11);
constexpr uint32_t contig_pages_advanced = get_compile_time_arg_val(12);
constexpr uint32_t N_DRAM_BANKS = get_compile_time_arg_val(13);

constexpr uint32_t num_max_targets = std::max(num_targets_forward_direction, num_targets_backward_direction);
constexpr uint32_t num_sync_targets_forward = dynamic_alternate ? num_max_targets : num_targets_forward_direction;
constexpr uint32_t num_sync_targets_backward = dynamic_alternate ? num_max_targets : num_targets_backward_direction;

constexpr uint32_t wait_sem_value = (ring_size - 1);

constexpr uint32_t NUM_SENDERS = ring_size - 1;

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    address_t intermediate_buffer_addr = get_arg_val<address_t>(arg_idx++);
    address_t output_buffer_addr = get_arg_val<address_t>(arg_idx++);
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_row_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_col_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_core_x = get_arg_val<uint32_t>(arg_idx++);
    uint32_t receiver_core_y = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;

    auto fabric_connection = FabricConnectionManager::build_from_args<
        FabricConnectionManager::BuildFromArgsMode::BUILD_AND_OPEN_CONNECTION_START_ONLY>(arg_idx);

    uint32_t out_row_end = out_row_start + input_shard_row_tiles;
    uint32_t out_col_end = out_col_start + input_shard_col_tiles;

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto intermediate_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = intermediate_buffer_addr,
        .page_size = tensor0_page_size,
        .data_format = get_dataformat(cb0_id)};
    auto output_tensor_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = output_buffer_addr, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open_finish();
    }

    // Set up seminc packet
    uint64_t output_semaphore_noc_addr_in_pkt =
        safe_get_noc_addr(receiver_core_x, receiver_core_y, global_semaphore_addr, 0);
    volatile PACKET_HEADER_TYPE* pkt_hdr_seminc =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);

    pkt_hdr_seminc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        output_semaphore_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});

    volatile PACKET_HEADER_TYPE* cur_pkt_header;

    bool cur_is_forward = num_targets_forward_direction > num_targets_backward_direction;
    uint32_t forward_hops = 1;
    uint32_t backward_hops = 1;
    for (uint32_t i = 0; i < ring_size - 1; ++i) {
        if (forward_hops == num_targets_forward_direction + 1) {
            cur_is_forward = false;
        }
        if (backward_hops == num_targets_backward_direction + 1) {
            cur_is_forward = true;
        }
        uint32_t& cur_hops = cur_is_forward ? forward_hops : backward_hops;
        uint32_t dst_ring_id =
            cur_is_forward ? (my_ring_id + cur_hops) % ring_size : (my_ring_id - cur_hops + ring_size) % ring_size;
        tt::tt_fabric::WorkerToFabricEdmSender& cur_connection =
            cur_is_forward ? fabric_connection.get_forward_connection() : fabric_connection.get_backward_connection();
        cur_pkt_header = cur_is_forward ? pkt_hdr_forward : pkt_hdr_backward;
        cur_pkt_header->to_chip_unicast(cur_hops);

        const uint32_t my_relative_ring_id = (my_ring_id < dst_ring_id) ? my_ring_id : my_ring_id - 1;
        uint32_t packet_id = 0;

        uint32_t prev_chunk_id = 0;

        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += packet_size_in_pages) {
                cb_wait_front(cb0_id, packet_size_in_pages);
                size_t l1_read_addr = get_read_ptr(cb0_id);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, packet_size_in_pages);

                constexpr uint32_t payload_size_bytes = contig_pages_advanced * tensor0_page_size;

                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    // Calculate the tile id of the first tile in the pair to send. Guaranteed to be in the same bank.
                    uint32_t global_id = my_relative_ring_id + packet_id * NUM_SENDERS;
                    uint32_t first_id =
                        (global_id % N_DRAM_BANKS) + contig_pages_advanced * N_DRAM_BANKS * (global_id / N_DRAM_BANKS);

                    packet_id++;  // increment packet_id for chunk calculation

                    uint64_t noc0_dest_noc_addr =
                        get_noc_addr(first_id, intermediate_tensor_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    uint32_t current_chunk_id = packet_id / chunk_granularity;
                    if (current_chunk_id != prev_chunk_id) {
                        // Fused payload write with atomic inc
                        cur_pkt_header->to_noc_fused_unicast_write_atomic_inc(
                            tt::tt_fabric::NocUnicastAtomicIncFusedCommandHeader{
                                noc0_dest_noc_addr, output_semaphore_noc_addr_in_pkt, 1, 32, false},
                            payload_size_bytes);
                        cur_connection.wait_for_empty_write_slot();
                        cur_connection.send_payload_without_header_non_blocking_from_address(
                            l1_read_addr, payload_size_bytes);
                        cur_connection.send_payload_flush_non_blocking_from_address(
                            (uint32_t)cur_pkt_header, sizeof(PACKET_HEADER_TYPE));

                        prev_chunk_id = current_chunk_id;
                    } else {
                        // Unicast payload write
                        cur_pkt_header->to_noc_unicast_write(
                            tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
                        cur_connection.wait_for_empty_write_slot();
                        cur_connection.send_payload_without_header_non_blocking_from_address(
                            l1_read_addr, payload_size_bytes);
                        cur_connection.send_payload_flush_non_blocking_from_address(
                            (uint32_t)cur_pkt_header, sizeof(PACKET_HEADER_TYPE));
                    }

                    noc_async_writes_flushed();

                    // Advance local read address
                    l1_read_addr += payload_size_bytes;
                }

                cb_pop_front(cb0_id, packet_size_in_pages);
            }
        }

        // Handle final incomplete chunk
        if (packet_id % chunk_granularity != 0) {
            pkt_hdr_seminc->to_chip_unicast(cur_hops);
            cur_connection.wait_for_empty_write_slot();
            cur_connection.send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
        }

        cur_hops++;
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
}
