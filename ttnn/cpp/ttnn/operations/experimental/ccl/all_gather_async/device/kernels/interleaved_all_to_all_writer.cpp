// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

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

constexpr uint32_t num_max_targets = std::max(num_targets_forward_direction, num_targets_backward_direction);
constexpr uint32_t num_sync_targets_forward = dynamic_alternate ? num_max_targets : num_targets_forward_direction;
constexpr uint32_t num_sync_targets_backward = dynamic_alternate ? num_max_targets : num_targets_backward_direction;

constexpr uint32_t wait_sem_value = (ring_size - 1);

constexpr uint32_t N_DRAM_BANKS = 12;
constexpr uint32_t NUM_SENDERS = ring_size - 1;

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
    address_t intermediate_buffer_addr = get_arg_val<address_t>(arg_idx++);
    address_t output_buffer_addr = get_arg_val<address_t>(arg_idx++);
    uint32_t out_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_row_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_col_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_row_tiles = get_arg_val<uint32_t>(arg_idx++);
    uint32_t input_shard_col_tiles = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t global_semaphore_addr = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* receiver_cores_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += ring_size;
    tt_l1_ptr uint32_t* receiver_cores_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += ring_size;
    size_t arg_for_fab = arg_idx;
    DPRINT << "arg_for_fab: " << (uint32_t)arg_for_fab << "\n";
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    uint32_t out_row_end = out_row_start + input_shard_row_tiles;
    uint32_t out_col_end = out_col_start + input_shard_col_tiles;

    // DPRINT << "ct args: \n";
    // DPRINT << "my_ring_id: " << (uint32_t)my_ring_id << "\n";
    // DPRINT << "reserved_packet_header_cb_id: " << (uint32_t)reserved_packet_header_cb_id << "\n";
    // DPRINT << "num_packet_headers_storable: " << (uint32_t)num_packet_headers_storable << "\n";
    // DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    // DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    // DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    // DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    // DPRINT << "num_targets_forward_direction: " << (uint32_t)num_targets_forward_direction << "\n";
    // DPRINT << "num_targets_backward_direction: " << (uint32_t)num_targets_backward_direction << "\n";

    // DPRINT << "rt args: \n";
    // DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";

    // DPRINT << "wait_output_semaphore: " << (uint32_t)wait_output_semaphore << "\n";
    // DPRINT << "reset_global_semaphore: " << (uint32_t)reset_global_semaphore << "\n";
    // DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << "\n";
    // DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << "\n";
    // DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << "\n";
    // DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << "\n";

    // DPRINT << "arg_for_fab: " << (uint32_t)arg_for_fab << "\n";
    // DPRINT << "fabric_connection arg 0" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    // DPRINT << "fabric_connection arg 1" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    // DPRINT << "fabric_connection arg 2" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    // DPRINT << "fabric_connection arg 3" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    // DPRINT << "fabric_connection arg 4" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";

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
    // DPRINT << "packet_header_buffer_addr_forward: " << (uint32_t)packet_header_buffer_addr_forward << "\n";
    // DPRINT << "packet_header_buffer_addr_backward: " << (uint32_t)packet_header_buffer_addr_backward << "\n";
    // DPRINT << "packet_header_buffer_seminc: " << (uint32_t)packet_header_buffer_seminc << "\n";

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

    DPRINT << "fabric_connection.is_logically_connected(): " << (uint32_t)fabric_connection.is_logically_connected()
           << "\n";
    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    DPRINT << "device " << (uint32_t)my_ring_id << " opened fabric\n";

    // 1. mcast via fabric to remote tensor addresses
    // DPRINT << "num_targets_forward_direction: " << num_targets_forward_direction << "\n";
    // DPRINT << "num_targets_backward_direction: " << num_targets_backward_direction << "\n";
    // DPRINT << "my_ring_id: " << my_ring_id << "\n";

    // DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";
    // DPRINT << "packet size in pages: " << (uint32_t)packet_size_in_pages << "\n";

    bool cur_is_forward = num_targets_forward_direction > num_targets_backward_direction;
    uint32_t forward_hops = 1;
    uint32_t backward_hops = 1;
    uint32_t dst_ring_id;
    for (uint32_t i = 0; i < ring_size - 1; ++i) {
        DeviceZoneScopedN("WriteToDevice");
        if (forward_hops == num_targets_forward_direction + 1) {
            cur_is_forward = false;
        }
        if (backward_hops == num_targets_backward_direction + 1) {
            cur_is_forward = true;
        }
        if (cur_is_forward) {
            dst_ring_id = (my_ring_id + forward_hops) % ring_size;
            pkt_hdr_forward->to_chip_unicast(forward_hops);
        } else {
            dst_ring_id = (my_ring_id - backward_hops + ring_size) % ring_size;
            pkt_hdr_backward->to_chip_unicast(backward_hops);
        }

        DPRINT << "from device " << (uint32_t)my_ring_id << " to device " << (uint32_t)dst_ring_id << "\n";
        // TODO: Why do I pass a list of receiver cores if each device only needs to know about its own receiver core?
        const uint32_t receiver_core_x = receiver_cores_x[my_ring_id];
        const uint32_t receiver_core_y = receiver_cores_y[my_ring_id];

        const uint32_t my_relative_ring_id = (my_ring_id < dst_ring_id) ? my_ring_id : my_ring_id - 1;
        uint32_t packet_id = 0;
        /*
        global_id = relative_id + tile_count * N_SENDERS
        first_id = (global_id % 12) + 24 * (global_id / 12)
        second_id = first_id + 12
        */

        for (uint32_t out_row_id = out_row_start; out_row_id < out_row_end; out_row_id++) {
            for (uint32_t out_col_id = out_col_start; out_col_id < out_col_end; out_col_id += packet_size_in_pages) {
                cb_wait_front(cb0_id, packet_size_in_pages);
                size_t l1_read_addr = get_read_ptr(cb0_id);
                uint32_t num_pages_to_read = std::min(out_col_end - out_col_id, packet_size_in_pages);

                constexpr uint32_t contig_pages_advanced = 2;  // TODO: CT arg

                constexpr uint32_t payload_size_bytes = contig_pages_advanced * tensor0_page_size;
                for (uint32_t j = 0; j < num_pages_to_read; j += contig_pages_advanced) {
                    // Calculate the tile id of the first tile in the pair to send. Guaranteed to be in the same bank.
                    uint32_t global_id = my_relative_ring_id + packet_id * NUM_SENDERS;
                    uint32_t first_id = (global_id % N_DRAM_BANKS) + 2 * N_DRAM_BANKS * (global_id / N_DRAM_BANKS);
                    DPRINT << "Writing tile pair (" << first_id << ", " << (first_id + 12) << ")" << ENDL();

                    uint64_t noc0_dest_noc_addr =
                        get_noc_addr(first_id, intermediate_tensor_addrgen, 0 /*offset*/, 0 /*noc_id*/);

                    if (cur_is_forward) {
                        pkt_hdr_forward->to_noc_unicast_write(
                            tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
                        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
                        fabric_connection.get_forward_connection()
                            .send_payload_without_header_non_blocking_from_address(l1_read_addr, payload_size_bytes);
                        fabric_connection.get_forward_connection().send_payload_flush_non_blocking_from_address(
                            (uint32_t)pkt_hdr_forward, sizeof(PACKET_HEADER_TYPE));
                    } else {
                        pkt_hdr_backward->to_noc_unicast_write(
                            tt::tt_fabric::NocUnicastCommandHeader{noc0_dest_noc_addr}, payload_size_bytes);
                        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
                        fabric_connection.get_backward_connection()
                            .send_payload_without_header_non_blocking_from_address(l1_read_addr, payload_size_bytes);
                        fabric_connection.get_backward_connection().send_payload_flush_non_blocking_from_address(
                            (uint32_t)pkt_hdr_backward, sizeof(PACKET_HEADER_TYPE));
                    }

                    noc_async_writes_flushed();

                    // Advance local read address
                    l1_read_addr += payload_size_bytes;
                    packet_id++;
                }

                cb_pop_front(cb0_id, packet_size_in_pages);
            }
        }

        // Unicast semaphore increment to receiver core of receiver device
        uint64_t output_semaphore_noc_addr_in_pkt =
            safe_get_noc_addr(receiver_core_x, receiver_core_y, global_semaphore_addr, 0);
        volatile PACKET_HEADER_TYPE* pkt_hdr =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
        pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            output_semaphore_noc_addr_in_pkt,
            static_cast<uint16_t>(1),  // increment 1
            32});
        // Write the packet
        if (cur_is_forward) {
            fabric_connection.get_forward_connection().wait_for_empty_write_slot();
            pkt_hdr->to_chip_unicast(forward_hops);
            fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));

            // Increment forward_hops for next iteration
            forward_hops++;
        } else {
            pkt_hdr->to_chip_unicast(backward_hops);
            fabric_connection.get_backward_connection().wait_for_empty_write_slot();
            fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
                packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));

            backward_hops++;
        }
    }

    // // 3. wait for mcast output ready semaphore
    // if (wait_output_semaphore) {
    //     DPRINT << "waiting for waitval " << (uint32_t)wait_sem_value << "\n";
    //     while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr) < wait_sem_value);

    //     DPRINT << "waitval done\n";
    // }

    // // 4. global semaphore reset
    // if (reset_global_semaphore) {
    //     *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(global_semaphore_addr) = 0;
    //     DPRINT << "reset done\n";
    // }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();

    DPRINT << "DONE \n";
}
