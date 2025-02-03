// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// NOTE: This should ideally be merged with `ccl_send_reader` when we are able to support compile time args
//       that don't require macros to function

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include <tt-metalium/buffer_constants.hpp>
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/all_gather/device/kernels/dataflow/worker_ring_gather_utils.hpp"

#include "cpp/ttnn/operations/ccl/common/kernels/command_processor.hpp"

#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "cpp/ttnn/operations/ccl/kernels/edm_fabric/fabric_edm_packet_header.hpp"

#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/io_descriptors.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "cpp/ttnn/tensor/enum_types.hpp"
#include <cstdint>
#include <utility>

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr uint32_t cb0_id = get_compile_time_arg_val(3);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(4);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(5);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(7);

FORCE_INLINE void write_and_advance_local_read_address_for_fabric_write(
    uint64_t noc0_dest_noc_addr,
    size_t packet_header_buffer_addr,
    uint32_t num_targets_forward_direction,
    uint32_t num_targets_backward_direction,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    uint32_t payload_size_bytes) {
    const auto [dest_noc_xy, dest_addr] = get_noc_address_components(noc0_dest_noc_addr);
    const size_t payload_l1_address = l1_read_addr;

    auto pkt_hdr = reinterpret_cast<volatile tt::fabric::PacketHeader*>(packet_header_buffer_addr);
#ifdef DEBUG_PRINT_ENABLED
    pkt_hdr->reserved2 = my_chip_id;
#endif

    size_t packet_send_size_bytes = payload_size_bytes + sizeof(tt::fabric::PacketHeader);
    pkt_hdr->to_write()->to_noc_unicast(tt::fabric::NocUnicastCommandHeader{
        dest_addr, packet_send_size_bytes, static_cast<uint8_t>(dest_noc_xy.x), static_cast<uint8_t>(dest_noc_xy.y)});

    noc_async_write(payload_l1_address, safe_get_noc_addr(dest_noc_xy.x, dest_noc_xy.y, dest_addr), payload_size_bytes);
    if (fabric_connection.has_forward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        fabric_connection.get_forward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            (uint32_t)pkt_hdr, sizeof(tt::fabric::PacketHeader));
    }

    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_without_header_non_blocking_from_address(
            l1_read_addr, payload_size_bytes);
        fabric_connection.get_backward_connection().send_payload_flush_blocking_from_address(
            (uint32_t)pkt_hdr, sizeof(tt::fabric::PacketHeader));
    }

    l1_read_addr += payload_size_bytes;
}

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
    uint32_t num_tiles_per_core = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_to_read = get_arg_val<uint32_t>(arg_idx++);
    uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
    arg_idx += num_cores;
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "reserved_packet_header_cb_id: " << (uint32_t)reserved_packet_header_cb_id << "\n";
    DPRINT << "num_packet_headers_storable: " << (uint32_t)num_packet_headers_storable << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << "num_targets_forward_direction: " << (uint32_t)num_targets_forward_direction << "\n";
    DPRINT << "num_targets_backward_direction: " << (uint32_t)num_targets_backward_direction << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "num_tiles_per_core: " << (uint32_t)num_tiles_per_core << "\n";
    DPRINT << "num_tiles_to_read: " << (uint32_t)num_tiles_to_read << "\n";
    DPRINT << "first_core_tile_start_offset: " << (uint32_t)first_core_tile_start_offset << "\n";
    DPRINT << "num_cores: " << (uint32_t)num_cores << "\n";
    for (uint32_t i = 0; i < num_cores; i++) {
        DPRINT << "core_noc_x[" << i << "]: " << (uint32_t)core_noc_x[i] << "\n";
        DPRINT << "core_noc_y[" << i << "]: " << (uint32_t)core_noc_y[i] << "\n";
    }
    DPRINT << "wait_output_semaphore: " << (uint32_t)wait_output_semaphore << "\n";
    DPRINT << "reset_global_semaphore: " << (uint32_t)reset_global_semaphore << "\n";
    DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << "\n";
    DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << "\n";
    DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << "\n";
    DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << "\n";

    DPRINT << "arg_for_fab: " << (uint32_t)arg_for_fab << "\n";
    DPRINT << "fabric_connection arg 0" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 1" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 2" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 3" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 4" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, num_packet_headers_storable);
    auto packet_header_buffer_addr = get_write_ptr(reserved_packet_header_cb_id);

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    // 1. mcast via fabric to remote tensor addresses
    uint32_t tiles_read = 0;
    uint32_t shard_tile_id = 0;  // first_core_tile_start_offset;
    uint32_t core_id = 0;
    uint32_t writer_chip_offset = my_chip_id * num_tiles_per_core * tensor0_page_size;

    while (tiles_read < num_tiles_to_read) {
        DPRINT << "tiles_read: " << tiles_read << "\n";
        uint32_t num_tiles_to_read_this_core = std::min(num_tiles_per_core - shard_tile_id, packet_size_in_pages);
        num_tiles_to_read_this_core = 1;  // std::min(num_tiles_to_read-tiles_read, num_tiles_to_read_this_core);
        cb_wait_front(cb0_id, num_tiles_to_read_this_core);
        size_t l1_read_addr = get_read_ptr(cb0_id);

        uint64_t noc0_dest_noc_addr =
            get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0 + writer_chip_offset, 0 /*noc_id*/);

        // Offset the writer chip offset
        // noc0_dest_noc_addr +=  writer_chip_offset;

        DPRINT << "core_noc_x[core_id]: " << (uint32_t)core_noc_x[core_id] << "\n";
        DPRINT << "core_noc_y[core_id]: " << (uint32_t)core_noc_y[core_id] << "\n";
        DPRINT << "noc0_dest_noc_addr_base: " << noc0_dest_noc_addr << "\n";
        noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;

        DPRINT << "core_id: " << core_id << "\n";
        DPRINT << "num_tiles_to_read_this_core: " << num_tiles_to_read_this_core << "\n";
        DPRINT << "noc0_dest_noc_addr: " << noc0_dest_noc_addr << "\n";
        DPRINT << "shard_tile_id: " << shard_tile_id << "\n";

        write_and_advance_local_read_address_for_fabric_write(
            noc0_dest_noc_addr,
            packet_header_buffer_addr,
            num_targets_forward_direction,
            num_targets_backward_direction,
            fabric_connection,
            l1_read_addr,
            num_tiles_to_read_this_core * tensor0_page_size);
        noc_async_writes_flushed();

        cb_pop_front(cb0_id, num_tiles_to_read_this_core);
        tiles_read += num_tiles_to_read_this_core;
        shard_tile_id += num_tiles_to_read_this_core;
        if (shard_tile_id >= num_tiles_per_core) {
            shard_tile_id = 0;
            core_id++;
        }
    }

    // 2. mcast output ready semaphore
    auto* pkt_hdr = reinterpret_cast<tt::fabric::PacketHeader*>(packet_header_buffer_addr);
    pkt_hdr->to_atomic_inc();
    pkt_hdr->to_noc_unicast_atomic_inc(tt::fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_bank_addr,
        static_cast<uint16_t>(1),  // increment 1
        32,
        static_cast<uint8_t>(out_ready_sem_noc0_x),
        static_cast<uint8_t>(out_ready_sem_noc0_y)});
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr->to_chip_multicast(
            tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_addr, sizeof(tt::fabric::PacketHeader));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            packet_header_buffer_addr, sizeof(tt::fabric::PacketHeader));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);
    DPRINT << "inc done\n";

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        while (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr) < out_ready_sem_wait_value);
        DPRINT << "waitval done\n";
    }

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem_bank_addr);
        noc_inline_dw_write(dest_noc_addr, 0);
        DPRINT << "reset done\n";
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
    DPRINT << "DONE \n";
}
