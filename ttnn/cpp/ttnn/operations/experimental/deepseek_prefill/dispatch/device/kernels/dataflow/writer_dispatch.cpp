// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#define ENABLE_DISPATCH_DEBUG 0

#if ENABLE_DISPATCH_DEBUG
#define DPRINT_DISPATCH DPRINT
#else
#define DPRINT_DISPATCH \
    if (0)              \
    DebugPrinter()
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-9)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_payload_for_writer_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_metadata_for_writer_id = get_compile_time_arg_val(6);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(7);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(8);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(9);

    // Page counts (indices 10-16)
    constexpr uint32_t input_pages = get_compile_time_arg_val(10);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(11);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(12);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(13);
    constexpr uint32_t output_pages = get_compile_time_arg_val(14);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(15);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(16);

    // Page sizes (indices 17-23)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(21);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(22);
    constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(23);

    // Operation parameters (indices 24-31)
    constexpr uint32_t num_devices = get_compile_time_arg_val(24);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(25);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(26);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(27);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(28);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(29);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(30);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(31);

    // Mesh information (indices 32-36)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(32);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(33);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(34);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(35);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(36);

    // Aligned page sizes (indices 37-43)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(38);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(39);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(40);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(41);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(42);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(43);

    // Fabric configuration (indices 44-47)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(44);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(45);
    constexpr uint32_t num_links = get_compile_time_arg_val(46);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(47);

    // TensorAccessorArgs for all 7 tensors (starting at index 48)
    constexpr auto input_args = TensorAccessorArgs<48>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<output_args.next_compile_time_args_offset()>();
    constexpr auto dispatch_table_args = TensorAccessorArgs<metadata_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t metadata_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatch_table_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatch_core_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_dispatch_cores = get_arg_val<uint32_t>(rt_args_idx++);

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t dispatch_devices = num_devices;
#endif

    DPRINT_DISPATCH << "Writer kernel: dispatch_core=" << dispatch_core_idx << "/" << num_dispatch_cores
                    << " dispatch_devices=" << dispatch_devices << ENDL();

#ifdef DEST_CHIP_ID
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    constexpr std::array<bool, 4> directions = DIRECTIONS;

    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);

    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* sem_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    open_direction_connections_barrier(directions, fabric_connections);

    // Init semaphore exchange
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        num_devices>(fabric_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, dispatch_devices - 1);
    noc_semaphore_set(init_sem_ptr, 0);

    DPRINT_DISPATCH << "Fabric setup complete" << ENDL();
#endif

    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, aligned_output_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, aligned_metadata_page_size);

    // Sentinel-terminated fabric send loop
    while (true) {
        cb_wait_front(cb_route_info_id, 1);
        volatile uint32_t* route_info = (volatile uint32_t*)(get_read_ptr(cb_route_info_id));

        uint32_t route = route_info[0];
        if (route == ROUTE_INFO_SENTINEL) {
            cb_pop_front(cb_route_info_id, 1);
            break;
        }
        uint32_t distance = route_info[1];
        uint32_t page_idx = route_info[2];
        cb_pop_front(cb_route_info_id, 1);

        cb_wait_front(cb_payload_for_writer_id, 1);
        cb_wait_front(cb_metadata_for_writer_id, 1);
        uint32_t payload_addr = get_read_ptr(cb_payload_for_writer_id);
        uint32_t metadata_addr = get_read_ptr(cb_metadata_for_writer_id);

        DPRINT_DISPATCH << "Fabric send: route=" << route << " distance=" << distance << " page_idx=" << page_idx
                        << ENDL();

#ifdef DEST_CHIP_ID
        // Send payload
        fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);
        fabric_send_noc_unicast<fabric_max_packet_size>(
            output_addr_gen,
            fabric_connections[route],
            unicast_packet_header,
            payload_addr,
            page_idx,
            (int)aligned_output_page_size,
            l1_alignment);

        // Send metadata
        fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);
        fabric_send_noc_unicast<fabric_max_packet_size>(
            metadata_addr_gen,
            fabric_connections[route],
            unicast_packet_header,
            metadata_addr,
            page_idx,
            (int)aligned_metadata_page_size,
            l1_alignment);

        noc_async_write_barrier();
#endif

        cb_pop_front(cb_payload_for_writer_id, 1);
        cb_pop_front(cb_metadata_for_writer_id, 1);
    }

#ifdef DEST_CHIP_ID
    // Exit semaphore exchange
    {
        const uint64_t exit_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            axis,
            num_devices>(
            fabric_connections, unicast_packet_header, dest_chip_ids, dest_mesh_ids, exit_noc_semaphore_addr);

        volatile tt_l1_ptr uint32_t* exit_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
        noc_semaphore_wait(exit_sem_ptr, dispatch_devices - 1);
        noc_semaphore_set(exit_sem_ptr, 0);
    }

    close_direction_connections(directions, fabric_connections);
#endif
}
