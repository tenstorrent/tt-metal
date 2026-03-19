// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#define ENABLE_COMBINE_DEBUG 0
#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE DPRINT
#else
#define DPRINT_COMBINE \
    if (0)             \
    DebugPrinter()
#endif

constexpr uint32_t ROUTE_INFO_SENTINEL = 0xFFFFFFFF;

void kernel_main() {
    using namespace ttnn::operations::ccl::common;

    // ===== Compile Time Args =====
    // CB IDs (indices 0-5)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_output_for_writer_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(5);

    // Page counts (indices 6-9)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(6);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(7);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(8);
    constexpr uint32_t output_pages = get_compile_time_arg_val(9);

    // Page sizes (indices 10-13)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(13);

    // Operation parameters (indices 14-18)
    constexpr uint32_t num_chips = get_compile_time_arg_val(14);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(15);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(16);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(17);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(18);

    // Hidden dimension (index 19)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(19);

    // Aligned page sizes (indices 20-23)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(21);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(22);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(23);

    // Mesh information (indices 24-28)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(24);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(25);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(26);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(27);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(28);

    // Fabric configuration (indices 29-32)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(29);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(30);
    constexpr uint32_t num_links = get_compile_time_arg_val(31);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(32);

    // TensorAccessorArgs for all 4 tensors (starting at index 33)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<33>();
    constexpr auto dispatched_metadata_args =
        TensorAccessorArgs<dispatched_buffer_args.next_compile_time_args_offset()>();
    constexpr auto experts_tok_counter_args =
        TensorAccessorArgs<dispatched_metadata_args.next_compile_time_args_offset()>();
    constexpr auto output_args = TensorAccessorArgs<experts_tok_counter_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    size_t rt_args_idx = 0;
    uint32_t dispatched_buffer_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatched_metadata_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t experts_tok_counter_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t output_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t zero_init_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t zero_init_barrier_semaphore_id = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_cores = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t zero_init_semaphore_address = get_semaphore(zero_init_semaphore_id);
    uint32_t zero_init_barrier_l1_offset = get_semaphore(zero_init_barrier_semaphore_id);

    // Read NOC coordinates for all cores (for inter-core barrier signaling)
    uint64_t all_core_barrier_noc_addrs[2];
    for (uint32_t c = 0; c < num_cores; c++) {
        uint32_t noc_x = get_arg_val<uint32_t>(rt_args_idx++);
        uint32_t noc_y = get_arg_val<uint32_t>(rt_args_idx++);
        all_core_barrier_noc_addrs[c] = get_noc_addr(noc_x, noc_y, zero_init_barrier_l1_offset);
    }

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t combine_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t combine_devices = num_chips;
#endif

    DPRINT_COMBINE << "Combine Writer: experts=[" << expert_start_idx << "," << expert_end_idx << ")"
                   << " linearized_mesh_coord=" << linearized_mesh_coord << ENDL();

    // Wait for reader to complete zero-init
    volatile tt_l1_ptr uint32_t* zero_init_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_init_semaphore_address);
    noc_semaphore_wait(zero_init_sem_ptr, 1);

#ifdef DEST_CHIP_ID
    constexpr uint32_t total_mesh_devices = mesh_rows * mesh_cols;
    constexpr uint8_t dest_chip_ids[total_mesh_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[total_mesh_devices] = DEST_MESH_ID;
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
        total_mesh_devices>(
        fabric_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, combine_devices - 1);
    noc_semaphore_set(init_sem_ptr, 0);

    DPRINT_COMBINE << "Fabric setup complete" << ENDL();
#endif

    // Signal ALL readers that global init exchange is done.
    // Each writer increments every reader's barrier sem so each reader
    // collects num_cores signals before proceeding.
    for (uint32_t c = 0; c < num_cores; c++) {
        noc_semaphore_inc(all_core_barrier_noc_addrs[c], 1);
    }
    noc_async_write_barrier();

    const auto output_addr_gen = TensorAccessor(output_args, output_addr, aligned_output_page_size);

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
        uint32_t output_page_idx = route_info[2];
        cb_pop_front(cb_route_info_id, 1);

        cb_wait_front(cb_output_for_writer_id, 1);
        uint32_t output_data_addr = get_read_ptr(cb_output_for_writer_id);

        DPRINT_COMBINE << "Fabric send: route=" << route << " distance=" << distance << " page_idx=" << output_page_idx
                       << ENDL();

#ifdef DEST_CHIP_ID
        fabric_set_unicast_route<false>((volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);
        fabric_send_noc_unicast<fabric_max_packet_size>(
            output_addr_gen,
            fabric_connections[route],
            unicast_packet_header,
            output_data_addr,
            output_page_idx,
            (int)aligned_output_page_size,
            l1_alignment);

        noc_async_write_barrier();
#endif

        cb_pop_front(cb_output_for_writer_id, 1);
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
            total_mesh_devices>(
            fabric_connections, unicast_packet_header, dest_chip_ids, dest_mesh_ids, exit_noc_semaphore_addr);

        volatile tt_l1_ptr uint32_t* exit_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
        noc_semaphore_wait(exit_sem_ptr, combine_devices - 1);
        noc_semaphore_set(exit_sem_ptr, 0);
    }

    close_direction_connections(directions, fabric_connections);
#endif
}
