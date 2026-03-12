// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill dispatch writer kernel (combined metadata+payload variant)
// Handles fabric sends only. Route/distance/page_idx are pre-computed by
// the reader and communicated via cb_route_info. Combined pages (metadata+
// payload) arrive via cb_combined.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "tt_metal/tools/profiler/kernel_profiler.hpp"

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
    // CB IDs (indices 0-4)
    constexpr uint32_t cb_combined_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(4);

    // Page counts (indices 5-10)
    constexpr uint32_t input_pages = get_compile_time_arg_val(5);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(6);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(7);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(8);
    constexpr uint32_t combined_output_pages = get_compile_time_arg_val(9);
    constexpr uint32_t counter_pages = get_compile_time_arg_val(10);

    // Page sizes (indices 11-16)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(11);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(12);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(13);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t combined_output_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t counter_page_size = get_compile_time_arg_val(16);

    // Operation parameters (indices 17-25)
    constexpr uint32_t num_devices = get_compile_time_arg_val(17);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(18);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(19);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(20);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(21);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(22);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(23);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(24);
    constexpr uint32_t padded_metadata_bytes = get_compile_time_arg_val(25);

    // Mesh information (indices 26-30)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(26);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(27);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(28);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(29);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(30);

    // Aligned page sizes (indices 31-36)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(31);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(32);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(33);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_combined_output_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_counter_page_size = get_compile_time_arg_val(36);

    // Fabric configuration (indices 37-40)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(37);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(38);
    constexpr uint32_t num_links = get_compile_time_arg_val(39);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(40);

    // Additional CB IDs (indices 41-42)
    constexpr uint32_t cb_route_info_id = get_compile_time_arg_val(41);
    constexpr uint32_t cb_scratch_id = get_compile_time_arg_val(42);

    constexpr uint32_t combined_cb_page_size = padded_metadata_bytes + aligned_input_page_size;

    // TensorAccessorArgs for 6 tensors (starting at index 43)
    constexpr auto input_args = TensorAccessorArgs<43>();
    constexpr auto indices_args = TensorAccessorArgs<input_args.next_compile_time_args_offset()>();
    constexpr auto weights_args = TensorAccessorArgs<indices_args.next_compile_time_args_offset()>();
    constexpr auto offsets_args = TensorAccessorArgs<weights_args.next_compile_time_args_offset()>();
    constexpr auto combined_output_args = TensorAccessorArgs<offsets_args.next_compile_time_args_offset()>();
    constexpr auto counter_args = TensorAccessorArgs<combined_output_args.next_compile_time_args_offset()>();

    // ===== Runtime Args =====
    size_t rt_args_idx = 0;
    uint32_t input_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t indices_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t weights_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t offsets_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t combined_output_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t counter_tensor_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t cross_device_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t init_semaphore_address = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t token_end_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dispatch_core_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_dispatch_cores = get_arg_val<uint32_t>(rt_args_idx++);
    rt_args_idx += 4;  // skip reader-only args: peer_noc_x, peer_noc_y, batch_ready_sem_id, batch_consumed_sem_id

    DPRINT_DISPATCH << "Writer combined kernel: dispatch_core=" << dispatch_core_idx << "/" << num_dispatch_cores
                    << ENDL();

#ifndef DEST_CHIP_ID
#define DEST_CHIP_ID
#endif

#ifdef DEST_CHIP_ID
    DPRINT_DISPATCH << "Fabric enabled: num_links=" << num_links << " topology=" << (uint32_t)topology << ENDL();

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    constexpr std::array<bool, 4> directions = DIRECTIONS;

    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header =
        reinterpret_cast<volatile tt::tt_fabric::LowLatencyPacketHeader*>(packet_header_buffer_address);

    std::array<std::array<tt::tt_fabric::WorkerToFabricEdmSender, num_links>, 4> fabric_connections;
    {
        DeviceZoneScopedN("dispatch-combined-open-connections");
        for (uint32_t dir = 0; dir < 4; dir++) {
            for (uint32_t link = 0; link < num_links; link++) {
                if (directions[dir]) {
                    fabric_connections[dir][link] =
                        tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(
                            rt_args_idx);
                    fabric_connections[dir][link].open_start();
                }
            }
        }
        for (uint32_t dir = 0; dir < 4; dir++) {
            for (uint32_t link = 0; link < num_links; link++) {
                if (directions[dir]) {
                    fabric_connections[dir][link].open_finish();
                }
            }
        }
    }

    {
        DeviceZoneScopedN("dispatch-combined-init-sync");
        std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> init_connections;
        for (uint32_t dir = 0; dir < 4; dir++) {
            if (directions[dir]) {
                init_connections[dir] = fabric_connections[dir][0];
            }
        }

        auto* sem_packet_header = reinterpret_cast<volatile tt::tt_fabric::LowLatencyPacketHeader*>(
            packet_header_buffer_address + sizeof(tt::tt_fabric::LowLatencyPacketHeader));

        const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            ReplicateGroup::NONE,
            num_devices>(init_connections, sem_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

        for (uint32_t dir = 0; dir < 4; dir++) {
            if (directions[dir]) {
                fabric_connections[dir][0] = init_connections[dir];
            }
        }
    }

    {
        DeviceZoneScopedN("dispatch-combined-wait-init");
        noc_semaphore_wait((uint32_t*)init_semaphore_address, num_devices - 1);
        noc_semaphore_set((uint32_t*)init_semaphore_address, 0);
    }

    DPRINT_DISPATCH << "Fabric setup complete" << ENDL();
#endif

    const auto combined_addr_gen =
        TensorAccessor(combined_output_args, combined_output_address, aligned_combined_output_page_size);

    {
        DeviceZoneScopedN("dispatch-combined-fabric-loop");
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
            uint32_t link = route_info[3];
            cb_pop_front(cb_route_info_id, 1);

            cb_wait_front(cb_combined_id, 1);
            uint32_t combined_read_addr = get_read_ptr(cb_combined_id);

            DPRINT_DISPATCH << "Fabric send: route=" << route << " distance=" << distance << " page_idx=" << page_idx
                            << " link=" << link << ENDL();

            {
                DeviceZoneScopedN("dispatch-combined-fabric");
                fabric_set_unicast_route<false>(
                    (volatile tt_l1_ptr LowLatencyPacketHeader*)unicast_packet_header, distance);

                fabric_send_noc_unicast<fabric_max_packet_size>(
                    combined_addr_gen,
                    fabric_connections[route][link],
                    unicast_packet_header,
                    combined_read_addr,
                    page_idx,
                    (int)aligned_combined_output_page_size,
                    l1_alignment);
            }

            {
                DeviceZoneScopedN("dispatch-combined-write-barrier");
                noc_async_write_barrier();
            }
            cb_pop_front(cb_combined_id, 1);
        }
    }

#ifdef DEST_CHIP_ID
    {
        DeviceZoneScopedN("dispatch-combined-close-connections");
        for (uint32_t dir = 0; dir < 4; dir++) {
            for (uint32_t link = 0; link < num_links; link++) {
                if (directions[dir]) {
                    fabric_connections[dir][link].close();
                }
            }
        }
    }
#endif
}
