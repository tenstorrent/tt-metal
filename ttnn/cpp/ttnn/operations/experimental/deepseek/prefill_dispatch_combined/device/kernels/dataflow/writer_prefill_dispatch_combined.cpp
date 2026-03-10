// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill dispatch writer kernel (combined metadata+payload variant)
// Metadata is written in-place at offset 0 of the combined CB page (payload
// sits at padded_metadata_bytes). A single DRAM write or fabric send
// transfers both metadata and payload together.

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

    constexpr uint32_t combined_cb_page_size = padded_metadata_bytes + aligned_input_page_size;

    // TensorAccessorArgs for 6 tensors (starting at index 41)
    constexpr auto input_args = TensorAccessorArgs<41>();
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
    uint32_t expert_start_idx = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t expert_end_idx = get_arg_val<uint32_t>(rt_args_idx++);

    DPRINT_DISPATCH << "Writer combined kernel: tokens=[" << token_start_idx << "," << token_end_idx << ")"
                    << " experts=[" << expert_start_idx << "," << expert_end_idx << ")"
                    << " padded_metadata_bytes=" << padded_metadata_bytes << ENDL();

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

        // Use a second packet header slot for semaphore sends
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

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;
    constexpr uint32_t dispatch_index = axis == ReplicateGroup::COLS ? row : col;
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (axis == ReplicateGroup::COLS) ? (col + mesh_rows * mesh_cols) : (row * mesh_cols + mesh_cols);
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t dispatch_devices = num_devices;
    constexpr uint32_t dispatch_index = linearized_mesh_coord;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_end_idx = num_devices;
    constexpr uint32_t device_stride = 1;
#endif

    {
        DeviceZoneScopedN("dispatch-combined-wait-init");
        noc_semaphore_wait((uint32_t*)init_semaphore_address, num_devices - 1);
        noc_semaphore_set((uint32_t*)init_semaphore_address, 0);
    }

    uint32_t fabric_send_counter = 0;
    DPRINT_DISPATCH << "Fabric setup complete" << ENDL();
#endif

    // Wait for offsets
    cb_wait_front(cb_offsets_id, offsets_pages);
    int32_t* offsets = (int32_t*)(get_read_ptr(cb_offsets_id));

    // TensorAccessor for the combined output buffer in DRAM
    const auto combined_addr_gen =
        TensorAccessor(combined_output_args, combined_output_address, aligned_combined_output_page_size);

    {
        DeviceZoneScopedN("dispatch-combined-token-loop");
        for (uint32_t token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
            DPRINT_DISPATCH << "Processing token_idx: " << token_idx << ENDL();

            {
                DeviceZoneScopedN("dispatch-combined-wait-reader");
                cb_wait_front(cb_indices_id, 1);
                cb_wait_front(cb_weights_id, 1);
                cb_wait_front(cb_combined_id, 1);
            }

            // The combined CB page layout: [metadata_area | payload]
            // metadata_area: padded_metadata_bytes (writer fills in metadata here)
            // payload: aligned_input_page_size bytes (reader already wrote input here)
            uint32_t combined_read_addr = get_read_ptr(cb_combined_id);
            int32_t* indices = (int32_t*)(get_read_ptr(cb_indices_id));
            uint16_t* weights = (uint16_t*)(get_read_ptr(cb_weights_id));

            for (uint32_t k = 0; k < num_experts_per_tok; ++k) {
                auto routed_expert = indices[k];
                if ((uint32_t)routed_expert < expert_start_idx || (uint32_t)routed_expert >= expert_end_idx) {
                    continue;
                }
                auto expert_chip = routed_expert / experts_per_chip;
                auto expert_index_within_chip = routed_expert % experts_per_chip;

                DPRINT_DISPATCH << "  Expert [" << k << "]=" << routed_expert << " (chip=" << expert_chip << ")"
                                << ENDL();

                auto& offset = offsets[routed_expert];
                auto page_idx = expert_index_within_chip * max_dispatched_tokens_per_expert + offset;

                // Write metadata in-place at the start of the combined CB page
                volatile tt_l1_ptr int32_t* metadata =
                    reinterpret_cast<volatile tt_l1_ptr int32_t*>(combined_read_addr);
                metadata[0] = linearized_mesh_coord;
                metadata[1] = token_idx;
                metadata[2] = k;
                metadata[3] = routed_expert;
                metadata[4] = weights[k];

                if (expert_chip == linearized_mesh_coord) {
                    DPRINT_DISPATCH << "    Local dispatch" << ENDL();
                    {
                        DeviceZoneScopedN("dispatch-combined-local");
                        noc_async_write_page(page_idx, combined_addr_gen, combined_read_addr);
                        noc_async_writes_flushed();
                    }
                } else {
                    DPRINT_DISPATCH << "    Remote dispatch to chip " << expert_chip << ENDL();
                    if constexpr (is_1d_topology<topology>()) {
                        uint32_t route = get_route<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);
                        uint32_t link = fabric_send_counter % num_links;
                        fabric_send_counter++;

                        uint32_t distance =
                            manhattan_distance<topology, mesh_rows, mesh_cols>(linearized_mesh_coord, expert_chip);

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
                    }
                }

                offset++;
            }
            noc_async_write_barrier();

            cb_pop_front(cb_indices_id, 1);
            cb_pop_front(cb_weights_id, 1);
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
