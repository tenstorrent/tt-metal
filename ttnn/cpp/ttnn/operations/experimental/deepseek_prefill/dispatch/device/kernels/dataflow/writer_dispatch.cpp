// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill dispatch writer kernel

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "api/debug/assert.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

// Debug print control - set to 0 to disable dispatch debug prints, 1 to enable
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
    // CB IDs (indices 0-6)
    constexpr uint32_t cb_input_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_indices_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_weights_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_offsets_id = get_compile_time_arg_val(3);
    constexpr uint32_t cb_metadata_temp_id = get_compile_time_arg_val(4);
    constexpr uint32_t cb_packet_header_id = get_compile_time_arg_val(5);
    constexpr uint32_t cb_dispatch_table_id = get_compile_time_arg_val(6);

    // Page counts (indices 7-13)
    constexpr uint32_t input_pages = get_compile_time_arg_val(7);
    constexpr uint32_t indices_pages = get_compile_time_arg_val(8);
    constexpr uint32_t weights_pages = get_compile_time_arg_val(9);
    constexpr uint32_t offsets_pages = get_compile_time_arg_val(10);
    constexpr uint32_t output_pages = get_compile_time_arg_val(11);
    constexpr uint32_t metadata_pages = get_compile_time_arg_val(12);
    constexpr uint32_t dispatch_table_pages = get_compile_time_arg_val(13);

    // Page sizes (indices 14-20)
    constexpr uint32_t input_page_size = get_compile_time_arg_val(14);
    constexpr uint32_t indices_page_size = get_compile_time_arg_val(15);
    constexpr uint32_t weights_page_size = get_compile_time_arg_val(16);
    constexpr uint32_t offsets_page_size = get_compile_time_arg_val(17);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t metadata_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t dispatch_table_page_size = get_compile_time_arg_val(20);

    // Operation parameters (indices 21-28)
    constexpr uint32_t num_devices = get_compile_time_arg_val(21);
    constexpr uint32_t hidden_size = get_compile_time_arg_val(22);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(23);
    constexpr uint32_t n_routed_experts = get_compile_time_arg_val(24);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(25);
    constexpr uint32_t metadata_len = get_compile_time_arg_val(26);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(27);
    constexpr uint32_t tokens_per_device = get_compile_time_arg_val(28);

    // Mesh information (indices 29-33)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(29);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(30);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(31);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(32);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(33);

    // Aligned page sizes (indices 34-40)
    constexpr uint32_t aligned_input_page_size = get_compile_time_arg_val(34);
    constexpr uint32_t aligned_indices_page_size = get_compile_time_arg_val(35);
    constexpr uint32_t aligned_weights_page_size = get_compile_time_arg_val(36);
    constexpr uint32_t aligned_offsets_page_size = get_compile_time_arg_val(37);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(38);
    constexpr uint32_t aligned_metadata_page_size = get_compile_time_arg_val(39);
    constexpr uint32_t aligned_dispatch_table_page_size = get_compile_time_arg_val(40);

    // Fabric configuration (indices 41-44)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(41);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(42);
    constexpr uint32_t num_links = get_compile_time_arg_val(43);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(44);

    // TensorAccessorArgs for all 7 tensors (starting at index 45)
    constexpr auto input_args = TensorAccessorArgs<45>();
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

    // Fabric connection args follow (appended by append_fabric_connection_rt_args)
    // These will be read by fabric API calls

    // Print key compile time args for debugging (using DPRINT_DATA0 - writer runs on RISCV_0)
    DPRINT_DISPATCH << "linearized_mesh_coord=" << linearized_mesh_coord << " src_mesh_id=" << src_mesh_id
                    << " src_chip_id=" << src_chip_id << " mesh_rows=" << mesh_rows << " mesh_cols=" << mesh_cols
                    << ENDL();

    DPRINT_DISPATCH << "Writer kernel: CBs=" << cb_input_id << "," << cb_indices_id << "," << cb_weights_id << ","
                    << cb_offsets_id << " tokens=[" << token_start_idx << "," << token_end_idx << ")"
                    << " hidden_size=" << hidden_size << " experts_per_chip=" << experts_per_chip << ENDL();

#ifdef AXIS
    constexpr ReplicateGroup axis = ReplicateGroup(AXIS);
    constexpr uint32_t dispatch_devices = axis == ReplicateGroup::COLS ? mesh_rows : mesh_cols;
    constexpr uint32_t row = linearized_mesh_coord / mesh_cols;
    constexpr uint32_t col = linearized_mesh_coord % mesh_cols;

    constexpr uint32_t dispatch_index = axis == ReplicateGroup::COLS ? row : col;
    // Based on cluster axis, we only need to dispatch to the devices that are along the axis
    // If ReplicateGroup is COLs/AXIS is 1, then we dispatch alonw the ROW, and vice versa
    // For ReplicateGroup COLs/AXIS is 1, the device_begin_idx is the start of the row, and the device_end_idx is the
    // end of the row For ReplicateGroup ROWs/AXIS is 0, the device_begin_idx is the start of the column, and the
    // device_end_idx is the end of the column
    constexpr uint32_t device_begin_idx = axis == ReplicateGroup::COLS ? col : row * mesh_cols;
    constexpr uint32_t device_end_idx =
        (axis == ReplicateGroup::COLS)
            ? (col + mesh_rows * mesh_cols)   // last is col+(mesh_rows-1)*mesh_cols; add one stride
            : (row * mesh_cols + mesh_cols);  // last is row*mesh_cols+(mesh_cols-1); add one
    constexpr uint32_t device_stride = axis == ReplicateGroup::COLS ? mesh_cols : 1;
#else
    constexpr ReplicateGroup axis = ReplicateGroup::NONE;
    constexpr uint32_t dispatch_devices = num_devices;
    constexpr uint32_t dispatch_index = linearized_mesh_coord;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_end_idx = num_devices;
    constexpr uint32_t device_stride = 1;
#endif

    DPRINT_DISPATCH << "dispatch_devices=" << dispatch_devices << ". axis=" << (int)axis << " mesh_rows=" << mesh_rows
                    << " mesh_cols=" << mesh_cols << ENDL();
    DPRINT_DISPATCH << "Fabric configuration: max_packet_size=" << fabric_max_packet_size
                    << " l1_alignment=" << l1_alignment << " num_links=" << num_links
                    << " topology=" << (uint32_t)topology << " device_begin_idx=" << device_begin_idx
                    << " device_end_idx=" << device_end_idx << " device_stride=" << device_stride << ENDL();

    if constexpr (is_1d_topology<topology>()) {
        DPRINT_DISPATCH << "Using 1D topology fabric send" << ENDL();
    } else {
        DPRINT_DISPATCH << "Using non-1D topology fabric send" << ENDL();
    }

#ifdef DEST_CHIP_ID
    // Fabric is enabled - set up connections
    DPRINT_DISPATCH << "Fabric enabled: num_links=" << num_links << " topology=" << (uint32_t)topology << ENDL();
    DPRINT_DISPATCH << "DEBUG: Before creating dest arrays, rt_args_idx=" << rt_args_idx << ENDL();

    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    constexpr std::array<bool, 4> directions = DIRECTIONS;

    DPRINT_DISPATCH << "dst_chip_ids: ";
    for (uint32_t i = 0; i < num_devices; ++i) {
        DPRINT_DISPATCH << (uint32_t)dest_chip_ids[i] << " ";
    }
    DPRINT_DISPATCH << ENDL();
    DPRINT_DISPATCH << "dst_mesh_ids: ";
    for (uint32_t i = 0; i < num_devices; ++i) {
        DPRINT_DISPATCH << (uint32_t)dest_mesh_ids[i] << " ";
    }
    DPRINT_DISPATCH << ENDL();
    DPRINT_DISPATCH << "directions: ";
    for (uint32_t i = 0; i < directions.size(); ++i) {
        DPRINT_DISPATCH << (int)directions[i] << " ";
    }
    DPRINT_DISPATCH << ENDL();
    //

    DPRINT_DISPATCH << "DEBUG: Before open_direction_connections_async" << ENDL();
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);
    DPRINT_DISPATCH << "DEBUG: After open_direction_connections_async" << ENDL();

    // Set up packet headers from CB (cb_packet_header_id from compile-time args)
    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);
    auto* metadata_packet_header =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address + sizeof(PACKET_HEADER_TYPE));

    DPRINT_DISPATCH << "DEBUG: Before open_direction_connections_barrier" << ENDL();
    open_direction_connections_barrier(directions, fabric_connections);
    DPRINT_DISPATCH << "DEBUG: After open_direction_connections_barrier" << ENDL();

    // CRITICAL: Send init semaphore increments to other devices BEFORE waiting
    // Each device sends to configured targets, then waits for (num_devices - 1) increments
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    DPRINT_DISPATCH << "DEBUG: Before send_init_semaphore_to_configured_targets" << ENDL();
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        num_devices>(fabric_connections, metadata_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);
    DPRINT_DISPATCH << "DEBUG: After send_init_semaphore_to_configured_targets" << ENDL();

    // Wait for all devices to complete fabric initialization
    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, dispatch_devices - 1);
    noc_semaphore_set(init_sem_ptr, 0);  // clear if needed for later use

    DPRINT_DISPATCH << "Fabric setup complete" << ENDL();
#endif

    // ====
    // wait for offsets to be ready
    cb_wait_front(cb_offsets_id, offsets_pages);
    int32_t* offsets = (int32_t*)(get_read_ptr(cb_offsets_id));

    for (uint32_t o = 0; o < n_routed_experts; ++o) {
        DPRINT_DISPATCH << "Offset for expert " << o << " is " << offsets[o] << ENDL();
    }

    // ====
    // wait for expert dispatch table to be ready
    cb_wait_front(cb_dispatch_table_id, dispatch_table_pages);
    int32_t* expert_dispatch_table = (int32_t*)(get_read_ptr(cb_dispatch_table_id));

    for (uint32_t e = 0; e < n_routed_experts; ++e) {
        DPRINT_DISPATCH << "Dispatch table for expert " << e << " is " << expert_dispatch_table[e] << ENDL();
    }

    DPRINT_DISPATCH << "aligned_metadata_page_size=" << aligned_metadata_page_size
                    << " metadata_page_size=" << metadata_page_size << ENDL();

    // ====
    // process tokens/indices/weights one by one as they arrive to CB
    const auto output_addr_gen = TensorAccessor(output_args, output_tensor_address, aligned_output_page_size);
    const auto metadata_addr_gen = TensorAccessor(metadata_args, metadata_tensor_address, aligned_metadata_page_size);

    DPRINT_DISPATCH << "metadata_tensor_address=" << HEX() << metadata_tensor_address << DEC()
                    << " aligned_metadata_page_size=" << aligned_metadata_page_size << ENDL();

    // Reserve CB for metadata buffer (using L1 memory accessible by NOC)
    cb_reserve_back(cb_metadata_temp_id, 1);

    for (uint32_t token_idx = token_start_idx; token_idx < token_end_idx; ++token_idx) {
        DPRINT_DISPATCH << "Processing token_idx: " << token_idx << ENDL();

        cb_wait_front(cb_indices_id, 1);
        cb_wait_front(cb_weights_id, 1);
        cb_wait_front(cb_input_id, 1);

        uint32_t input_token_read_addr = get_read_ptr(cb_input_id);
        int32_t* indices = (int32_t*)(get_read_ptr(cb_indices_id));
        uint16_t* weights = (uint16_t*)(get_read_ptr(cb_weights_id));  // this is really a bfloat16 data
        for (uint32_t k = 0; k < num_experts_per_tok; ++k) {
            auto routed_expert = indices[k];

            // Use expert dispatch table to map expert ID to destination chip
            auto expert_chip_og = expert_dispatch_table[routed_expert];
            if (expert_chip_og == -1) {
                // Expert not present in this EP rank, skip
                DPRINT_DISPATCH << "token" << token_idx << "  Expert [" << k << "]=" << routed_expert
                                << " not in this EP rank, skipping" << ENDL();
                continue;
            }
            auto expert_chip = device_begin_idx + expert_chip_og * device_stride;
            auto expert_index_within_chip = routed_expert % experts_per_chip;

            DPRINT_DISPATCH << "token" << token_idx << "  Expert [" << k << "]=" << routed_expert
                            << "  (chip=" << expert_chip_og << " =>" << expert_chip << ")" << ENDL();

            auto& offset = offsets[routed_expert];

            // Calculate byte offsets from page indices
            // aligned_page_size is in elements, need to multiply by element size to get bytes
            auto page_idx = expert_index_within_chip * max_dispatched_tokens_per_expert + offset;
            auto output_token_write_addr =
                output_addr_gen.get_noc_addr(0) + page_idx * aligned_output_page_size * 2;  // bfloat16 = 2 bytes
            auto metadata_write_addr =
                metadata_addr_gen.get_noc_addr(0) + page_idx * aligned_metadata_page_size * 4;  // int32 = 4 bytes

            if (expert_chip == linearized_mesh_coord) {
                noc_async_write_page(page_idx, output_addr_gen, input_token_read_addr);
                noc_async_writes_flushed();

                uint32_t metadata_cb_addr = get_write_ptr(cb_metadata_temp_id);
                volatile tt_l1_ptr int32_t* metadata = reinterpret_cast<volatile tt_l1_ptr int32_t*>(metadata_cb_addr);

                // Set actual metadata values
                metadata[0] = linearized_mesh_coord;
                metadata[1] = token_idx;
                metadata[2] = k;
                metadata[3] = routed_expert;
                metadata[4] = static_cast<int16_t>(weights[k]);

                // Write metadata from CB to output tensor
                noc_async_write_page(page_idx, metadata_addr_gen, metadata_cb_addr);
                noc_async_writes_flushed();

            } else {
                if constexpr (is_1d_topology<topology>()) {
                    fabric_send_chip_unicast_noc_unicast_1d<
                        linearized_mesh_coord,
                        topology,
                        mesh_rows,
                        mesh_cols,
                        fabric_max_packet_size>(
                        output_addr_gen,
                        fabric_connections,
                        unicast_packet_header,
                        expert_chip,  // linearized_dest_mesh_coord
                        input_token_read_addr,
                        page_idx,
                        (int)aligned_output_page_size,  // bfloat16 = 2 bytes
                        l1_alignment);

                    uint32_t metadata_cb_addr = get_write_ptr(cb_metadata_temp_id);
                    volatile tt_l1_ptr int32_t* metadata =
                        reinterpret_cast<volatile tt_l1_ptr int32_t*>(metadata_cb_addr);

                    // Set actual metadata values
                    metadata[0] = linearized_mesh_coord;
                    metadata[1] = token_idx;
                    metadata[2] = k;
                    metadata[3] = routed_expert;
                    metadata[4] = static_cast<int16_t>(weights[k]);

                    fabric_send_chip_unicast_noc_unicast_1d<
                        linearized_mesh_coord,
                        topology,
                        mesh_rows,
                        mesh_cols,
                        fabric_max_packet_size>(
                        metadata_addr_gen,
                        fabric_connections,
                        metadata_packet_header,
                        expert_chip,  // linearized_dest_mesh_coord
                        metadata_cb_addr,
                        page_idx,
                        (int)aligned_metadata_page_size,  // int32 = 4 bytes
                        l1_alignment);
                } else {
                    // Not implemented
                    DPRINT << "Error: 2D topology fabric not implemented." << ENDL();
                    ASSERT(false, "2D topology fabric not implemented");
                    return;
                }
            }

            offset++;
        }
        noc_async_write_barrier();

        cb_pop_front(cb_indices_id, 1);
        cb_pop_front(cb_weights_id, 1);
        cb_pop_front(cb_input_id, 1);
    }

    // Release CB for next use
    cb_push_back(cb_metadata_temp_id, 1);

    // Pop the offsets and dispatch table CBs that were read at the beginning
    cb_pop_front(cb_offsets_id, offsets_pages);
    cb_pop_front(cb_dispatch_table_id, dispatch_table_pages);

    {
        // Send exit semaphore to all devices
        const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
        DPRINT_DISPATCH << "Sending exit semaphore to configured targets..." << ENDL();
        send_init_semaphore_to_configured_targets<
            linearized_mesh_coord,
            topology,
            src_chip_id,
            mesh_rows,
            mesh_cols,
            axis,
            num_devices>(
            fabric_connections, unicast_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

        // Wait for all devices to complete fabric initialization
        DPRINT_DISPATCH << "Waiting for all devices to complete fabric transfers..." << ENDL();
        volatile tt_l1_ptr uint32_t* init_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
        noc_semaphore_wait(init_sem_ptr, dispatch_devices - 1);
        noc_semaphore_set(init_sem_ptr, 0);

        DPRINT_DISPATCH << "Fabric transfers complete" << ENDL();
    }

#ifdef DEST_CHIP_ID
    // Close fabric connections to prevent resource conflicts with subsequent operations
    close_direction_connections(directions, fabric_connections);
#endif
}
