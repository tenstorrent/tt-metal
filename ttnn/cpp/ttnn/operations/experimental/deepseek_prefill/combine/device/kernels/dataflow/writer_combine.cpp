// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Prefill combine writer kernel
// This kernel receives combined data from CBs and writes to output tensor

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

// Debug print control - set to 0 to disable combine debug prints, 1 to enable
#define ENABLE_COMBINE_DEBUG 0

#if ENABLE_COMBINE_DEBUG
#define DPRINT_COMBINE DPRINT
#else
#define DPRINT_COMBINE \
    if (0)             \
    DebugPrinter()
#endif

void kernel_main() {
    // ===== Compile Time Args =====
    // CB IDs (indices 0-3)
    constexpr uint32_t cb_dispatched_buffer_id = get_compile_time_arg_val(0);
    constexpr uint32_t cb_dispatched_metadata_id = get_compile_time_arg_val(1);
    constexpr uint32_t cb_experts_tok_counter_id = get_compile_time_arg_val(2);
    constexpr uint32_t cb_output_id = get_compile_time_arg_val(3);

    // Page counts (indices 4-7)
    constexpr uint32_t dispatched_buffer_pages = get_compile_time_arg_val(4);
    constexpr uint32_t dispatched_metadata_pages = get_compile_time_arg_val(5);
    constexpr uint32_t experts_tok_counter_pages = get_compile_time_arg_val(6);
    constexpr uint32_t output_pages = get_compile_time_arg_val(7);

    // Page sizes (indices 8-11)
    constexpr uint32_t dispatched_buffer_page_size = get_compile_time_arg_val(8);
    constexpr uint32_t dispatched_metadata_page_size = get_compile_time_arg_val(9);
    constexpr uint32_t experts_tok_counter_page_size = get_compile_time_arg_val(10);
    constexpr uint32_t output_page_size = get_compile_time_arg_val(11);

    // Operation parameters (indices 12-16)
    constexpr uint32_t num_chips = get_compile_time_arg_val(12);
    constexpr uint32_t experts_per_chip = get_compile_time_arg_val(13);
    constexpr uint32_t num_experts_per_tok = get_compile_time_arg_val(14);
    constexpr uint32_t seq_len_per_chip = get_compile_time_arg_val(15);
    constexpr uint32_t max_dispatched_tokens_per_expert = get_compile_time_arg_val(16);

    // Hidden dimension (index 17)
    constexpr uint32_t hidden_size = get_compile_time_arg_val(17);

    // Aligned page sizes (indices 18-21)
    constexpr uint32_t aligned_dispatched_buffer_page_size = get_compile_time_arg_val(18);
    constexpr uint32_t aligned_dispatched_metadata_page_size = get_compile_time_arg_val(19);
    constexpr uint32_t aligned_experts_tok_counter_page_size = get_compile_time_arg_val(20);
    constexpr uint32_t aligned_output_page_size = get_compile_time_arg_val(21);

    // Mesh information (indices 22-26)
    constexpr uint32_t src_mesh_id = get_compile_time_arg_val(22);
    constexpr uint32_t src_chip_id = get_compile_time_arg_val(23);
    constexpr uint32_t mesh_rows = get_compile_time_arg_val(24);
    constexpr uint32_t mesh_cols = get_compile_time_arg_val(25);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(26);

    // Fabric configuration (indices 27-30)
    constexpr uint32_t fabric_max_packet_size = get_compile_time_arg_val(27);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(28);
    constexpr uint32_t num_links = get_compile_time_arg_val(29);
    constexpr tt::tt_fabric::Topology topology = (tt::tt_fabric::Topology)get_compile_time_arg_val(30);

    // TensorAccessorArgs for all 4 tensors (starting at index 31)
    constexpr auto dispatched_buffer_args = TensorAccessorArgs<31>();
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
    uint32_t zero_init_semaphore_address = get_semaphore(zero_init_semaphore_id);  // Local semaphore
    uint32_t init_semaphore_address =
        get_arg_val<uint32_t>(rt_args_idx++);  // Global semaphore (address passed directly)

    // Fabric connection args follow (appended by append_fabric_connection_rt_args)

    // Print key compile time args for debugging (RISCV_0)
    DPRINT_COMBINE << "Combine Writer: CBs=" << cb_dispatched_buffer_id << "," << cb_dispatched_metadata_id << ","
                   << cb_experts_tok_counter_id << "," << cb_output_id << " num_chips=" << num_chips
                   << " experts_per_chip=" << experts_per_chip << " num_experts_per_tok=" << num_experts_per_tok
                   << " seq_len_per_chip=" << seq_len_per_chip
                   << " max_dispatched_tokens_per_expert=" << max_dispatched_tokens_per_expert
                   << " hidden_size=" << hidden_size << " linearized_mesh_coord" << linearized_mesh_coord
                   << " src_mesh_id=" << src_mesh_id << " src_chip_id=" << src_chip_id << " mesh_rows=" << mesh_rows
                   << " mesh_cols=" << mesh_cols << ENDL();

#ifdef DEST_CHIP_ID
    // Fabric is enabled - set up connections
    using namespace ttnn::operations::ccl::common;

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
    constexpr uint32_t dispatch_devices = num_chips;
    constexpr uint32_t dispatch_index = linearized_mesh_coord;
    constexpr uint32_t device_begin_idx = 0;
    constexpr uint32_t device_end_idx = num_chips;
    constexpr uint32_t device_stride = 1;
#endif

    DPRINT_COMBINE << "dispatch_devices=" << dispatch_devices << " axis=" << (int)axis
                   << " device_begin_idx=" << device_begin_idx << " device_end_idx=" << device_end_idx
                   << " device_stride=" << device_stride << ENDL();

    DPRINT_COMBINE << "Fabric enabled: num_links=" << num_links << " topology=" << (uint32_t)topology << ENDL();

    constexpr uint32_t num_devices = mesh_rows * mesh_cols;
    constexpr uint8_t dest_chip_ids[num_devices] = DEST_CHIP_ID;
    constexpr uint8_t dest_mesh_ids[num_devices] = DEST_MESH_ID;
    constexpr std::array<bool, 4> directions = DIRECTIONS;

    DPRINT_COMBINE << "Opening fabric connections async..." << ENDL();
    std::array<tt::tt_fabric::WorkerToFabricEdmSender, 4> fabric_connections;
    open_direction_connections_async(directions, fabric_connections, rt_args_idx);

    // Set up packet header from CB (cb_id 4)
    constexpr uint32_t cb_packet_header_id = 4;
    uint32_t packet_header_buffer_address = get_read_ptr(cb_packet_header_id);
    auto* unicast_packet_header = reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_address);

    DPRINT_COMBINE << "Waiting for fabric connections barrier..." << ENDL();
    open_direction_connections_barrier(directions, fabric_connections);

    // Wait for local reader to complete zero-init (or signal if INIT_ZEROS=0)
    DPRINT_COMBINE << "Waiting for local reader zero-init..." << ENDL();
    volatile tt_l1_ptr uint32_t* zero_init_sem_ptr =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zero_init_semaphore_address);
    noc_semaphore_wait(zero_init_sem_ptr, 1);
    noc_semaphore_set(zero_init_sem_ptr, 0);
    DPRINT_COMBINE << "Local reader zero-init done" << ENDL();

    // Send init semaphore to all devices (fabric aand zeros completed)
    const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
    DPRINT_COMBINE << "Sending init semaphore to configured targets..." << ENDL();
    send_init_semaphore_to_configured_targets<
        linearized_mesh_coord,
        topology,
        src_chip_id,
        mesh_rows,
        mesh_cols,
        axis,
        num_devices>(fabric_connections, unicast_packet_header, dest_chip_ids, dest_mesh_ids, init_noc_semaphore_addr);

    // Wait for all devices to complete fabric initialization
    DPRINT_COMBINE << "Waiting for all devices to complete fabric init..." << ENDL();
    volatile tt_l1_ptr uint32_t* init_sem_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
    noc_semaphore_wait(init_sem_ptr, dispatch_devices - 1);
    noc_semaphore_set(init_sem_ptr, 0);

    DPRINT_COMBINE << "Fabric and zero-init setup complete" << ENDL();
#endif

    cb_wait_front(cb_experts_tok_counter_id, 1);
    DPRINT_COMBINE << "cb_experts_tok_counter_id: " << cb_experts_tok_counter_id << ENDL();
    uint32_t experts_tok_counter_cb_addr = get_read_ptr(cb_experts_tok_counter_id);
    volatile tt_l1_ptr uint32_t* experts_tok_counter =
        reinterpret_cast<volatile tt_l1_ptr uint32_t*>(experts_tok_counter_cb_addr);

    const auto output_addr_gen = TensorAccessor(output_args, output_addr, aligned_output_page_size);

    for (uint32_t expert_idx = 0; expert_idx < experts_per_chip; ++expert_idx) {
        DPRINT_COMBINE << "Processing expert " << expert_idx << "/" << experts_per_chip << ENDL();
        for (uint32_t tok_idx = 0; tok_idx < experts_tok_counter[expert_idx]; ++tok_idx) {
            cb_wait_front(cb_dispatched_buffer_id, 1);
            cb_wait_front(cb_dispatched_metadata_id, 1);

            uint32_t buffer_cb_addr = get_read_ptr(cb_dispatched_buffer_id);
            uint32_t metadata_cb_addr = get_read_ptr(cb_dispatched_metadata_id);
            volatile tt_l1_ptr uint32_t* metadata = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(metadata_cb_addr);

            auto dst_chip = metadata[0];
            auto dst_token_idx = metadata[1];
            auto dst_topk_indice = metadata[2];

            DPRINT_COMBINE << "  Token " << tok_idx << "/" << experts_tok_counter[expert_idx]
                           << ": dst_chip=" << dst_chip << " dst_token_idx=" << dst_token_idx
                           << " dst_topk_indice=" << dst_topk_indice << ENDL();

            // Calculate output page index
            // Output shape: (num_chips, seq_len_per_chip, num_experts_per_tok, hidden_dim)
            // Pages are laid out as: token * num_experts_per_tok + topk_indice
            uint32_t output_page_idx = dst_token_idx * num_experts_per_tok + dst_topk_indice;

            if (dst_chip == linearized_mesh_coord) {
                // DPRINT_COMBINE << "    Token" << dst_token_idx << " is local to this chip." << ENDL();
                // Local write - direct NOC write to DRAM
                noc_async_write_page(output_page_idx, output_addr_gen, buffer_cb_addr);
                noc_async_write_barrier();
            } else {
                // Remote write via fabric
                // DPRINT_COMBINE << "    Token" << dst_token_idx << " is remote to this chip. Destination chip: "
                //                << dst_chip << ENDL();
#ifdef DEST_CHIP_ID
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
                        dst_chip,
                        buffer_cb_addr,
                        output_page_idx,
                        (int)aligned_output_page_size,
                        l1_alignment);
                }
#endif
            }

            cb_pop_front(cb_dispatched_buffer_id, 1);
            cb_pop_front(cb_dispatched_metadata_id, 1);
        }
    }
    {
        // Send init semaphore to all devices (fabric aand zeros completed)
        const uint64_t init_noc_semaphore_addr = get_noc_addr(init_semaphore_address);
        DPRINT_COMBINE << "Sending init semaphore to configured targets..." << ENDL();
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
        DPRINT_COMBINE << "Waiting for all devices to complete fabric init..." << ENDL();
        volatile tt_l1_ptr uint32_t* init_sem_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(init_semaphore_address);
        noc_semaphore_wait(init_sem_ptr, dispatch_devices - 1);
        noc_semaphore_set(init_sem_ptr, 0);

        DPRINT_COMBINE << "Fabric and zero-init setup complete" << ENDL();
    }

#ifdef DEST_CHIP_ID
    // Close fabric connections to prevent resource conflicts with subsequent operations
    close_direction_connections(directions, fabric_connections);
#endif
}
