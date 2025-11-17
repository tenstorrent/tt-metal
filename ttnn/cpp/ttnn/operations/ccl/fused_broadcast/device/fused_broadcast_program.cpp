// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_broadcast_program.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/point_to_point/device/host/point_to_point_device_op.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks fused_broadcast_multicore(
    const Tensor& input_tensor,
    Tensor& output_tensor,
    const MeshCoordinate& device_coord,
    const MeshCoordinate& root_coord,
    const MeshCoordinate& mesh_shape,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    tt::tt_fabric::Topology topology,
    const GlobalSemaphore& semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    auto device = input_tensor.device();
    auto program = tt::tt_metal::CreateProgram();
    // print mesh shape
    log_info(tt::LogOp, "Fused broadcast mesh shape: ({}, {})", mesh_shape[0], mesh_shape[1]);

    // Validate topology support
    TT_FATAL(
        topology == tt::tt_fabric::Topology::Linear || topology == tt::tt_fabric::Topology::Ring,
        "Fused broadcast supports Linear and Ring topologies for 2D mesh communication");

    // Get current device coordinates in mesh
    auto mesh_device = device->get_mesh_device();

    // Automatically determine which axis is for TP (2 devices) and which is for SP (4 devices)
    // TP axis should have 2 devices, SP axis should have 4 devices
    uint32_t tp_axis, sp_axis;
    if (mesh_shape[0] == 2 && mesh_shape[1] == 4) {
        // (2,4) mesh: axis 0 for TP (2 devices), axis 1 for SP (4 devices)
        tp_axis = 0;
        sp_axis = 1;
    } else if (mesh_shape[0] == 4 && mesh_shape[1] == 2) {
        // (4,2) mesh: axis 1 for TP (2 devices), axis 0 for SP (4 devices)
        tp_axis = 1;
        sp_axis = 0;
    } else {
        TT_FATAL(
            false,
            "Fused broadcast only supports (2,4) and (4,2) mesh shapes. Got: ({}, {})",
            mesh_shape[0],
            mesh_shape[1]);
    }

    log_info(tt::LogOp, "Fused broadcast device coord: ({}, {})", device_coord[0], device_coord[1]);
    log_info(tt::LogOp, "Fused broadcast root coord: ({}, {})", root_coord[0], root_coord[1]);
    log_info(tt::LogOp, "Mesh layout: TP axis={} (2 devices), SP axis={} (4 devices)", tp_axis, sp_axis);

    // Find TP partner along TP axis (2 devices total)
    auto tp_forward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, device_coord, 1, topology, tp_axis);
    auto tp_backward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, device_coord, -1, topology, tp_axis);

    // Find SP neighbors along SP axis (4 devices total)
    auto sp_forward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, device_coord, 1, topology, sp_axis);
    auto sp_backward_coord =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(input_tensor, device_coord, -1, topology, sp_axis);

    // Add debug information
    log_info(tt::LogOp, "Fused broadcast device info:");
    log_info(tt::LogOp, "  Current device: ({}, {})", device_coord[0], device_coord[1]);
    log_info(tt::LogOp, "  Root coord: ({}, {})", root_coord[0], root_coord[1]);
    log_info(tt::LogOp, "  Mesh shape: ({}, {})", mesh_shape[0], mesh_shape[1]);
    log_info(
        tt::LogOp,
        "  TP forward: {}",
        tp_forward_coord.has_value() ? fmt::format("({}, {})", tp_forward_coord.value()[0], tp_forward_coord.value()[1])
                                     : "None");
    log_info(
        tt::LogOp,
        "  TP backward: {}",
        tp_backward_coord.has_value()
            ? fmt::format("({}, {})", tp_backward_coord.value()[0], tp_backward_coord.value()[1])
            : "None");
    log_info(
        tt::LogOp,
        "  SP forward: {}",
        sp_forward_coord.has_value() ? fmt::format("({}, {})", sp_forward_coord.value()[0], sp_forward_coord.value()[1])
                                     : "None");
    log_info(
        tt::LogOp,
        "  SP backward: {}",
        sp_backward_coord.has_value()
            ? fmt::format("({}, {})", sp_backward_coord.value()[0], sp_backward_coord.value()[1])
            : "None");

    // Validate device coordinates are within mesh bounds
    TT_FATAL(
        device_coord[0] < mesh_shape[0] && device_coord[1] < mesh_shape[1],
        "Device coord ({}, {}) outside mesh bounds ({}, {})",
        device_coord[0],
        device_coord[1],
        mesh_shape[0],
        mesh_shape[1]);

    bool is_root = (device_coord == root_coord);
    bool is_tp_partner = false;
    bool is_receiver = false;
    std::optional<MeshCoordinate> tp_partner_coord;

    if (is_root) {
        // Root device - find its TP partner for P2P phase
        // TP partner should be the other device on the TP axis, not the root itself
        MeshCoordinate tp_partner_candidate = device_coord;
        if (mesh_shape[tp_axis] == 2) {
            // If TP axis is rows (axis 0), swap row
            tp_partner_candidate[tp_axis] = (root_coord[tp_axis] == 0) ? 1 : 0;
        } else {
            // If TP axis is columns (axis 1), swap col
            tp_partner_candidate[tp_axis] = (root_coord[tp_axis] == 0) ? 1 : 0;
        }
        tp_partner_coord = tp_partner_candidate;
        TT_FATAL(
            tp_partner_coord.has_value(),
            "Fused broadcast root device could not find TP partner. Candidate: (%d,%d)",
            tp_partner_candidate[0],
            tp_partner_candidate[1]);
        log_info(
            tt::LogOp,
            "  Root device using TP partner: ({}, {})",
            tp_partner_coord.value()[0],
            tp_partner_coord.value()[1]);
    } else if (device_coord == MeshCoordinate{(root_coord[tp_axis] == 0) ? 1 : 0, root_coord[1 - tp_axis]}) {
        // This device is the TP partner of the root (other device on TP axis)
        is_tp_partner = true;
        tp_partner_coord = root_coord;  // For symmetry, though not used in TP partner kernels
        log_info(tt::LogOp, "  TP partner device");
    } else {
        // This device is a receiver (column device)
        is_receiver = true;
        log_info(tt::LogOp, "  Receiver device");
    }

    // Setup tensor properties
    bool sharded = input_tensor.is_sharded();

    uint32_t row_size = input_tensor.logical_shape()[-1] * input_tensor.element_size();

    // Calculate packet parameters
    constexpr uint32_t max_packet_size = 4096;  // Common ethernet packet size
    uint32_t num_rows_per_packet = std::max(1u, max_packet_size / row_size);

    // Calculate packet parameters following reference patterns
    const uint32_t input_num_pages = input_tensor.buffer()->num_pages();
    const uint32_t input_page_size_bytes = input_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // For P2P packet sizing
    const auto [packet_size_bytes, num_pages_per_packet, num_page_segments, total_packets] =
        ttnn::operations::point_to_point::detail::compute_aligned_packet_dims(
            input_tensor.dtype(), input_page_size_bytes, input_num_pages, l1_alignment);

    // Setup circular buffers using proper data format and P2P pattern
    constexpr uint32_t cb0_id = tt::CB::c_in0;
    constexpr uint32_t cb1_id = tt::CB::c_out0;

    tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t aligned_input_page_size_bytes = tt::round_up(input_page_size_bytes, l1_alignment);
    constexpr auto cb_num_pages = 2;  // From P2P pattern

    auto input_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * aligned_input_page_size_bytes, {{cb0_id, input_dataformat}})
            .set_page_size(cb0_id, aligned_input_page_size_bytes);

    auto output_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * aligned_input_page_size_bytes, {{cb1_id, input_dataformat}})
            .set_page_size(cb1_id, aligned_input_page_size_bytes);

    // Create CBs based on device role
    if (is_root || is_tp_partner) {
        // Root and TP partner need both input and output CBs
        tt::tt_metal::CreateCircularBuffer(program, CoreRangeSet({CoreRange({0, 0}, {0, 0})}), input_cb_config);
        tt::tt_metal::CreateCircularBuffer(program, CoreRangeSet({CoreRange({0, 0}, {0, 0})}), output_cb_config);
    } else {
        // Receivers only need output CB
        tt::tt_metal::CreateCircularBuffer(program, CoreRangeSet({CoreRange({0, 0}, {0, 0})}), output_cb_config);
    }

    // Setup fabric circular buffers for P2P operations
    constexpr uint32_t packet_header_cb_id = tt::CB::c_in1;
    constexpr uint32_t packet_cb_id = tt::CB::c_in2;

    if (is_root || is_tp_partner || is_receiver) {
        // Add packet header CB for fabric operations (following P2P pattern)
        constexpr auto buffering_factor = 2;
        constexpr auto num_packet_headers_storable = 2;
        const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        auto packet_header_cb_config = tt::tt_metal::CircularBufferConfig(
                                           num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
                                           {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
                                           .set_page_size(packet_header_cb_id, packet_header_size_bytes);

        tt::tt_metal::CreateCircularBuffer(program, CoreRangeSet({CoreRange({0, 0}, {0, 0})}), packet_header_cb_config);

        // Add packet CB for fabric data transfer
        tt::DataFormat input_dataformat = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        auto packet_cb_config =
            tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{packet_cb_id, input_dataformat}})
                .set_page_size(packet_cb_id, packet_size_bytes);

        tt::tt_metal::CreateCircularBuffer(program, CoreRangeSet({CoreRange({0, 0}, {0, 0})}), packet_cb_config);
    }

    // Setup compile-time args following exact P2P and broadcast patterns
    std::vector<uint32_t> root_reader_compile_args;
    std::vector<uint32_t> root_writer_compile_args;
    std::vector<uint32_t> tp_reader_compile_args;
    std::vector<uint32_t> tp_writer_compile_args;
    std::vector<uint32_t> receiver_reader_compile_args;
    std::vector<uint32_t> receiver_writer_compile_args;

    // Root reader compile args: exactly like P2P sender reader
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(root_reader_compile_args);

    // Root writer compile args: P2P args + TensorAccessor + broadcast args (matching kernel expectations)
    root_writer_compile_args = {cb0_id, packet_header_cb_id, packet_cb_id, l1_alignment};

    // Add broadcast-specific compile args for the SP phase (starting at index 8 in kernel)
    uint32_t num_packets_per_page = static_cast<uint32_t>(std::ceil(static_cast<double>(row_size) / packet_size_bytes));
    root_writer_compile_args.insert(
        root_writer_compile_args.end(),
        {
            input_page_size_bytes,                  // page_size (index 8)
            row_size,                               // row_size (index 9)
            packet_size_bytes,                      // max_packet_size (index 10)
            num_rows_per_packet,                    // num_rows_per_packet (index 11)
            num_packets_per_page,                   // num_packets_per_row (index 12)
            sp_forward_coord.has_value() ? 1 : 0,   // num_targets_forward_direction (index 13)
            sp_backward_coord.has_value() ? 1 : 0,  // num_targets_backward_direction (index 14)
            1,                                      // start_distance_in_hops_forward (index 16)  ????
            mesh_shape[sp_axis] - 1,                // range_hops_forward (index 17)   ????
            1,                                      // start_distance_in_hops_backward (index 18) ????
            mesh_shape[sp_axis] - 1                 // range_hops_backward (index 19) ????
        });

    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(root_writer_compile_args);

    // TP reader compile args: exactly like P2P receiver reader
    tp_reader_compile_args = {packet_header_cb_id, packet_cb_id, cb1_id, l1_alignment};
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(tp_reader_compile_args);  // Intermediate buffer

    // TP writer compile args: like broadcast writer but simplified for fused operation
    tp_writer_compile_args = {
        cb1_id,                                 // cb_id_in
        num_pages_per_packet,                   // packet_size_in_pages
        input_page_size_bytes,                  // page_size
        sp_forward_coord.has_value() ? 1 : 0,   // num_targets_forward_direction
        sp_backward_coord.has_value() ? 1 : 0,  // num_targets_backward_direction
        1,                                      // is_sender
        1,                                      // start_distance_in_hops_forward
        mesh_shape[sp_axis] - 1,                // range_hops_forward
        1,                                      // start_distance_in_hops_backward
        mesh_shape[sp_axis] - 1                 // range_hops_backward
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(tp_writer_compile_args);

    // Receiver reader compile args: empty since receiver reader is empty (fabric writes directly)
    receiver_reader_compile_args = {};

    // Receiver writer compile args: like broadcast writer (non-sender)
    receiver_writer_compile_args = {
        cb1_id,                   // cb0_id
        input_page_size_bytes,    // page_size
        row_size,                 // row_size
        packet_size_bytes,        // max_packet_size
        num_rows_per_packet,      // num_rows_per_packet
        1,                        // num_packets_per_row
        0,                        // num_targets_forward_direction (receiver doesn't send)
        0,                        // num_targets_backward_direction (receiver doesn't send)
        0,                        // is_sender (false)
        1,                        // start_distance_in_hops_forward
        mesh_shape[sp_axis] - 1,  // range_hops_forward
        1,                        // start_distance_in_hops_backward
        mesh_shape[sp_axis] - 1   // range_hops_backward
    };
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(receiver_writer_compile_args);

    // Debug: Print all compile-time arguments for verification
    printf("=== FUSED BROADCAST COMPILE-TIME ARGS DEBUG ===\n");
    printf(
        "Device: %d, coord=(%d,%d), is_root: %d, is_tp_partner: %d, is_receiver: %d\n",
        device->id(),
        device_coord[0],
        device_coord[1],
        is_root,
        is_tp_partner,
        is_receiver);

    if (is_root) {
        printf("ROOT READER compile args (%zu total):\n", root_reader_compile_args.size());
        for (size_t i = 0; i < root_reader_compile_args.size(); i++) {
            printf("  [%zu] = %u\n", i, root_reader_compile_args[i]);
        }

        printf("ROOT WRITER compile args (%zu total):\n", root_writer_compile_args.size());
        for (size_t i = 0; i < root_writer_compile_args.size(); i++) {
            printf("  [%zu] = %u\n", i, root_writer_compile_args[i]);
        }
    } else if (is_tp_partner) {
        printf("TP READER compile args (%zu total):\n", tp_reader_compile_args.size());
        for (size_t i = 0; i < tp_reader_compile_args.size(); i++) {
            printf("  [%zu] = %u\n", i, tp_reader_compile_args[i]);
        }

        printf("TP WRITER compile args (%zu total):\n", tp_writer_compile_args.size());
        for (size_t i = 0; i < tp_writer_compile_args.size(); i++) {
            printf("  [%zu] = %u\n", i, tp_writer_compile_args[i]);
        }
    } else if (is_receiver) {
        printf("RECEIVER READER compile args (%zu total):\n", receiver_reader_compile_args.size());
        for (size_t i = 0; i < receiver_reader_compile_args.size(); i++) {
            printf("  [%zu] = %u\n", i, receiver_reader_compile_args[i]);
        }

        printf("RECEIVER WRITER compile args (%zu total):\n", receiver_writer_compile_args.size());
        for (size_t i = 0; i < receiver_writer_compile_args.size(); i++) {
            printf("  [%zu] = %u\n", i, receiver_writer_compile_args[i]);
        }
    }
    printf("=== END COMPILE-TIME ARGS DEBUG ===\n");

    // Add sharding args if needed
    std::map<std::string, std::string> kernel_defines;
    if (sharded) {
        kernel_defines["SHARDED"] = "1";
        shard_builder::extend_sharding_compile_time_args(input_tensor, root_reader_compile_args);
        shard_builder::extend_sharding_compile_time_args(output_tensor, root_writer_compile_args);
        shard_builder::extend_sharding_compile_time_args(input_tensor, tp_reader_compile_args);
        shard_builder::extend_sharding_compile_time_args(output_tensor, tp_writer_compile_args);
        shard_builder::extend_sharding_compile_time_args(input_tensor, receiver_reader_compile_args);
        shard_builder::extend_sharding_compile_time_args(output_tensor, receiver_writer_compile_args);
    }

    // Create kernels based on device role
    CoreCoord core = {0, 0};  // Single core operation
    CoreRangeSet core_range = CoreRangeSet({CoreRange(core, core)});

    std::optional<KernelHandle> reader_kernel_id;
    std::optional<KernelHandle> writer_kernel_id;

    if (is_root) {
        // Root device: Reader loads input, Writer does TP P2P + SP broadcast
        reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/fused_broadcast/device/kernels2/fused_broadcast_root_reader.cpp",
            core_range,
            tt::tt_metal::ReaderDataMovementConfig(root_reader_compile_args, kernel_defines));

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/fused_broadcast/device/kernels2/fused_broadcast_root_writer.cpp",
            core_range,
            tt::tt_metal::WriterDataMovementConfig(root_writer_compile_args, kernel_defines));
    } else if (is_tp_partner) {
        // TP partner: Reader receives from root, Writer does SP broadcast
        reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/fused_broadcast/device/kernels2/fused_broadcast_tp_reader.cpp",
            core_range,
            tt::tt_metal::ReaderDataMovementConfig(tp_reader_compile_args, kernel_defines));

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/fused_broadcast/device/kernels2/fused_broadcast_tp_writer.cpp",
            core_range,
            tt::tt_metal::WriterDataMovementConfig(tp_writer_compile_args, kernel_defines));
    } else if (is_receiver) {
        // Column receiver: Reader receives SP broadcast, Writer copies to output
        reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/fused_broadcast/device/kernels2/fused_broadcast_receiver_reader.cpp",
            core_range,
            tt::tt_metal::ReaderDataMovementConfig(receiver_reader_compile_args, kernel_defines));

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/ccl/fused_broadcast/device/kernels2/fused_broadcast_receiver_writer.cpp",
            core_range,
            tt::tt_metal::WriterDataMovementConfig(receiver_writer_compile_args, kernel_defines));
    }

    // Setup runtime arguments following exact P2P and broadcast patterns
    std::vector<uint32_t> reader_runtime_args;
    std::vector<uint32_t> writer_runtime_args;
    auto barrier_core = mesh_device->worker_core_from_logical_core({0, 0});

    if (is_root) {
        // Root reader args: exactly like P2P sender reader
        reader_runtime_args = {
            input_tensor.buffer()->address(),
            input_num_pages,       // increment - total pages to read
            0,                     // page_idx_start
            input_page_size_bytes  // src_page_size_bytes
        };

        // Root writer args: Combined P2P + broadcast pattern
        TT_FATAL(tp_partner_coord.has_value(), "Root device must have valid TP partner coordinate");

        // P2P phase args (to TP partner)
        uint32_t tp_num_hops = 1;
        bool tp_dst_is_forward = tp_forward_coord.has_value();

        // SP broadcast phase args
        uint32_t sp_num_connections = (sp_forward_coord.has_value() ? 1 : 0) + (sp_backward_coord.has_value() ? 1 : 0);

        writer_runtime_args = {
            0,  // sp_fabric_arg_start_index (will be updated after TP fabric args)

            // P2P phase (TP horizontal)
            output_tensor.buffer()->address(),         // tp_receiver_base_address
            0,                                         // tp_page_idx_start
            input_num_pages,                           // tp_page_idx_end
            tp_num_hops,                               // tp_dst_num_hops
            input_page_size_bytes,                     // tp_page_size_bytes
            packet_size_bytes,                         // tp_payload_size_bytes
            num_pages_per_packet,                      // tp_max_pages_per_packet
            num_page_segments,                         // tp_page_segments
            semaphore.address(),                       // tp_receive_semaphore_addr
            static_cast<uint32_t>(tp_dst_is_forward),  // tp_dst_is_forward

            // SP broadcast phase (column vertical)
            output_tensor.buffer()->address(),  // sp_tensor_address
            barrier_semaphore.address(),        // sp_out_ready_sem_bank_addr
            0,                                  // sp_row_id_start
            input_num_pages,                    // sp_row_id_end
            0,                                  // sp_wait_output_semaphore (false for sender)
            0,                                  // sp_reset_global_semaphore (false for sender)
            0,                                  // sp_out_ready_sem_noc0_x (not used)
            0,                                  // sp_out_ready_sem_noc0_y (not used)
            1,                                  // sp_out_ready_sem_wait_value
            barrier_semaphore.address(),        // sp_barrier_sem
            barrier_core.x,                     // sp_barrier_sem_noc0_x (not used)
            barrier_core.y,                     // sp_barrier_sem_noc0_y (not used)
            sp_num_connections                  // sp_num_connections
        };

        // Add fabric connection runtime args for TP P2P
        auto src_fabric_id = mesh_device->get_fabric_node_id(device_coord);
        auto dst_fabric_id = mesh_device->get_fabric_node_id(tp_partner_coord.value());
        constexpr uint32_t link_idx = 0;

        if (tp_dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_id, dst_fabric_id, link_idx, program, core, writer_runtime_args);
        }
        writer_runtime_args.emplace_back(!tp_dst_is_forward);
        if (!tp_dst_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_id, dst_fabric_id, link_idx, program, core, writer_runtime_args);
        }

        // Record where SP fabric args will start
        uint32_t sp_fabric_arg_start_index = writer_runtime_args.size();
        writer_runtime_args[0] = sp_fabric_arg_start_index;  // Update the first arg with exact index

        // Add fabric connection for SP broadcast (using broadcast routing pattern)
        std::vector<tt::tt_fabric::FabricNodeId> sp_dst_nodes;
        if (sp_forward_coord.has_value()) {
            sp_dst_nodes.push_back(mesh_device->get_fabric_node_id(sp_forward_coord.value()));
        }
        if (sp_backward_coord.has_value()) {
            sp_dst_nodes.push_back(mesh_device->get_fabric_node_id(sp_backward_coord.value()));
        }

        if (!sp_dst_nodes.empty()) {
            append_routing_plane_connection_manager_rt_args(
                src_fabric_id,
                sp_dst_nodes,
                {link_idx},
                program,
                writer_kernel_id.value(),
                {core},
                writer_runtime_args);
        }
    } else if (is_tp_partner) {
        // TP partner reader args: exactly like P2P receiver reader
        bool sender_is_forward = tp_forward_coord.has_value() && tp_forward_coord.value() == root_coord;

        reader_runtime_args = {
            0,                                        // page_idx_start
            input_num_pages,                          // page_idx_end
            num_pages_per_packet,                     // max_pages_per_packet
            input_tensor.buffer()->address(),         // intermediate_base_addr (for packet buffer)
            packet_size_bytes,                        // packet_size_bytes
            input_page_size_bytes,                    // page_size_bytes
            num_page_segments,                        // page_segments
            semaphore.address(),                      // sender_semaphore_addr
            1,                                        // sender_num_hops (from root)
            static_cast<uint32_t>(sender_is_forward)  // sender_is_forward
        };

        // TP partner writer args: follow broadcast writer pattern for SP phase
        uint32_t sp_num_connections = (sp_forward_coord.has_value() ? 1 : 0) + (sp_backward_coord.has_value() ? 1 : 0);

        writer_runtime_args = {
            // Local tensor args (P2P writer part)
            output_tensor.buffer()->address(),  // local_tensor_addr
            input_num_pages,                    // num_tiles
            0,                                  // start_tile_id

            // Broadcast args (SP writer part)
            output_tensor.buffer()->address(),  // broadcast_tensor_addr
            barrier_semaphore.address(),        // out_ready_sem_bank_addr
            0,                                  // tile_id_start
            input_num_pages,                    // tile_id_end
            0,                                  // wait_output_semaphore (false for sender)
            0,                                  // reset_global_semaphore (false for sender)
            0,                                  // out_ready_sem_noc0_x (not used)
            0,                                  // out_ready_sem_noc0_y (not used)
            1,                                  // out_ready_sem_wait_value
            barrier_semaphore.address(),        // barrier_sem
            barrier_core.x,                     // barrier_sem_noc0_x
            barrier_core.y,                     // barrier_sem_noc0_y
            sp_num_connections                  // num_connections
        };

        // Add fabric connection for TP reader (receiving from root)
        auto root_fabric_id = mesh_device->get_fabric_node_id(root_coord);
        auto tp_fabric_id = mesh_device->get_fabric_node_id(device_coord);
        constexpr uint32_t link_idx = 0;

        if (sender_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                root_fabric_id, tp_fabric_id, link_idx, program, core, reader_runtime_args);
        }
        reader_runtime_args.emplace_back(!sender_is_forward);
        if (!sender_is_forward) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                root_fabric_id, tp_fabric_id, link_idx, program, core, reader_runtime_args);
        }

        // Add fabric connection for TP writer SP broadcast
        std::vector<tt::tt_fabric::FabricNodeId> sp_dst_nodes;
        if (sp_forward_coord.has_value()) {
            sp_dst_nodes.push_back(mesh_device->get_fabric_node_id(sp_forward_coord.value()));
        }
        if (sp_backward_coord.has_value()) {
            sp_dst_nodes.push_back(mesh_device->get_fabric_node_id(sp_backward_coord.value()));
        }

        if (!sp_dst_nodes.empty()) {
            append_routing_plane_connection_manager_rt_args(
                tp_fabric_id, sp_dst_nodes, {link_idx}, program, writer_kernel_id.value(), {core}, writer_runtime_args);
        }
    } else if (is_receiver) {
        // Column receiver reader args: follow broadcast receiver pattern (fabric writes directly - no reader)
        reader_runtime_args = {};  // Empty - receiver reader is empty, fabric writes directly to CB

        // Column receiver writer args: follow broadcast writer pattern (non-sender)
        writer_runtime_args = {
            output_tensor.buffer()->address(),  // tensor_address0
            barrier_semaphore.address(),        // out_ready_sem_bank_addr
            0,                                  // tile_id_start
            input_num_pages,                    // tile_id_end
            1,                                  // wait_output_semaphore (true for receiver)
            1,                                  // reset_global_semaphore (true for receiver)
            0,                                  // out_ready_sem_noc0_x (will be set by fabric)
            0,                                  // out_ready_sem_noc0_y (will be set by fabric)
            1,                                  // out_ready_sem_wait_value
            barrier_semaphore.address(),        // barrier_sem
            barrier_core.x,                     // barrier_sem_noc0_x (will be set by fabric)
            barrier_core.y,                     // barrier_sem_noc0_y (will be set by fabric)
            0                                   // num_connections (receiver doesn't send)
        };

        // No fabric connection needed for receiver - it receives via broadcast routing
        // The SP broadcast from root/TP partner will automatically reach this device
    }

    // Set runtime arguments
    if (reader_kernel_id.has_value()) {
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id.value(), core, reader_runtime_args);
    }
    if (writer_kernel_id.has_value()) {
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id.value(), core, writer_runtime_args);
    }

    // Debug: Print runtime arguments
    printf("=== FUSED BROADCAST RUNTIME ARGS DEBUG ===\n");
    printf(
        "Device: %d, coord=(%d,%d), is_root: %d, is_tp_partner: %d, is_receiver: %d\n",
        device->id(),
        device_coord[0],
        device_coord[1],
        is_root,
        is_tp_partner,
        is_receiver);

    if (is_root) {
        printf("ROOT READER runtime args (%zu total):\n", reader_runtime_args.size());
        for (size_t i = 0; i < reader_runtime_args.size(); i++) {
            printf("  [%zu] = %u\n", i, reader_runtime_args[i]);
        }

        printf("ROOT WRITER runtime args (%zu total):\n", writer_runtime_args.size());
        for (size_t i = 0; i < writer_runtime_args.size(); i++) {
            printf("  [%zu] = %u\n", i, writer_runtime_args[i]);
        }
    } else if (is_tp_partner) {
        printf("TP READER runtime args (%zu total):\n", reader_runtime_args.size());
        for (size_t i = 0; i < reader_runtime_args.size(); i++) {
            printf("  [%zu] = %u\n", i, reader_runtime_args[i]);
        }

        printf("TP WRITER runtime args (%zu total):\n", writer_runtime_args.size());
        for (size_t i = 0; i < writer_runtime_args.size(); i++) {
            printf("  [%zu] = %u\n", i, writer_runtime_args[i]);
        }
    } else if (is_receiver) {
        printf("RECEIVER READER runtime args (%zu total):\n", reader_runtime_args.size());
        for (size_t i = 0; i < reader_runtime_args.size(); i++) {
            printf("  [%zu] = %u\n", i, reader_runtime_args[i]);
        }

        printf("RECEIVER WRITER runtime args (%zu total):\n", writer_runtime_args.size());
        for (size_t i = 0; i < writer_runtime_args.size(); i++) {
            printf("  [%zu] = %u\n", i, writer_runtime_args[i]);
        }
    }
    printf("=== END RUNTIME ARGS DEBUG ===\n");

    // Create runtime args callback for dynamic updates
    auto override_runtime_arguments_callback =
        [reader_kernel_id, writer_kernel_id, core, is_root](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            if (reader_kernel_id.has_value()) {
                auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id.value(), core);
                reader_runtime_args[0] = input.buffer()->address();
            }

            if (writer_kernel_id.has_value()) {
                auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id.value(), core);
                if (is_root) {
                    // Root writer: sp_fabric_arg_start_index (0), tp_receiver_base_address (1), ..., sp_tensor_address
                    // (11)
                    writer_runtime_args[1] = output.buffer()->address();   // tp_receiver_base_address
                    writer_runtime_args[11] = output.buffer()->address();  // sp_tensor_address
                } else {
                    // Other writers: tensor address is at index 0
                    writer_runtime_args[0] = output.buffer()->address();
                }
            }
        };

    return {std::move(program), override_runtime_arguments_callback};
}

}  // namespace ttnn
