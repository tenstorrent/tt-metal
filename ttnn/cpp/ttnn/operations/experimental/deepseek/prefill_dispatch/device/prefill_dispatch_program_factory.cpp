// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "prefill_dispatch_device_operation.hpp"
#include <algorithm>
#include <array>
#include <utility>
#include <limits>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <ttnn/global_semaphore.hpp>
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
namespace ttnn::operations::experimental::deepseek::prefill_dispatch {

namespace detail {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

void create_tensor_cb(
    tt::tt_metal::Program& program,
    const CoreRangeSet& core_range_set,
    const ttnn::Tensor& tensor,
    uint32_t buffering_factor,
    tt::CBIndex cb_id,
    const std::string& tensor_name = "tensor") {
    auto page_size = get_page_size(tensor);
    auto num_pages = detail::get_num_pages(tensor);
    auto aligned_page_size = get_aligned_page_size(tensor);
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());

    uint32_t cb_size = buffering_factor * aligned_page_size;

    log_debug(
        tt::LogOp,
        "{} shape: {}, pages: {}, page_size: {}, aligned_page_size: {} buffering_factor: {} cb_id: {} cb_size: {} "
        "cb_dtype: {}",
        tensor_name,
        tensor.logical_shape(),
        num_pages,
        page_size,
        aligned_page_size,
        buffering_factor,
        cb_id,
        cb_size,
        data_format);

    tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(cb_size, {{cb_id, data_format}}).set_page_size(cb_id, aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_config);
}

}  // namespace detail

PrefillDispatchDeviceOperation::PrefillDispatchProgramFactory::cached_mesh_workload_t
PrefillDispatchDeviceOperation::PrefillDispatchProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();

    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            tensor_coords,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<PrefillDispatchDeviceOperation::PrefillDispatchProgramFactory::shared_variables_t>
PrefillDispatchDeviceOperation::PrefillDispatchProgramFactory::create_at(
    const operation_attributes_t& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    tt::tt_metal::Program program{};

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.indices_tensor;
    auto weights_tensor = tensor_args.weights_tensor;
    auto offsets_tensor = tensor_args.chip_to_n_routed_expert_offset_tensor;
    auto dispatch_table_tensor = tensor_args.expert_dispatch_table_tensor;

    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);

    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;
    log_debug(
        tt::LogOp,
        "Creating prefill dispatch program for mesh coordinate: ({}, {}) with topology: {} num_links: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        topology,
        num_links);

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    log_debug(
        tt::LogOp,
        "\nCreating all to all dispatch program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized mesh coord: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord);

    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    // When num_links=0 (fabric disabled), we use 1 core for local dispatch
    uint32_t effective_num_links = std::max(num_links, 1u);
    TT_FATAL(
        subdevice_cores.size() >= effective_num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        effective_num_links);

    // Figure out worker cores
    auto logical_volume = input_tensor.logical_shape().volume();
    auto hidden_size = input_tensor.logical_shape()[-1];
    auto tokens_per_device = logical_volume / hidden_size;

    // effective_num_links already calculated above (when num_links=0, use 1 core for local dispatch)
    uint32_t tokens_per_core = tt::div_up(tokens_per_device, effective_num_links);
    uint32_t num_cores = std::min<uint32_t>(effective_num_links, tt::div_up(tokens_per_device, tokens_per_core));
    log_debug(
        tt::LogOp,
        "num_links{}: tokens_per_device: {}, tokens_per_core: {}, num_cores: {}",
        num_links,
        tokens_per_device,
        tokens_per_core,
        num_cores);
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);
    log_debug(
        tt::LogOp,
        "Selected sender cores for mesh coordinate ({}, {}): {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        sender_cores);

    // Create reader CBs
    detail::create_tensor_cb(
        program, sender_core_grid, input_tensor, /*buffering_factor=*/3, /*cb_id=*/tt::CBIndex::c_0, "input_tensor");
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        indices_tensor,
        /*buffering_factor=*/3,
        /*cb_id=*/tt::CBIndex::c_1,
        "indices_tensor");
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        weights_tensor,
        /*buffering_factor=*/3,
        /*cb_id=*/tt::CBIndex::c_2,
        "weights_tensor");
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        offsets_tensor,
        /*buffering_factor=*/detail::get_num_pages(offsets_tensor),  // everyone reads entire offsets tensor
        /*cb_id=*/tt::CBIndex::c_3,
        "offsets_tensor");
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        dispatch_table_tensor,
        /*buffering_factor=*/detail::get_num_pages(dispatch_table_tensor),  // everyone reads entire dispatch table
        /*cb_id=*/tt::CBIndex::c_9,
        "dispatch_table_tensor");

    // create writer CBs
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        output_tensor,
        /*buffering_factor=*/2,
        /*cb_id=*/tt::CBIndex::c_4,
        "output_tensor");
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        metadata_tensor,
        /*buffering_factor=*/2,
        /*cb_id=*/tt::CBIndex::c_5,
        "metadata_tensor");

    // Create CB for temporary metadata buffer (used by writer kernel)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        metadata_tensor,
        /*buffering_factor=*/1,  // Only need 1 page at a time
        /*cb_id=*/tt::CBIndex::c_7,
        "metadata_temp_buffer");

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    // Create packet header CB for fabric sends (if fabric is enabled)
    if (operation_attributes.num_links > 0) {
        constexpr uint32_t num_packet_headers = 2;  // unicast + metadata
        auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        uint32_t packet_header_cb_size = num_packet_headers * packet_header_size_bytes;

        tt::tt_metal::CircularBufferConfig packet_header_cb_config =
            tt::tt_metal::CircularBufferConfig(packet_header_cb_size, {{tt::CBIndex::c_8, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_8, packet_header_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);

        log_debug(
            tt::LogOp,
            "Created packet header CB: packet_header_size_bytes={}, total_cb_size={}, num_headers={}, topology={}",
            packet_header_size_bytes,
            packet_header_cb_size,
            num_packet_headers,
            static_cast<int>(topology));
    }
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    log_debug(tt::LogOp, "dest_chip_id: {}", ccl::common::stringify(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", ccl::common::stringify(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", ccl::common::stringify(directions));

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    log_debug(
        tt::LogOp, "Fabric max packet size: {} bytes, L1 alignment: {} bytes", fabric_max_packet_size, l1_alignment);

    // tt::tt_metal::KernelHandle reader_kernel_id{};  // placeholder
    // tt::tt_metal::KernelHandle writer_kernel_id{};  // placeholder
    // std::vector<CoreCoord> cores;                   // placeholder

    // create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        // CB IDs (7)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_input_id
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_indices_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_weights_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_offsets_id
        static_cast<uint32_t>(tt::CBIndex::c_7),  // cb_metadata_temp_id
        static_cast<uint32_t>(tt::CBIndex::c_8),  // cb_packet_header_id
        static_cast<uint32_t>(tt::CBIndex::c_9),  // cb_dispatch_table_id

        // Page counts (7)
        detail::get_num_pages(input_tensor),
        detail::get_num_pages(indices_tensor),
        detail::get_num_pages(weights_tensor),
        detail::get_num_pages(offsets_tensor),
        detail::get_num_pages(output_tensor),
        detail::get_num_pages(metadata_tensor),
        detail::get_num_pages(dispatch_table_tensor),

        // Page sizes (7)
        detail::get_page_size(input_tensor),
        detail::get_page_size(indices_tensor),
        detail::get_page_size(weights_tensor),
        detail::get_page_size(offsets_tensor),
        detail::get_page_size(output_tensor),
        detail::get_page_size(metadata_tensor),
        detail::get_page_size(dispatch_table_tensor),

        // Operation parameters (8)
        mesh_view.num_devices(),  // num_devices
        (uint32_t)hidden_size,
        operation_attributes.experts_per_chip,
        operation_attributes.n_routed_experts,
        operation_attributes.num_experts_per_tok,
        operation_attributes.metadata_len,
        operation_attributes.max_dispatched_tokens_per_expert,
        (uint32_t)tokens_per_device,

        // Mesh information (5)
        src_mesh_id,
        src_chip_id,
        mesh_view.num_rows(),
        mesh_view.num_cols(),
        linearized_mesh_coord,

        // Aligned page sizes (7)
        detail::get_aligned_page_size(input_tensor),
        detail::get_aligned_page_size(indices_tensor),
        detail::get_aligned_page_size(weights_tensor),
        detail::get_aligned_page_size(offsets_tensor),
        detail::get_aligned_page_size(output_tensor),
        detail::get_aligned_page_size(metadata_tensor),
        detail::get_aligned_page_size(dispatch_table_tensor),

        // Fabric configuration (4)
        (uint32_t)fabric_max_packet_size,
        l1_alignment,
        operation_attributes.num_links,
        static_cast<uint32_t>(topology),
    };

    // Append TensorAccessorArgs for all 7 tensors
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weights_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(offsets_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatch_table_tensor.buffer()).append_to(reader_compile_time_args);

    std::map<std::string, std::string> reader_defines = {
        {"AXIS", std::to_string(operation_attributes.axis.has_value() ? operation_attributes.axis.value() : -1)},
    };
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/prefill_dispatch/device/kernels/dataflow/"
        "reader_prefill_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    // create writer kernel - shares same compile time args as reader
    const auto& writer_compile_time_args = reader_compile_time_args;

    // Code-gen a mesh-position to fabric chip ID array for the writer kernel
    // Code-gen a mesh-position to mesh-id array for the writer kernel
    // Code-gen a direction array that is set to true when a direction has a valid connection (when a neighbor exists or
    // if it's along a valid cluster axis)
    std::map<std::string, std::string> writer_defines;

    // Only enable fabric if num_links > 0 (explicitly requested by user)
    if (operation_attributes.num_links > 0) {
        writer_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        writer_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        writer_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }

    if (operation_attributes.axis.has_value()) {
        writer_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek/prefill_dispatch/device/kernels/dataflow/"
        "writer_prefill_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});

    // Set up runtime args for all cores
    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        weights_tensor.buffer()->address(),
        offsets_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        dispatch_table_tensor.buffer()->address(),
        (uint32_t)cross_device_semaphore.address(),
        (uint32_t)init_semaphore.address(),
        0,  // token_start_idx (set per core)
        0,  // token_end_idx (set per core)
    };

    // Distribute work across cores
    uint32_t tokens_per_core_start = 0;
    for (const auto& sender_core : sender_cores) {
        std::vector<uint32_t> writer_runtime_args = reader_runtime_args;  // Copy base args

        // Set token range for this core
        reader_runtime_args[9] = tokens_per_core_start;  // token_start_idx
        reader_runtime_args[10] =
            std::min(tokens_per_core_start + tokens_per_core, (uint32_t)tokens_per_device);  // token_end_idx
        writer_runtime_args[9] = tokens_per_core_start;
        writer_runtime_args[10] = reader_runtime_args[10];

        tokens_per_core_start = reader_runtime_args[10];

        // Append fabric connection args for each neighbor (only if fabric is enabled)
        if (operation_attributes.num_links > 0) {
            for (const auto& neighbor_coordinate : neighbors) {
                // Skip self-connections (same chip)
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    log_debug(
                        tt::LogOp,
                        "Skipping self-connection for mesh coord ({}, {}) at core {}",
                        mesh_coordinate[0],
                        mesh_coordinate[1],
                        sender_core);
                    continue;
                }

                log_debug(
                    tt::LogOp,
                    "Connection between mesh coord ({}, {}) and ({}, {}) at core {} and handles "
                    "token indices from {} to {}",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    neighbor_coordinate[0],
                    neighbor_coordinate[1],
                    sender_core,
                    reader_runtime_args[9],
                    reader_runtime_args[10]);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    0,  // link_id - use 0 for single link
                    program,
                    sender_core,
                    writer_runtime_args);
            }
        }

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, sender_core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_core, writer_runtime_args);
        // link_id++;  // Unused when fabric connection setup is disabled
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = sender_cores,
         .init_semaphore = init_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

void PrefillDispatchDeviceOperation::PrefillDispatchProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& reader_kernel_id = shared_variables.reader_kernel_id;
        const auto& writer_kernel_id = shared_variables.writer_kernel_id;
        const auto& cores = shared_variables.cores;

        const auto& output_tensor = tensor_return_value.at(0);
        const auto& metadata_tensor = tensor_return_value.at(1);

        for (const auto& core : cores) {
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);

            // Update buffer addresses for all 7 tensors (indices 0-6)
            reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.indices_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.weights_tensor.buffer()->address();
            reader_runtime_args.at(3) = tensor_args.chip_to_n_routed_expert_offset_tensor.buffer()->address();
            reader_runtime_args.at(4) = output_tensor.buffer()->address();
            reader_runtime_args.at(5) = metadata_tensor.buffer()->address();
            reader_runtime_args.at(6) = tensor_args.expert_dispatch_table_tensor.buffer()->address();

            // Update semaphore addresses (indices 7-8)
            reader_runtime_args.at(7) = (uint32_t)shared_variables.cross_device_semaphore.address();
            reader_runtime_args.at(8) = (uint32_t)shared_variables.init_semaphore.address();

            // Update writer runtime args (same first 9 args)
            writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.indices_tensor.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.weights_tensor.buffer()->address();
            writer_runtime_args.at(3) = tensor_args.chip_to_n_routed_expert_offset_tensor.buffer()->address();
            writer_runtime_args.at(4) = output_tensor.buffer()->address();
            writer_runtime_args.at(5) = metadata_tensor.buffer()->address();
            writer_runtime_args.at(6) = tensor_args.expert_dispatch_table_tensor.buffer()->address();
            writer_runtime_args.at(7) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(8) = (uint32_t)shared_variables.init_semaphore.address();

            // Note: token ranges (indices 9-10) and fabric args remain unchanged
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek::prefill_dispatch
