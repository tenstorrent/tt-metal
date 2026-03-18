// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dispatch_device_operation.hpp"
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
namespace ttnn::operations::experimental::deepseek_prefill::dispatch {

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

DispatchProgramFactory::cached_mesh_workload_t DispatchProgramFactory::create_mesh_workload(
    const DispatchParams& operation_attributes,
    const MeshCoordinateRangeSet& tensor_coords,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, DispatchSharedVariables> shared_variables;

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

ttnn::device_operation::CachedProgram<DispatchSharedVariables> DispatchProgramFactory::create_at(
    const DispatchParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value,
    const MeshCoordinateRangeSet& tensor_coords,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    tt::tt_metal::Program program{};

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.indices_tensor;
    auto weights_tensor = tensor_args.weights_tensor;
    auto offsets_tensor = tensor_args.expert_offsets_tensor;
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
    uint32_t effective_num_links = num_links >= 2 ? 2u : 1u;
    TT_FATAL(
        subdevice_cores.size() >= effective_num_links,
        "Not enough cores {} for {} links",
        subdevice_cores.size(),
        effective_num_links);

    auto logical_volume = input_tensor.logical_shape().volume();
    auto hidden_size = input_tensor.logical_shape()[-1];
    auto tokens_per_device = logical_volume / hidden_size;

    uint32_t num_cores = effective_num_links;
    log_debug(
        tt::LogOp,
        "num_links: {}, effective_num_links: {}, tokens_per_device: {}, num_cores: {}",
        num_links,
        effective_num_links,
        tokens_per_device,
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

    constexpr uint32_t read_batch_size = 8;
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // c_0: input scratch (reader-only, batched DRAM reads)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        input_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_0,
        "input_scratch");
    // c_1: indices scratch (reader-only)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        indices_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_1,
        "indices_scratch");
    // c_2: weights scratch (reader-only)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        weights_tensor,
        /*buffering_factor=*/read_batch_size,
        /*cb_id=*/tt::CBIndex::c_2,
        "weights_scratch");
    // c_3: offsets (reader-only, full tensor)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        offsets_tensor,
        /*buffering_factor=*/detail::get_num_pages(offsets_tensor),
        /*cb_id=*/tt::CBIndex::c_3,
        "offsets_tensor");

    // c_4: route_info (reader->writer, 4 x uint32_t per entry)
    {
        uint32_t route_info_page_size = l1_alignment;
        constexpr uint32_t route_info_buffering = 16;
        tt::tt_metal::CircularBufferConfig route_info_cb_config =
            tt::tt_metal::CircularBufferConfig(
                route_info_buffering * route_info_page_size, {{tt::CBIndex::c_4, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_4, route_info_page_size);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, route_info_cb_config);
    }

    // c_5: payload_for_writer (reader->writer, input pages for fabric sends)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        input_tensor,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_5,
        "payload_for_writer");
    // c_6: metadata_for_writer (reader->writer, metadata pages for fabric sends)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        metadata_tensor,
        /*buffering_factor=*/16,
        /*cb_id=*/tt::CBIndex::c_6,
        "metadata_for_writer");

    // c_7: metadata_temp (reader-only, for constructing metadata locally)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        metadata_tensor,
        /*buffering_factor=*/1,
        /*cb_id=*/tt::CBIndex::c_7,
        "metadata_temp_buffer");

    const auto [neighbors, directions] =
        ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    // c_8: packet header CB for fabric sends (writer-only)
    if (operation_attributes.num_links > 0) {
        constexpr uint32_t num_packet_headers = 2;  // unicast + metadata
        auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        uint32_t packet_header_cb_size = num_packet_headers * packet_header_size_bytes;

        tt::tt_metal::CircularBufferConfig packet_header_cb_config =
            tt::tt_metal::CircularBufferConfig(packet_header_cb_size, {{tt::CBIndex::c_8, tt::DataFormat::UInt8}})
                .set_page_size(tt::CBIndex::c_8, packet_header_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);
    }

    // c_9: dispatch_table (reader-only, full tensor)
    detail::create_tensor_cb(
        program,
        sender_core_grid,
        dispatch_table_tensor,
        /*buffering_factor=*/detail::get_num_pages(dispatch_table_tensor),
        /*cb_id=*/tt::CBIndex::c_9,
        "dispatch_table_tensor");

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
    log_debug(
        tt::LogOp, "Fabric max packet size: {} bytes, L1 alignment: {} bytes", fabric_max_packet_size, l1_alignment);

    // Compile-time args shared by reader and writer
    std::vector<uint32_t> compile_time_args = {
        // CB IDs (10)
        static_cast<uint32_t>(tt::CBIndex::c_0),  // cb_input_id
        static_cast<uint32_t>(tt::CBIndex::c_1),  // cb_indices_id
        static_cast<uint32_t>(tt::CBIndex::c_2),  // cb_weights_id
        static_cast<uint32_t>(tt::CBIndex::c_3),  // cb_offsets_id
        static_cast<uint32_t>(tt::CBIndex::c_4),  // cb_route_info_id
        static_cast<uint32_t>(tt::CBIndex::c_5),  // cb_payload_for_writer_id
        static_cast<uint32_t>(tt::CBIndex::c_6),  // cb_metadata_for_writer_id
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
        operation_attributes.num_routed_experts,
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
        (uint32_t)std::min(operation_attributes.num_links, 1u),
        static_cast<uint32_t>(topology),
    };

    // Append TensorAccessorArgs for all 7 tensors
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(weights_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(offsets_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatch_table_tensor.buffer()).append_to(compile_time_args);

    // Both reader and writer get fabric defines so the reader can compute routes
    std::map<std::string, std::string> fabric_defines;
    if (operation_attributes.num_links > 0) {
        fabric_defines["DEST_CHIP_ID"] = ccl::common::stringify(dest_chip_id);
        fabric_defines["DEST_MESH_ID"] = ccl::common::stringify(dest_mesh_id);
        fabric_defines["DIRECTIONS"] = ccl::common::stringify(directions);
    }
    if (operation_attributes.axis.has_value()) {
        fabric_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "reader_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_read(mesh_device->arch()),
            .compile_args = compile_time_args,
            .defines = fabric_defines});

    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/dispatch/device/kernels/dataflow/"
        "writer_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::detail::preferred_noc_for_dram_write(mesh_device->arch()),
            .compile_args = compile_time_args,
            .defines = fabric_defines});

    // Runtime args: all cores process all tokens, experts split round-robin
    std::vector<uint32_t> base_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        weights_tensor.buffer()->address(),
        offsets_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        dispatch_table_tensor.buffer()->address(),
        (uint32_t)cross_device_semaphore.address(),
        (uint32_t)init_semaphore.address(),
        0,                            // token_start_idx (all tokens)
        (uint32_t)tokens_per_device,  // token_end_idx (all tokens)
        0,                            // dispatch_core_idx (set per core)
        num_cores,                    // num_dispatch_cores
    };

    uint32_t core_idx = 0;
    for (const auto& sender_core : sender_cores) {
        std::vector<uint32_t> reader_runtime_args = base_runtime_args;
        std::vector<uint32_t> writer_runtime_args = base_runtime_args;

        reader_runtime_args[11] = core_idx;
        writer_runtime_args[11] = core_idx;

        if (operation_attributes.num_links > 0) {
            uint32_t core_link = core_idx % num_links;
            for (const auto& neighbor_coordinate : neighbors) {
                if (neighbor_coordinate[0] == mesh_coordinate[0] && neighbor_coordinate[1] == mesh_coordinate[1]) {
                    continue;
                }

                log_debug(
                    tt::LogOp,
                    "Connection between mesh coord ({}, {}) and ({}, {}) at core {} link {}",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    neighbor_coordinate[0],
                    neighbor_coordinate[1],
                    sender_core,
                    core_link);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    core_link,
                    program,
                    sender_core,
                    writer_runtime_args);
            }
        }

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, sender_core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, sender_core, writer_runtime_args);
        core_idx++;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = sender_cores,
         .init_semaphore = init_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

void DispatchProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const DispatchParams& /*operation_attributes*/,
    const DispatchInputs& tensor_args,
    DispatchProgramFactory::tensor_return_value_t& tensor_return_value) {
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

            reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.indices_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.weights_tensor.buffer()->address();
            reader_runtime_args.at(3) = tensor_args.expert_offsets_tensor.buffer()->address();
            reader_runtime_args.at(4) = output_tensor.buffer()->address();
            reader_runtime_args.at(5) = metadata_tensor.buffer()->address();
            reader_runtime_args.at(6) = tensor_args.expert_dispatch_table_tensor.buffer()->address();
            reader_runtime_args.at(7) = (uint32_t)shared_variables.cross_device_semaphore.address();
            reader_runtime_args.at(8) = (uint32_t)shared_variables.init_semaphore.address();

            writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.indices_tensor.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.weights_tensor.buffer()->address();
            writer_runtime_args.at(3) = tensor_args.expert_offsets_tensor.buffer()->address();
            writer_runtime_args.at(4) = output_tensor.buffer()->address();
            writer_runtime_args.at(5) = metadata_tensor.buffer()->address();
            writer_runtime_args.at(6) = tensor_args.expert_dispatch_table_tensor.buffer()->address();
            writer_runtime_args.at(7) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(8) = (uint32_t)shared_variables.init_semaphore.address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch
