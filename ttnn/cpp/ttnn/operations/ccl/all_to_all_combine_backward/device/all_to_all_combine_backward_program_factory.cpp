// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "cpp/ttnn/operations/ccl/all_to_all_combine_backward/device/all_to_all_combine_backward_device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

AllToAllCombineBackwardDeviceOperation::AllToAllCombineBackwardToDense::cached_mesh_workload_t
AllToAllCombineBackwardDeviceOperation::AllToAllCombineBackwardToDense::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.grad_output.device();
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    auto final_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, {});

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_coords.coords(),
            tensor_args,
            tensor_return_value,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<
    AllToAllCombineBackwardDeviceOperation::AllToAllCombineBackwardToDense::shared_variables_t>
AllToAllCombineBackwardDeviceOperation::AllToAllCombineBackwardToDense::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    Program program{};

    const auto& grad_output = tensor_args.grad_output;
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;
    const auto& output_tensor = tensor_return_value;
    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    const auto grad_dtype = grad_output.dtype();

    auto* mesh_device = grad_output.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const auto& mapping_shape = mapping_tensor.tensor_spec().logical_shape();
    const auto& metadata_shape = metadata_tensor.tensor_spec().logical_shape();

    const uint32_t num_devices = mesh_view.num_devices();
    const uint32_t batch_size = metadata_shape[1];   // global batch
    const uint32_t seq_size = metadata_shape[2];     // global seq
    const uint32_t selected_experts_k = metadata_shape[-1];
    const uint32_t experts = mapping_shape[-2];

    TT_FATAL(experts % num_devices == 0, "Currently assuming that experts are evenly split among devices");

    const auto& grad_spec = grad_output.tensor_spec();
    const auto& mapping_spec = mapping_tensor.tensor_spec();
    const auto& metadata_spec = metadata_tensor.tensor_spec();

    const bool grad_is_dram = grad_output.buffer()->buffer_type() == BufferType::DRAM;

    const auto grad_page_size_bytes = grad_spec.compute_page_size_bytes();
    const auto mapping_page_size_bytes = mapping_spec.compute_page_size_bytes();
    const auto metadata_page_size_bytes = metadata_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto dram_alignment = hal::get_dram_alignment();

    const auto aligned_grad_page_size_bytes =
        tt::align(grad_page_size_bytes, grad_is_dram ? dram_alignment : l1_alignment);
    const auto aligned_mapping_page_size_bytes = tt::align(mapping_page_size_bytes, l1_alignment);
    const auto aligned_metadata_page_size_bytes = tt::align(metadata_page_size_bytes, l1_alignment);

    const auto grad_data_format = datatype_to_dataformat_converter(grad_output.dtype());
    const auto mapping_data_format = datatype_to_dataformat_converter(mapping_tensor.dtype());
    const auto metadata_data_format = datatype_to_dataformat_converter(metadata_tensor.dtype());

    // Buffering factor: enough to pipeline K grad pages per token
    const uint32_t buffering_factor = selected_experts_k;

    // CB c_0: grad data pages (K per token)
    constexpr auto data_cb_id = tt::CBIndex::c_0;
    CircularBufferConfig cb_data_config =
        CircularBufferConfig(buffering_factor * aligned_grad_page_size_bytes, {{data_cb_id, grad_data_format}})
            .set_page_size(data_cb_id, aligned_grad_page_size_bytes);

    // CB c_1: mapping page temp buffer
    constexpr auto mapping_tensor_cb_id = tt::CBIndex::c_1;
    CircularBufferConfig cb_mapping_tensor_config =
        CircularBufferConfig(aligned_mapping_page_size_bytes, {{mapping_tensor_cb_id, mapping_data_format}})
            .set_page_size(mapping_tensor_cb_id, aligned_mapping_page_size_bytes);

    // CB c_2: expert_to_device[] and expert_to_local_idx[] arrays (2 * num_experts * uint16_t)
    constexpr auto expert_device_map_cb_id = tt::CBIndex::c_2;
    using expert_map_t = uint16_t;
    const auto aligned_expert_map_page_size_bytes =
        tt::align(2 * experts * sizeof(expert_map_t), l1_alignment);
    const auto expert_map_dataformat = datatype_to_dataformat_converter(convert_to_data_type<expert_map_t>());
    CircularBufferConfig cb_expert_device_map_config =
        CircularBufferConfig(
            aligned_expert_map_page_size_bytes, {{expert_device_map_cb_id, expert_map_dataformat}})
            .set_page_size(expert_device_map_cb_id, aligned_expert_map_page_size_bytes);

    // CB c_3: metadata pages
    constexpr auto metadata_cb_id = tt::CBIndex::c_3;
    CircularBufferConfig cb_metadata_config =
        CircularBufferConfig(aligned_metadata_page_size_bytes, {{metadata_cb_id, metadata_data_format}})
            .set_page_size(metadata_cb_id, aligned_metadata_page_size_bytes);

    // CB c_4: packet headers (data unicast + atomic inc)
    constexpr auto num_headers = 2;
    constexpr auto client_interface_cb_id = tt::CBIndex::c_4;
    CircularBufferConfig client_interface_cb_config =
        CircularBufferConfig(num_headers * CLIENT_INTERFACE_SIZE, {{client_interface_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(client_interface_cb_id, CLIENT_INTERFACE_SIZE);

    const auto subdevice_cores = corerange_to_cores(operation_attributes.worker_core_range_set);

    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    // Work-split: tokens across links/cores
    // In the backward each device has tokens_per_device local tokens
    const auto& axis = operation_attributes.axis;
    const uint32_t replicate_dim =
        axis.has_value() ? (uint32_t)mesh_device->shape()[!axis.value()] : 1;
    const uint32_t replicate_group_devices =
        axis.has_value() ? num_devices / replicate_dim : num_devices;
    const uint32_t tokens_per_device = batch_size * seq_size / replicate_group_devices;

    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);
    uint32_t num_cores = std::min(num_links, tt::div_up(tokens_per_device, tokens_per_core));
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, operation_attributes.worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    // Create circular buffers
    CreateCircularBuffer(program, sender_core_grid, cb_data_config);
    CreateCircularBuffer(program, sender_core_grid, cb_mapping_tensor_config);
    CreateCircularBuffer(program, sender_core_grid, cb_expert_device_map_config);
    CreateCircularBuffer(program, sender_core_grid, cb_metadata_config);
    CreateCircularBuffer(program, sender_core_grid, client_interface_cb_config);

    const uint32_t flat_mesh_idx = common::get_linearized_index(mesh_coordinate, mesh_view);

    // Compute device_in_group and token_global_device_start
    // This determines which slice of global tokens this device "owns" in the forward output.
    uint32_t device_in_group;
    if (!axis.has_value()) {
        device_in_group = flat_mesh_idx;
    } else if (axis.value() == 0) {
        // axis=0 → ReplicateGroup::COLS → groups are columns; device_in_group = row index
        device_in_group = flat_mesh_idx / mesh_view.num_cols();
    } else {
        // axis=1 → ReplicateGroup::ROWS → groups are rows; device_in_group = col index
        device_in_group = flat_mesh_idx % mesh_view.num_cols();
    }
    const uint32_t token_global_device_start = device_in_group * tokens_per_device;

    // Reader compile-time args
    // Reader reads: mapping (to build expert maps), metadata, grad_output
    std::vector<uint32_t> reader_compile_time_args = {
        mapping_tensor_cb_id,
        expert_device_map_cb_id,
        metadata_cb_id,
        data_cb_id,
        experts,             // total number of experts (for building expert map)
        num_devices,         // to find which device has each expert
        batch_size,          // global batch
        seq_size,            // global seq
        selected_experts_k,  // K
        flat_mesh_idx,
        grad_page_size_bytes,
        mapping_page_size_bytes,
        metadata_page_size_bytes,
        operation_attributes.locally_reduced,
    };
    TensorAccessorArgs(grad_output.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(mapping_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(metadata_tensor.buffer()).append_to(reader_compile_time_args);

    const DataMovementConfig reader_config{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
        .compile_args = reader_compile_time_args};

    KernelHandle ternary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine_backward/device/kernels/dataflow/"
        "reader_all_to_all_combine_backward.cpp",
        sender_core_grid,
        reader_config);

    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes =
        grad_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes)
                                         : fabric_max_packet_size_bytes;

    // Writer compile-time args
    // Writer routes each (token, k) grad page to the expert device
    std::vector<uint32_t> writer_compile_time_args = {
        metadata_cb_id,
        expert_device_map_cb_id,
        client_interface_cb_id,
        data_cb_id,
        batch_size,          // global batch
        seq_size,            // global seq
        selected_experts_k,  // K
        experts,             // total experts (for indexing)
        num_devices,
        src_chip_id,
        grad_page_size_bytes,
        l1_alignment,
        mesh_view.num_rows(),
        mesh_view.num_cols(),
        max_packet_size_bytes,
        flat_mesh_idx,
        (uint32_t)topology,
        operation_attributes.locally_reduced,
    };
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    // Fabric routing info
    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : all_mesh_coordinates) {
        const auto fni = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*fni.mesh_id);
        dest_chip_id.push_back((uint32_t)fni.chip_id);
    }
    const auto [neighbors, directions] = common::get_neighbors(mesh_view, mesh_coordinate, topology, axis);

    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", common::stringify(dest_mesh_id)},
        {"DIRECTIONS", common::stringify(directions)}};

    if (axis.has_value()) {
        writer_defines["REPLICATE_GROUP_AXIS"] = std::to_string(axis.value());
    }

    const DataMovementConfig writer_config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_0,
        .compile_args = writer_compile_time_args,
        .defines = writer_defines};

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine_backward/device/kernels/dataflow/"
        "writer_all_to_all_combine_backward.cpp",
        sender_core_grid,
        writer_config);

    // Runtime args
    // Reader: [mapping_addr, metadata_addr, grad_addr, token_global_start, token_global_end,
    //          token_global_device_start, tokens_per_device]
    // Writer: [output_addr, cross_semaphore, init_semaphore, token_global_start, token_global_end, ...fabric]
    std::vector<uint32_t> reader_runtime_args = {
        mapping_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        grad_output.buffer()->address(),
        0,  // token_global_start (updated per core)
        0,  // token_global_end   (updated per core)
        token_global_device_start,
        tokens_per_device,
    };

    uint32_t link_id = 0;
    uint32_t tokens_per_core_start = 0;
    for (const auto& sender_core : sender_cores) {
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),
            (uint32_t)cross_device_semaphore.address(),
            (uint32_t)init_semaphore.address(),
            0,  // token_global_start (updated per core)
            0,  // token_global_end   (updated per core)
        };
        const uint32_t core_token_end =
            std::min(tokens_per_core_start + tokens_per_core, tokens_per_device);
        reader_runtime_args[3] = token_global_device_start + tokens_per_core_start;
        reader_runtime_args[4] = token_global_device_start + core_token_end;
        writer_runtime_args[3] = reader_runtime_args[3];
        writer_runtime_args[4] = reader_runtime_args[4];
        tokens_per_core_start = core_token_end;

        for (const auto& neighbor_coordinate : neighbors) {
            const auto neighbor_fabric_id = mesh_device->get_fabric_node_id(neighbor_coordinate);
            append_fabric_connection_rt_args(
                fabric_node_id, neighbor_fabric_id, link_id, program, sender_core, writer_runtime_args);
        }
        SetRuntimeArgs(program, ternary_reader_kernel_id, sender_core, reader_runtime_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, sender_core, writer_runtime_args);
        link_id++;
    }

    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .cores = sender_cores,
         .init_semaphore = init_semaphore,
         .cross_device_semaphore = cross_device_semaphore}};
}

void AllToAllCombineBackwardDeviceOperation::AllToAllCombineBackwardToDense::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = range.start_coord();
        TT_FATAL(
            coord == range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            range.end_coord());

        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        const auto& unary_writer_kernel_id = shared_variables.unary_writer_kernel_id;
        const auto& cores = shared_variables.cores;

        for (const auto& core : cores) {
            auto& reader_runtime_args = GetRuntimeArgs(program, ternary_reader_kernel_id, core);
            auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);

            // Indices 0-2: tensor addresses; 3-6: token range (preserved from create_at)
            reader_runtime_args.at(0) = tensor_args.mapping_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.metadata_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.grad_output.buffer()->address();

            writer_runtime_args.at(0) = tensor_return_value.buffer()->address();
            writer_runtime_args.at(1) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(2) = (uint32_t)shared_variables.init_semaphore.address();
        }
    }
}

}  // namespace ttnn::operations::ccl
