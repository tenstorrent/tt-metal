// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "cpp/ttnn/operations/ccl/all_to_all_combine/device/all_to_all_combine_device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>

namespace ttnn::operations::ccl {

AllToAllCombineDeviceOperation::AllToAllCombineFromSparse::cached_mesh_workload_t
AllToAllCombineDeviceOperation::AllToAllCombineFromSparse::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;
    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program =
            create_at(operation_attributes, coord, tensor_coords.coords(), tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllToAllCombineDeviceOperation::AllToAllCombineFromSparse::shared_variables_t>
AllToAllCombineDeviceOperation::AllToAllCombineFromSparse::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const std::vector<ttnn::MeshCoordinate>& all_mesh_coordinates,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;
    const auto& output_tensor = tensor_return_value;
    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    const auto input_dtype = input_tensor.get_dtype();

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    const auto src_physical_device_id = mesh_device->get_device(mesh_coordinate)->id();

    const auto fabric_node_id = get_fabric_node_id_from_physical_chip_id(src_physical_device_id);
    const uint32_t src_mesh_id = *fabric_node_id.mesh_id;
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const auto& input_shape = input_tensor.get_tensor_spec().logical_shape();
    const auto& mapping_shape = mapping_tensor.get_tensor_spec().logical_shape();
    const auto& metadata_shape = metadata_tensor.get_tensor_spec().logical_shape();

    const uint32_t num_devices = mesh_view.num_devices();
    const uint32_t hidden_size = input_shape[-1];
    const uint32_t batch_size = metadata_shape[1];
    const uint32_t seq_size = metadata_shape[2];
    const uint32_t selected_experts_k = metadata_shape[-1];
    const uint32_t experts = mapping_shape[-2];

    TT_FATAL(experts % num_devices == 0, "Currently assuming that experts are evenly split among devices");
    const uint32_t experts_per_device = experts / num_devices;

    const auto& input_spec = input_tensor.get_tensor_spec();
    const auto& mapping_spec = mapping_tensor.get_tensor_spec();
    const auto& metadata_spec = metadata_tensor.get_tensor_spec();

    const bool input_is_dram = input_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const bool output_is_dram = output_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const bool mapping_is_dram = mapping_tensor.buffer()->buffer_type() == BufferType::DRAM;
    const bool metadata_is_dram = metadata_tensor.buffer()->buffer_type() == BufferType::DRAM;

    const auto input_page_size_bytes = input_spec.compute_page_size_bytes();
    const auto mapping_page_size_bytes = mapping_spec.compute_page_size_bytes();
    const auto metadata_page_size_bytes = metadata_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto dram_alignment = hal::get_dram_alignment();

    const auto aligned_input_page_size_bytes = tt::align(input_page_size_bytes, input_is_dram? dram_alignment:l1_alignment);
    const auto aligned_mapping_page_size_bytes = tt::align(mapping_page_size_bytes, l1_alignment);
    const auto aligned_metadata_page_size_bytes = tt::align(metadata_page_size_bytes, l1_alignment);

    auto input_data_format = datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto mapping_data_format = datatype_to_dataformat_converter(mapping_tensor.get_dtype());
    auto metadata_data_format = datatype_to_dataformat_converter(metadata_tensor.get_dtype());

    // Anything less will lead to deadlocks. It's clear why, TODO fix it.
    const uint32_t buffering_factor = experts_per_device;

    // input sharded buffer
    const auto data_cb_id = tt::CBIndex::c_0;
    CircularBufferConfig cb_data_config =
        CircularBufferConfig(buffering_factor * aligned_input_page_size_bytes, {{data_cb_id, input_data_format}})
            .set_page_size(data_cb_id, aligned_input_page_size_bytes);

    // full mapping buffer
    const auto mapping_tensor_cb_id = tt::CBIndex::c_1;
    CircularBufferConfig cb_mapping_tensor_config =
        CircularBufferConfig(aligned_mapping_page_size_bytes, {{mapping_tensor_cb_id, mapping_data_format}})
            .set_page_size(mapping_tensor_cb_id, aligned_mapping_page_size_bytes);

    // scratch space to store and share indices of per device experts
    const auto local_experts_cb_id = tt::CBIndex::c_2;
    using local_experts_t = uint16_t;
    const auto aligned_local_expert_page_size_bytes =
        tt::align(experts_per_device * sizeof(local_experts_t), l1_alignment);
    const auto local_experts_dataformat = datatype_to_dataformat_converter(convert_to_data_type<local_experts_t>());
    CircularBufferConfig cb_local_experts_config =
        CircularBufferConfig(aligned_local_expert_page_size_bytes, {{local_experts_cb_id, local_experts_dataformat}})
            .set_page_size(local_experts_cb_id, aligned_local_expert_page_size_bytes);

    // metadata page buffer
    const auto metadata_cb_id = tt::CBIndex::c_3;
    CircularBufferConfig cb_metadata_config =
        CircularBufferConfig(aligned_metadata_page_size_bytes, {{metadata_cb_id, metadata_data_format}})
            .set_page_size(metadata_cb_id, aligned_metadata_page_size_bytes);

    // client interface
    constexpr auto num_headers = 2;  // data unicast headers and atomic inc "multicast" headers
    const auto client_interface_cb_id = tt::CBIndex::c_4;
    CircularBufferConfig client_interface_cb_config =
        CircularBufferConfig(num_headers * CLIENT_INTERFACE_SIZE, {{client_interface_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(client_interface_cb_id, CLIENT_INTERFACE_SIZE);

    const auto subdevice_id = operation_attributes.subdevice_id.value_or(mesh_device->get_sub_device_ids().at(0));
    auto subdevice_core_range_set = mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, subdevice_id);

    const auto subdevice_cores = corerange_to_cores(subdevice_core_range_set);

    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    std::vector<CoreCoord> sender_cores;

    // select
    for (uint32_t i = 0; i < num_links; i++) {
        sender_cores.push_back(subdevice_cores.at(i));
    }

    // select the first core as the sender core for now, in the future we will distribute the work evenly across links
    auto sender_core = sender_cores.at(0);

    // create circular buffers
    const auto input_cb_handle = CreateCircularBuffer(program, sender_core, cb_data_config);
    const auto mapping_cb_handle = CreateCircularBuffer(program, sender_core, cb_mapping_tensor_config);
    const auto local_experts_cb_handle = CreateCircularBuffer(program, sender_core, cb_local_experts_config);
    const auto metadata_cb_handle = CreateCircularBuffer(program, sender_core, cb_metadata_config);
    const auto client_interface_cb = CreateCircularBuffer(program, sender_core, client_interface_cb_config);

    const uint32_t flat_mesh_idx = mesh_coordinate[0] * mesh_view.num_cols() + mesh_coordinate[1];

    const std::vector<uint32_t> reader_compile_time_args = {
        mapping_tensor_cb_id,
        local_experts_cb_id,
        metadata_cb_id,
        data_cb_id,
        experts_per_device,
        batch_size,
        seq_size,
        experts,  // same as num_mapping_pages
        flat_mesh_idx,
        input_page_size_bytes,
        selected_experts_k,
        mapping_page_size_bytes,
        metadata_page_size_bytes,
        input_is_dram,
        mapping_is_dram,
        metadata_is_dram};

    const DataMovementConfig reader_config{
        .processor = DataMovementProcessor::RISCV_1, .noc = NOC::NOC_1, .compile_args = reader_compile_time_args};

    KernelHandle ternary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp",
        sender_core,
        reader_config);

    const auto& axis = operation_attributes.axis;

    const uint32_t batch_replicate_dim = axis.has_value() ? mesh_device->shape()[axis.value()] : 1;
    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes =
        input_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;

    const std::vector<uint32_t> writer_compile_time_args = {
        metadata_cb_id,
        local_experts_cb_id,
        client_interface_cb_id,
        data_cb_id,
        batch_size,
        seq_size,
        selected_experts_k,
        experts_per_device,
        num_devices,
        src_chip_id,
        input_page_size_bytes,
        l1_alignment,
        output_is_dram,
        mesh_view.num_rows(),
        mesh_view.num_cols(),
        max_packet_size_bytes,
    };

    // fabric routing info
    std::vector<uint32_t> dest_mesh_id, dest_chip_id, route;
    for (const auto& coord : all_mesh_coordinates) {
        auto device = mesh_device->get_device(coord);
        auto fabric_node_id = get_fabric_node_id_from_physical_chip_id(device->id());
        dest_mesh_id.push_back(*fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)fabric_node_id.chip_id);
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
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/writer_all_to_all_combine.cpp",
        sender_core,
        writer_config);

    std::vector<uint32_t> reader_runtime_args = {
        mapping_tensor.mesh_buffer()->get_device_buffer(mesh_coordinate)->address(),
        metadata_tensor.mesh_buffer()->get_device_buffer(mesh_coordinate)->address(),
        input_tensor.mesh_buffer()->get_device_buffer(mesh_coordinate)->address(),
    };

    std::vector<uint32_t> writer_runtime_args = {
        output_tensor.mesh_buffer()->get_device_buffer(mesh_coordinate)->address(),
        operation_attributes.cross_device_semaphore.address(),
    };
    for (auto& neighbor : neighbors) {
        auto neighbor_coordinate = mesh_view.find_device(neighbor->id());
        uint32_t link_id = common::select_link(mesh_view, mesh_coordinate, neighbor_coordinate, num_links, topology);
        const auto neighbor_fabric_id = get_fabric_node_id_from_physical_chip_id(neighbor->id());
        append_fabric_connection_rt_args(
            fabric_node_id, neighbor_fabric_id, link_id, program, sender_core, writer_runtime_args);
    }

    SetRuntimeArgs(program, ternary_reader_kernel_id, sender_cores.at(0), reader_runtime_args);
    SetRuntimeArgs(program, unary_writer_kernel_id, sender_cores.at(0), writer_runtime_args);

    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .core = sender_cores.at(0)}};
}

void AllToAllCombineDeviceOperation::AllToAllCombineFromSparse::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto & coord = range.start_coord();
        TT_FATAL(coord == range.end_coord(), "Expected single coordinate per program but got range of {} to {}", coord, range.end_coord());

        const auto& shared_variables = cached_workload.shared_variables.at(range);
        auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        auto& unary_writer_kernel_id = shared_variables.unary_writer_kernel_id;
        auto& core = shared_variables.core;

        auto& reader_runtime_args = GetRuntimeArgs(program, ternary_reader_kernel_id, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);

        reader_runtime_args.at(0) = tensor_args.mapping_tensor.mesh_buffer()->get_device_buffer(coord)->address();
        reader_runtime_args.at(1) = tensor_args.metadata_tensor.mesh_buffer()->get_device_buffer(coord)->address();
        reader_runtime_args.at(2) = tensor_args.input_tensor.mesh_buffer()->get_device_buffer(coord)->address();

        writer_runtime_args.at(0) = tensor_return_value.mesh_buffer()->get_device_buffer(coord)->address();
        writer_runtime_args.at(1) = operation_attributes.cross_device_semaphore.address();
    }
}

}  // namespace ttnn::operations::ccl
