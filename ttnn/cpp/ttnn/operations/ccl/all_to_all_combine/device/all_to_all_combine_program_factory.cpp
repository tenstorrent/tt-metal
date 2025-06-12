// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_combine_device_operation.hpp"
#include "ttnn/operations/ccl/all_to_all_dispatch/device/all_to_all_dispatch_device_operation.hpp"

#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
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

    tt::tt_metal::Program program{};

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& metadata_tensor = tensor_args.metadata_tensor;
    const auto& mapping_tensor = tensor_args.mapping_tensor;
    const auto& output_tensor = tensor_return_value;
    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    auto src_device = mesh_device->get_device(mesh_coordinate);
    const auto src_physical_device_id = mesh_device->get_device(mesh_coordinate)->id();

    const auto& control_plane = tt::tt_fabric::get_control_plane();
    const auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(src_device->id());
    const uint32_t src_mesh_id = *fabric_node_id.mesh_id;
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const auto& input_shape = input_tensor.get_tensor_spec().logical_shape();
    const auto& mapping_shape = mapping_tensor.get_tensor_spec().logical_shape();
    const auto& metadata_shape = metadata_tensor.get_tensor_spec().logical_shape();

    const uint32_t num_devices = mesh_view.num_devices();
    const uint32_t hidden_size = input_shape[-1];
    const uint32_t batch_size = input_shape[0] * num_devices;
    const uint32_t selected_experts_k = metadata_shape[-1];
    const uint32_t experts = mapping_shape[-2];

    // straightforward to lift this assumption
    TT_ASSERT(experts % num_devices == 0, "Currently assuming that experts are evenly split among devices");
    const uint32_t experts_per_device = experts / num_devices;

    auto input_spec = input_tensor.get_tensor_spec();
    auto mapping_spec = mapping_tensor.get_tensor_spec();
    auto output_spec = output_tensor.get_tensor_spec();
    auto metadata_spec = metadata_tensor.get_tensor_spec();

    auto input_page_size_bytes = input_spec.compute_page_size_bytes();
    auto mapping_page_size_bytes = mapping_spec.compute_page_size_bytes();
    auto output_page_size_bytes = output_spec.compute_page_size_bytes();
    auto metadata_page_size_bytes = metadata_spec.compute_page_size_bytes();

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.get_dtype());
    auto metadata_data_format = tt::tt_metal::datatype_to_dataformat_converter(metadata_tensor.get_dtype());

    const bool input_is_dram = input_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool mapping_is_dram = mapping_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool metadata_is_dram = metadata_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;
    const bool output_is_dram = output_tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM;

    constexpr uint32_t buffering_factor = 2;

    // input sharded buffer
    const auto data_cb_id = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_data_config =
        tt::tt_metal::CircularBufferConfig(buffering_factor * input_page_size_bytes, {{data_cb_id, input_data_format}})
            .set_page_size(data_cb_id, input_page_size_bytes);

    // full mapping buffer
    const auto mapping_tensor_cb_id = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_mapping_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            mapping_spec.compute_packed_buffer_size_bytes(), {{mapping_tensor_cb_id, mapping_data_format}})
            .set_page_size(mapping_tensor_cb_id, mapping_page_size_bytes);

    // scratch space to store and share indices of per device experts
    const auto local_experts_cb_id = tt::CBIndex::c_2;
    using local_experts_t = uint32_t;
    const auto local_experts_dataformat = datatype_to_dataformat_converter(convert_to_data_type<local_experts_t>());
    tt::tt_metal::CircularBufferConfig cb_local_experts_config =
        tt::tt_metal::CircularBufferConfig(
            experts_per_device * sizeof(local_experts_t), {{local_experts_cb_id, local_experts_dataformat}})
            .set_page_size(local_experts_cb_id, experts_per_device * sizeof(local_experts_t));

    // metadata page buffer
    const auto metadata_cb_id = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_metadata_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * metadata_page_size_bytes, {{metadata_cb_id, metadata_data_format}})
            .set_page_size(metadata_cb_id, metadata_page_size_bytes);

    // client interface
    const auto client_interface_cb_id = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(client_interface_cb_id, tt::tt_fabric::CLIENT_INTERFACE_SIZE);

    const auto src_core_grid = mesh_device->get_device(mesh_coordinate)->compute_with_storage_grid_size();
    const auto subdevice_cores = grid_to_cores({0, 0}, src_core_grid);

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
    const auto input_cb_handle = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_data_config);
    const auto mapping_cb_handle = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_mapping_tensor_config);
    const auto local_experts_cb_handle =
        tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_local_experts_config);
    const auto metadata_cb_handle = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_metadata_config);
    const auto client_interface_cb =
        tt::tt_metal::CreateCircularBuffer(program, sender_core, client_interface_cb_config);

    const std::vector<uint32_t> reader_compile_time_args = {
        mapping_tensor_cb_id,
        local_experts_cb_id,
        metadata_cb_id,
        data_cb_id,
        experts_per_device,
        batch_size,
        experts,  // same as num_mapping_pages
        src_physical_device_id,
        input_page_size_bytes,
        selected_experts_k,
        mapping_page_size_bytes,
        metadata_page_size_bytes,
        input_is_dram,
        mapping_is_dram,
        metadata_is_dram};

    tt::tt_metal::KernelHandle ternary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/reader_all_to_all_combine.cpp",
        sender_core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // TODO, figure this out from grid dims and mirror axis.
    const auto& axis = operation_attributes.axis;

    const uint32_t batch_mirror_dimension = axis.has_value() ? mesh_device->shape()[axis.value()] : 1;

    // TODO defines
    const std::vector<uint32_t> writer_compile_time_args = {
        metadata_cb_id,
        local_experts_cb_id,
        client_interface_cb_id,
        batch_size,
        experts_per_device,
        num_devices,
        src_chip_id,
        input_page_size_bytes,
        output_is_dram,
        mesh_view.num_cols(),
        batch_mirror_dimension,
    };

    // fabric routing info
    std::vector<uint32_t> dest_mesh_id, dest_chip_id, route;
    for (const auto& coord : all_mesh_coordinates) {
        auto device = mesh_device->get_device(coord);
        auto fabric_node_id = control_plane.get_fabric_node_id_from_physical_chip_id(device->id());
        dest_mesh_id.push_back(*fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)fabric_node_id.chip_id);
    }
    const auto [neighbors, directions] = detail::get_neighbors(mesh_view, mesh_coordinate, topology, axis);

    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", detail::stringify_vector(dest_chip_id)},
        {"DEST_MESH_ID", detail::stringify_vector(dest_mesh_id)},
        {"DIRECTIONS", detail::stringify_array(directions)}};

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_combine/device/kernels/dataflow/writer_all_to_all_combine.cpp",
        sender_core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    std::vector<uint32_t> reader_runtime_args = {
        mapping_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        input_tensor.buffer()->address(),
    };

    std::vector<uint32_t> writer_runtime_args = {
        output_tensor.buffer()->address(),
        operation_attributes.cross_device_semaphore.address(),
    };
    for (auto& neighbor : neighbors) {
        auto neighbor_coordinate = mesh_view.find_device(neighbor->id());
        uint32_t link_id = detail::select_link(mesh_view, mesh_coordinate, neighbor_coordinate, num_links, topology);
        // tt::log_info(
        //     tt::LogAlways,
        //     "Connection between ({}, {}) and ({}, {}) will choose link_id: {}",
        //     mesh_coordinate[0],
        //     mesh_coordinate[1],
        //     neighbor_coordinate[0],
        //     neighbor_coordinate[1],
        //     link_id);
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_physical_device_id, neighbor->id(), link_id, program, sender_core, writer_runtime_args);
    }

    tt::tt_metal::SetRuntimeArgs(program, ternary_reader_kernel_id, sender_cores.at(0), reader_runtime_args);
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, sender_cores.at(0), writer_runtime_args);

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
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        auto& unary_writer_kernel_id = shared_variables.unary_writer_kernel_id;
        auto& core = shared_variables.core;

        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, ternary_reader_kernel_id, core);
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);

        // !TODO probably update tensor addresses

        writer_runtime_args[1] = operation_attributes.cross_device_semaphore.address();
    }
}

}  // namespace ttnn::operations::ccl
