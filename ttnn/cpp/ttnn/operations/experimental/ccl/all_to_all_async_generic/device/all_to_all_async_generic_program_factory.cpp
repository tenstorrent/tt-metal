// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_generic_program_factory.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/global_semaphore.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <unordered_map>

namespace ttnn::experimental::prim {

namespace {
ttnn::Shape get_tiled_shape(const ttnn::Tensor& input_tensor) {
    const auto& tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const auto& shape = input_tensor.padded_shape();
    ttnn::SmallVector<uint32_t> tiled_shape;
    tiled_shape.reserve(shape.rank());
    for (int i = 0; i < shape.rank(); i++) {
        uint32_t dim = 0;
        if (i == shape.rank() - 1) {
            dim = shape[i] / tile_shape[1];
        } else if (i == shape.rank() - 2) {
            dim = shape[i] / tile_shape[0];
        } else {
            dim = shape[i];
        }
        tiled_shape.push_back(dim);
    }
    return ttnn::Shape(tiled_shape);
}
}  // namespace

AllToAllAsyncGenericProgram::cached_mesh_workload_t AllToAllAsyncGenericProgram::create_mesh_workload(
    const AllToAllAsyncGenericParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AllToAllAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();
    auto sub_device_id = operation_attributes.sub_device_id;
    auto subdevice = sub_device_id.has_value() ? *sub_device_id : mesh_device->get_sub_device_ids().at(0);
    const auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, subdevice);
    auto subdevices = {subdevice};

    auto init_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    auto final_barrier_semaphore = ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0);
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, subdevices);

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            init_barrier_semaphore,
            final_barrier_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllToAllAsyncGenericProgram::shared_variables_t>
AllToAllAsyncGenericProgram::create_at(
    const AllToAllAsyncGenericParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllToAllAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value,
    const tt::tt_metal::GlobalSemaphore& init_barrier_semaphore,
    const tt::tt_metal::GlobalSemaphore& final_barrier_semaphore) {
    log_debug(tt::LogOp, "DEBUG: create_at is called");

    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, operation_attributes.cluster_axis);

    const std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor, mesh_coordinate, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor,
        mesh_coordinate,
        -1,
        operation_attributes.topology,
        operation_attributes.cluster_axis);

    TT_FATAL(device_index < operation_attributes.num_devices, "DEBUG: device_index: {}", device_index);

    tt::tt_metal::Program program{};
    MeshDevice* device = tensor_args.input_tensor.device();

    std::vector<Tensor> input_tensors = {tensor_args.input_tensor};
    std::vector<Tensor> output_tensors = {tensor_return_value};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, operation_attributes.topology);

    const size_t num_senders_per_link = 1;
    const auto* topology_type = operation_attributes.topology == ttnn::ccl::Topology::Ring ? "RING" : "LINEAR";

    const auto [sender_worker_core_range, sender_worker_cores] = ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links, num_senders_per_link, device, operation_attributes.sub_device_id);

    // Create CB
    const uint32_t page_size = op_config.get_page_size();
    const uint32_t packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();

    const uint32_t number_pages_per_packet = 2;
    const uint32_t cb_size = (packet_size / page_size) * page_size * number_pages_per_packet;  // round_down
    const tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());

    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(cb_size, {{tt::CB::c_in0, data_format}})
                              .set_page_size(tt::CB::c_in0, page_size);

    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Create CB for fabric
    const auto reserved_packet_header_CB_index = tt::CB::c_in4;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    const uint32_t num_packet_headers_storable = 4;
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    const auto input_shape = get_tiled_shape(tensor_args.input_tensor);
    uint32_t src_out_dims = 1;
    uint32_t src_in_dims = 1;
    for (uint32_t i = 0; i < operation_attributes.out_dim; ++i) {
        src_out_dims *= input_shape[i];
    }

    for (uint32_t i = operation_attributes.out_dim + 1; i < input_shape.size(); ++i) {
        src_in_dims *= input_shape[i];
    }
    const auto output_shape = get_tiled_shape(tensor_return_value);
    uint32_t dst_out_dims = 1;
    uint32_t dst_in_dims = 1;
    for (uint32_t i = 0; i < operation_attributes.in_dim; ++i) {
        dst_out_dims *= output_shape[i];
    }

    const uint32_t reader_has_extra_half_tile =
        operation_attributes.out_dim == input_shape.size() - 2 &&
        tensor_return_value.logical_shape()[operation_attributes.out_dim] % 32 == 16;
    const uint32_t writer_has_extra_half_tile =
        operation_attributes.in_dim == input_shape.size() - 2 &&
        tensor_args.input_tensor.logical_shape()[operation_attributes.in_dim] % 32 == 16;
    for (uint32_t i = operation_attributes.in_dim + 1; i < output_shape.size(); ++i) {
        dst_in_dims *= output_shape[i];
    }

    auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_reader_kernel_config.defines.emplace("TOPOLOGY", topology_type);
    sender_reader_kernel_config.compile_args = {
        tt::CB::c_in0,                              // cb0_id
        page_size,                                  // tensor0_page_size
        device_index,                               // device_index
        operation_attributes.num_devices,           // num_devices
        src_out_dims,                               // outer_dims_size
        input_shape[operation_attributes.out_dim],  // split_dim_size
        src_in_dims,                                // inner_dims_size
        input_shape[input_shape.size() - 1],        // last_dim_sizes
        number_pages_per_packet,                    // number_pages_per_packet
        reader_has_extra_half_tile,                 // has_reader_tail
        writer_has_extra_half_tile,                 // has_writer_tail
    };

    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer())
        .append_to(sender_reader_kernel_config.compile_args);

    auto sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/"
        "all_to_all_sender_reader.cpp",
        sender_worker_core_range,
        sender_reader_kernel_config);

    auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_writer_kernel_config.defines.emplace("TOPOLOGY", topology_type);
    sender_writer_kernel_config.compile_args = {
        tt::CB::c_in0,                              // cb0_id
        device_index,                               // device_index
        operation_attributes.num_devices,           // num_devices
        dst_out_dims,                               // outer_dims_size
        output_shape[operation_attributes.in_dim],  // concat_dim_size
        dst_in_dims,                                // inner_dims_size
        number_pages_per_packet,                    // number_pages_per_packet
        writer_has_extra_half_tile,                 // has_writer_tail
        page_size,                                  // intermediate_page_size
        reserved_packet_header_CB_index,
    };

    tt::tt_metal::TensorAccessorArgs(tensor_return_value.buffer()).append_to(sender_writer_kernel_config.compile_args);

    auto sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async_generic/device/kernels/"
        "all_to_all_sender_writer.cpp",
        sender_worker_core_range,
        sender_writer_kernel_config);

    std::vector<uint32_t> sender_reader_rt_args = {tensor_args.input_tensor.buffer()->address()};
    tt::tt_metal::SetRuntimeArgs(program, sender_reader_kernel_id, sender_worker_core_range, sender_reader_rt_args);

    auto drain_sync_core = device->worker_core_from_logical_core({sender_worker_cores[0]});
    std::vector<uint32_t> sender_writer_rt_args = {
        tensor_return_value.buffer()->address(),
        init_barrier_semaphore.address(),
        final_barrier_semaphore.address(),
        drain_sync_core.x,
        drain_sync_core.y};
    sender_writer_rt_args.push_back(forward_coord.has_value());

    if (forward_coord.has_value()) {
        const auto sender_device_fabric_node_id = device->get_fabric_node_id(mesh_coordinate);
        const auto forward_device_fabric_node_id = device->get_fabric_node_id(forward_coord.value());
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_device_fabric_node_id,
            forward_device_fabric_node_id,
            0,
            program,
            {sender_worker_cores[0]},
            sender_writer_rt_args);
    }

    sender_writer_rt_args.push_back(backward_coord.has_value());
    if (backward_coord.has_value()) {
        const auto sender_device_fabric_node_id = device->get_fabric_node_id(mesh_coordinate);
        const auto backward_device_fabric_node_id = device->get_fabric_node_id(backward_coord.value());
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_device_fabric_node_id,
            backward_device_fabric_node_id,
            0,
            program,
            {sender_worker_cores[0]},
            sender_writer_rt_args);
    }
    tt::tt_metal::SetRuntimeArgs(program, sender_writer_kernel_id, sender_worker_core_range, sender_writer_rt_args);

    return {
        std::move(program),
        {.sender_reader_kernel_id = sender_reader_kernel_id,
         .sender_writer_kernel_id = sender_writer_kernel_id,
         .sender_worker_cores = sender_worker_cores,
         .init_barrier_semaphore = init_barrier_semaphore,
         .final_barrier_semaphore = final_barrier_semaphore}};
}

void AllToAllAsyncGenericProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AllToAllAsyncGenericParams& /*operation_attributes*/,
    const AllToAllAsyncGenericInputs& tensor_args,
    Tensor& tensor_return_value) {
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        const auto& coord = coordinate_range.start_coord();
        TT_FATAL(
            coord == coordinate_range.end_coord(),
            "Expected single coordinate per program but got range of {} to {}",
            coord,
            coordinate_range.end_coord());
        auto& shared_variables = cached_workload.shared_variables.at(coordinate_range);

        auto& sender_reader_runtime_args = GetRuntimeArgs(program, shared_variables.sender_reader_kernel_id);
        auto& sender_writer_runtime_args = GetRuntimeArgs(program, shared_variables.sender_writer_kernel_id);
        for (const auto& core : shared_variables.sender_worker_cores) {
            auto& worker_sender_reader_runtime_args = sender_reader_runtime_args[core.x][core.y];
            auto& worker_sender_writer_runtime_args = sender_writer_runtime_args[core.x][core.y];
            worker_sender_reader_runtime_args[0] = tensor_args.input_tensor.buffer()->address();
            worker_sender_writer_runtime_args[0] = tensor_return_value.buffer()->address();
            worker_sender_writer_runtime_args[1] = shared_variables.init_barrier_semaphore.address();
            worker_sender_writer_runtime_args[2] = shared_variables.final_barrier_semaphore.address();
        }
    }
}

}  // namespace ttnn::experimental::prim
