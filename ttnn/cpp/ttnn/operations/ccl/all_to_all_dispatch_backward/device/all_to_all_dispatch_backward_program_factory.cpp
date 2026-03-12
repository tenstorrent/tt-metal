// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "cpp/ttnn/operations/ccl/all_to_all_dispatch_backward/device/all_to_all_dispatch_backward_device_operation.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::operations::ccl {

AllToAllDispatchBackwardDeviceOperation::AllToAllDispatchBackwardToDense::cached_mesh_workload_t
AllToAllDispatchBackwardDeviceOperation::AllToAllDispatchBackwardToDense::create_mesh_workload(
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
    AllToAllDispatchBackwardDeviceOperation::AllToAllDispatchBackwardToDense::shared_variables_t>
AllToAllDispatchBackwardDeviceOperation::AllToAllDispatchBackwardToDense::create_at(
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
    const auto& output_tensor = tensor_return_value;
    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;
    const auto& axis = operation_attributes.axis;
    const uint32_t output_shard_dim = operation_attributes.output_shard_dim;

    const auto grad_dtype = grad_output.dtype();

    auto* mesh_device = grad_output.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const auto& grad_shape = grad_output.tensor_spec().logical_shape();

    const uint32_t num_devices = mesh_view.num_devices();

    // Compute dispatch group dimensions
    const uint32_t dispatch_devices =
        axis.has_value() ? (uint32_t)mesh_device->shape()[axis.value()] : num_devices;

    uint32_t batch_per_device, seq_per_device;
    if (output_shard_dim == 1) {
        batch_per_device = grad_shape[1] / dispatch_devices;
        seq_per_device = grad_shape[2];
    } else {
        batch_per_device = grad_shape[1];
        seq_per_device = grad_shape[2] / dispatch_devices;
    }

    const uint32_t grad_dim2 = grad_shape[2];
    const uint32_t total_pages = grad_shape[1] * grad_shape[2];

    const auto& grad_spec = grad_output.tensor_spec();

    const bool grad_is_dram = grad_output.buffer()->buffer_type() == BufferType::DRAM;

    const auto grad_page_size_bytes = grad_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto dram_alignment = hal::get_dram_alignment();

    const auto aligned_grad_page_size_bytes =
        tt::align(grad_page_size_bytes, grad_is_dram ? dram_alignment : l1_alignment);

    const auto grad_data_format = datatype_to_dataformat_converter(grad_output.dtype());

    // CB c_0: grad data page (1 page buffer)
    constexpr auto data_cb_id = tt::CBIndex::c_0;
    CircularBufferConfig cb_data_config =
        CircularBufferConfig(aligned_grad_page_size_bytes, {{data_cb_id, grad_data_format}})
            .set_page_size(data_cb_id, aligned_grad_page_size_bytes);

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

    // Work-split: grad output pages across cores
    uint32_t pages_per_core = tt::div_up(total_pages, num_links);
    uint32_t num_cores = std::min(num_links, tt::div_up(total_pages, pages_per_core));
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, operation_attributes.worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    // Create circular buffers
    CreateCircularBuffer(program, sender_core_grid, cb_data_config);
    CreateCircularBuffer(program, sender_core_grid, client_interface_cb_config);

    const uint32_t flat_mesh_idx = common::get_linearized_index(mesh_coordinate, mesh_view);

    // device_in_group: this device's position along the dispatch axis
    uint32_t device_in_group;
    if (!axis.has_value()) {
        device_in_group = flat_mesh_idx;
    } else if (axis.value() == 0) {
        // axis=0 → dispatch along rows; device_in_group = row index
        device_in_group = flat_mesh_idx / mesh_view.num_cols();
    } else {
        // axis=1 → dispatch along cols; device_in_group = col index
        device_in_group = flat_mesh_idx % mesh_view.num_cols();
    }

    // Reader compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        data_cb_id,
        grad_page_size_bytes,
    };
    TensorAccessorArgs(grad_output.buffer()).append_to(reader_compile_time_args);

    const DataMovementConfig reader_config{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
        .compile_args = reader_compile_time_args};

    KernelHandle ternary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch_backward/device/kernels/dataflow/"
        "reader_all_to_all_dispatch_backward.cpp",
        sender_core_grid,
        reader_config);

    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes =
        grad_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes)
                                         : fabric_max_packet_size_bytes;

    // Writer compile-time args
    std::vector<uint32_t> writer_compile_time_args = {
        client_interface_cb_id,
        data_cb_id,
        batch_per_device,
        seq_per_device,
        dispatch_devices,
        grad_dim2,
        device_in_group,
        output_shard_dim,
        num_devices,
        src_chip_id,
        grad_page_size_bytes,
        l1_alignment,
        mesh_view.num_rows(),
        mesh_view.num_cols(),
        max_packet_size_bytes,
        flat_mesh_idx,
        (uint32_t)topology,
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
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch_backward/device/kernels/dataflow/"
        "writer_all_to_all_dispatch_backward.cpp",
        sender_core_grid,
        writer_config);

    // Runtime args
    std::vector<uint32_t> reader_runtime_args = {
        grad_output.buffer()->address(),
        0,  // page_start (updated per core)
        0,  // page_end   (updated per core)
    };

    uint32_t link_id = 0;
    uint32_t pages_done = 0;
    for (const auto& sender_core : sender_cores) {
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),
            (uint32_t)cross_device_semaphore.address(),
            (uint32_t)init_semaphore.address(),
            0,  // page_start (updated per core)
            0,  // page_end   (updated per core)
        };
        const uint32_t core_page_end =
            std::min(pages_done + pages_per_core, total_pages);
        reader_runtime_args[1] = pages_done;
        reader_runtime_args[2] = core_page_end;
        writer_runtime_args[3] = pages_done;
        writer_runtime_args[4] = core_page_end;
        pages_done = core_page_end;

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

void AllToAllDispatchBackwardDeviceOperation::AllToAllDispatchBackwardToDense::override_runtime_arguments(
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

            reader_runtime_args.at(0) = tensor_args.grad_output.buffer()->address();

            writer_runtime_args.at(0) = tensor_return_value.buffer()->address();
            writer_runtime_args.at(1) = (uint32_t)shared_variables.cross_device_semaphore.address();
            writer_runtime_args.at(2) = (uint32_t)shared_variables.init_semaphore.address();
        }
    }
}

}  // namespace ttnn::operations::ccl
