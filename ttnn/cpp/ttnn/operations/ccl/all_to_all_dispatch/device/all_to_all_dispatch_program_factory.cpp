// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/erisc_datamover_builder.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/fabric.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include <limits>

namespace ttnn::operations::ccl {

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

std::pair<std::array<uint32_t, 6>, std::array<uint32_t, 6>> get_cb_sizes(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& mapping_tensor,
    uint32_t num_links,
    std::optional<uint32_t> axis) {
    auto aligned_input_page_size = get_aligned_page_size(input_tensor);
    auto aligned_indices_page_size = get_aligned_page_size(indices_tensor);
    auto aligned_mapping_page_size = get_aligned_page_size(mapping_tensor);
    uint32_t tokens_per_device = get_num_rows(input_tensor);
    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);

    auto mapping_pages = get_num_pages(mapping_tensor);

    auto mesh_view = input_tensor.mesh_device()->get_view();
    uint32_t num_devices = mesh_view.num_devices();

    uint32_t dispatch_devices =
        axis.has_value() ? (axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()) : num_devices;

    constexpr uint32_t buffering_factor = 2;
    constexpr uint32_t num_packet_headers = 2;

    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    std::array<uint32_t, 6> cb_sizes = {
        buffering_factor * aligned_input_page_size,
        tokens_per_core * aligned_indices_page_size,
        mapping_pages * aligned_mapping_page_size,
        num_devices * tokens_per_core * sizeof(uint8_t),
        tokens_per_device * dispatch_devices * aligned_indices_page_size,
        num_packet_headers * packet_header_size_bytes,
    };

    std::array<uint32_t, 6> cb_page_sizes = {
        aligned_input_page_size,
        aligned_indices_page_size,
        aligned_mapping_page_size,
        tokens_per_core * sizeof(uint8_t),
        aligned_indices_page_size,
        packet_header_size_bytes,
    };

    return {cb_sizes, cb_page_sizes};
}

}  // namespace detail

AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::cached_mesh_workload_t
AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value, tensor_coords);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::shared_variables_t>
AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::Program program{};

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;
    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);
    auto num_links = operation_attributes.num_links;
    auto topology = tt::tt_fabric::get_fabric_topology();

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    auto src_device = mesh_device->get_device(mesh_coordinate);
    auto src_physical_device_id = src_device->id();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = common::get_linearized_index(mesh_coordinate, mesh_view);

    log_debug(
        tt::LogOp,
        "\nCreating all to all dispatch program for mesh coordinate: ({}, {}) with physical device id: {} mesh id: {} "
        "chip id: {} linearized mesh coord: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_device->id(),
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord);

    const auto [neighbors, directions] =
        common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t dispatch_devices =
        operation_attributes.axis.has_value()
            ? operation_attributes.axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()
            : mesh_view.num_devices();

    uint32_t hidden_size = input_shape[-1];
    uint32_t batch_size = input_shape[0] * dispatch_devices;

    uint32_t tokens_per_device = detail::get_num_rows(input_tensor);
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t experts = mapping_shape[-2];

    auto input_page_size = detail::get_page_size(input_tensor);
    auto indices_page_size = detail::get_page_size(indices_tensor);
    auto mapping_page_size = detail::get_page_size(mapping_tensor);
    auto output_page_size = detail::get_page_size(output_tensor);
    auto metadata_page_size = detail::get_page_size(metadata_tensor);

    auto input_pages = detail::get_num_pages(input_tensor);
    auto indices_pages = detail::get_num_pages(indices_tensor);
    auto mapping_pages = detail::get_num_pages(mapping_tensor);
    auto output_pages = detail::get_num_pages(output_tensor);
    auto metadata_pages = detail::get_num_pages(metadata_tensor);

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());

    // input sharded buffer
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_1;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_2;
    // client interface
    uint32_t packet_header_cb_id = tt::CBIndex::c_3;
    // book-keeping buffer to avoid sending the same token multiple times
    uint32_t send_preparation_buffer_id = tt::CBIndex::c_4;
    // intermediate buffer for holding metadata before writing out to the device (for FullPacket impl)
    uint32_t metadata_buffer_id = tt::CBIndex::c_5;

    uint32_t aligned_input_page_size = detail::get_aligned_page_size(input_tensor);
    log_debug(
        tt::LogOp,
        "input shape: {}, input_pages: {}, input_page_size: {}, aligned_input_page_size: {}",
        input_tensor.logical_shape(),
        input_pages,
        input_page_size,
        aligned_input_page_size);

    uint32_t aligned_indices_page_size = detail::get_aligned_page_size(indices_tensor);
    log_debug(
        tt::LogOp,
        "indices shape: {}, indices_pages: {}, indices_page_size: {}, aligned_indices_page_size: {}",
        indices_tensor.logical_shape(),
        indices_pages,
        indices_page_size,
        aligned_indices_page_size);

    uint32_t aligned_mapping_page_size = detail::get_aligned_page_size(mapping_tensor);
    log_debug(
        tt::LogOp,
        "mapping shape: {}, mapping_pages: {}, mapping_page_size: {}, aligned_mapping_page_size: {}",
        mapping_tensor.logical_shape(),
        mapping_pages,
        mapping_page_size,
        aligned_mapping_page_size);

    uint32_t aligned_output_page_size = detail::get_aligned_page_size(output_tensor);
    log_debug(
        tt::LogOp,
        "output shape: {}, output_pages: {}, output_page_size: {}, aligned_output_page_size: {}",
        output_tensor.logical_shape(),
        output_pages,
        output_page_size,
        aligned_output_page_size);

    uint32_t aligned_metadata_page_size = detail::get_aligned_page_size(metadata_tensor);
    log_debug(
        tt::LogOp,
        "metadata shape: {}, metadata_pages: {}, metadata_page_size: {}, aligned_metadata_page_size: {}",
        metadata_tensor.logical_shape(),
        metadata_pages,
        metadata_page_size,
        aligned_metadata_page_size);

    auto [cb_sizes, cb_page_sizes] =
        detail::get_cb_sizes(input_tensor, indices_tensor, mapping_tensor, num_links, operation_attributes.axis);

    tt::tt_metal::CircularBufferConfig cb_input_tensor_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[0], {{input_tensor_cb_id, input_data_format}})
            .set_page_size(input_tensor_cb_id, cb_page_sizes[0]);

    tt::tt_metal::CircularBufferConfig cb_indices_tensor_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[1], {{indices_tensor_cb_id, indices_data_format}})
            .set_page_size(indices_tensor_cb_id, cb_page_sizes[1]);

    tt::tt_metal::CircularBufferConfig cb_mapping_tensor_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[2], {{mapping_tensor_cb_id, mapping_data_format}})
            .set_page_size(mapping_tensor_cb_id, cb_page_sizes[2]);

    tt::tt_metal::CircularBufferConfig cb_send_preparation_buffer_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[3], {{send_preparation_buffer_id, tt::DataFormat::UInt8}})
            .set_page_size(send_preparation_buffer_id, cb_page_sizes[3]);

    tt::tt_metal::CircularBufferConfig cb_metadata_buffer_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[4], {{metadata_buffer_id, mapping_data_format}})
            .set_page_size(metadata_buffer_id, cb_page_sizes[4]);

    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[5], {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, cb_page_sizes[5]);

    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);
    uint32_t num_cores = std::min(num_links, tt::div_up(tokens_per_device, tokens_per_core));
    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    // create circular buffers
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_input_tensor_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_indices_tensor_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_mapping_tensor_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_send_preparation_buffer_config);
    if (operation_attributes.impl == AllToAllTransferType::FullPacket) {
        tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_metadata_buffer_config);
    }

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    log_debug(tt::LogOp, "dest_chip_id: {}", common::stringify(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", common::stringify(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", common::stringify(directions));

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor.buffer()->is_dram(),
        indices_tensor.buffer()->is_dram(),
        mapping_tensor.buffer()->is_dram(),
        output_tensor.buffer()->is_dram(),
        metadata_tensor.buffer()->is_dram(),

        input_tensor_cb_id,
        indices_tensor_cb_id,
        mapping_tensor_cb_id,
        packet_header_cb_id,
        send_preparation_buffer_id,

        input_pages,
        indices_pages,
        mapping_pages,
        output_pages,
        metadata_pages,

        input_page_size,
        indices_page_size,
        mapping_page_size,
        output_page_size,
        metadata_page_size,

        num_devices,
        hidden_size,
        batch_size,
        selected_experts_k,
        experts,
        tokens_per_device,

        num_links,
        (uint32_t)topology,

        src_mesh_id,
        (uint32_t)src_chip_id,
        mesh_view.num_rows(),
        mesh_view.num_cols(),

        aligned_input_page_size,
        aligned_indices_page_size,
        aligned_mapping_page_size,
        aligned_output_page_size,
        aligned_metadata_page_size,

        (uint32_t)fabric_max_packet_size,

        l1_alignment,
        metadata_buffer_id,
        operation_attributes.impl == AllToAllTransferType::PageByPage ? 1 : 0,
        linearized_mesh_coord,
    };

    const auto& writer_compile_time_args = reader_compile_time_args;

    std::map<std::string, std::string> reader_defines = {
        {"AXIS", std::to_string(operation_attributes.axis.has_value() ? operation_attributes.axis.value() : -1)},
    };

    tt::tt_metal::KernelHandle ternary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/reader_all_to_all_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    // Code-gen a mesh-position to fabric chip ID array for the writer kernel
    // Code-gen a mesh-position to mesh-id array for the writer kernel
    // Code-gen a direction array that is set to true when a direction has a valid connection (when a neighbor exists or
    // if it's along a valid cluster axis)
    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", common::stringify(dest_mesh_id)},
        {"DIRECTIONS", common::stringify(directions)}};

    if (operation_attributes.axis.has_value()) {
        writer_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/writer_all_to_all_dispatch.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = writer_compile_time_args,
            .defines = writer_defines});

    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        mapping_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        (uint32_t)operation_attributes.cross_device_semaphore->address(),
        0,
        0,
    };

    uint32_t link_id = 0;
    uint32_t tokens_per_core_start = 0;
    for (uint32_t i = 0; i < sender_cores.size(); i++) {
        std::vector<uint32_t> writer_runtime_args = {
            input_tensor.buffer()->address(),
            indices_tensor.buffer()->address(),
            mapping_tensor.buffer()->address(),
            output_tensor.buffer()->address(),
            metadata_tensor.buffer()->address(),
            (uint32_t)operation_attributes.cross_device_semaphore->address(),
            0,
            0,
        };
        reader_runtime_args[6] = tokens_per_core_start;
        reader_runtime_args[7] = std::min(tokens_per_core_start + tokens_per_core, tokens_per_device);
        writer_runtime_args[6] = tokens_per_core_start;
        writer_runtime_args[7] = reader_runtime_args[7];
        tokens_per_core_start = reader_runtime_args[7];
        for (auto& neighbor_coordinate : neighbors) {
            log_debug(
                tt::LogOp,
                "Connection between mesh coord ({}, {}) and ({}, {}) at core {} will choose link_id: {} and handles "
                "token indices from {} to {}",
                mesh_coordinate[0],
                mesh_coordinate[1],
                neighbor_coordinate[0],
                neighbor_coordinate[1],
                sender_cores.at(i),
                link_id,
                reader_runtime_args[6],
                reader_runtime_args[7]);
            tt::tt_fabric::append_fabric_connection_rt_args(
                src_fabric_node_id,
                mesh_device->get_fabric_node_id(neighbor_coordinate),
                link_id,
                program,
                sender_cores.at(i),
                writer_runtime_args);
        }

        tt::tt_metal::SetRuntimeArgs(program, ternary_reader_kernel_id, sender_cores.at(i), reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, binary_writer_kernel_id, sender_cores.at(i), writer_runtime_args);
        link_id++;
    }

    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .binary_writer_kernel_id = binary_writer_kernel_id,
         .cores = sender_cores}};
}

void AllToAllDispatchDeviceOperation::AllToAllDispatchSparse::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        auto& binary_writer_kernel_id = shared_variables.binary_writer_kernel_id;
        auto& cores = shared_variables.cores;

        const auto& output_tensor = tensor_return_value.at(0);
        const auto& metadata_tensor = tensor_return_value.at(1);

        for (auto& core : cores) {
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, ternary_reader_kernel_id, core);
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, binary_writer_kernel_id, core);
            reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.expert_mapping_tensor.buffer()->address();
            reader_runtime_args.at(3) = output_tensor.buffer()->address();
            reader_runtime_args.at(4) = metadata_tensor.buffer()->address();
            reader_runtime_args.at(5) = (uint32_t)operation_attributes.cross_device_semaphore->address();

            writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.expert_mapping_tensor.buffer()->address();
            writer_runtime_args.at(3) = output_tensor.buffer()->address();
            writer_runtime_args.at(4) = metadata_tensor.buffer()->address();
            writer_runtime_args.at(5) = (uint32_t)operation_attributes.cross_device_semaphore->address();
        }
    }
}

}  // namespace ttnn::operations::ccl
