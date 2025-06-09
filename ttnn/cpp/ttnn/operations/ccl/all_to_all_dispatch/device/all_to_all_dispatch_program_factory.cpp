// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_device_operation.hpp"
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

namespace detail {

tt::tt_metal::Shape2D get_physical_size(const ttnn::Tensor& tensor) {
    auto memory_config = tensor.memory_config();
    auto tensor_spec = tensor.tensor_spec();
    auto page_config = tensor_spec.page_config();
    if (tensor.layout() == tt::tt_metal::Layout::TILE) {
        return page_config.get_tile().get_tile_shape();
    } else if (tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        if (memory_config.shard_spec().has_value()) {
            return {1, memory_config.shard_spec().value().shape[1]};
        } else {
            return {1, tensor.padded_shape()[-1]};
        }
    } else {
        TT_FATAL(false, "Invalid layout: neither tile nor row major");
    }
}

uint32_t get_num_pages(const ttnn::Tensor& tensor) {
    const auto& shape = tensor.padded_shape();
    auto physical_size = get_physical_size(tensor);
    auto num_pages = shape.volume() / (physical_size.height() * physical_size.width());
    return num_pages;
}

uint32_t get_page_size(const ttnn::Tensor& tensor) {
    auto memory_config = tensor.memory_config();
    auto tensor_spec = tensor.tensor_spec();
    auto page_config = tensor_spec.page_config();
    auto physical_size = get_physical_size(tensor);
    bool sharded = memory_config.shard_spec().has_value();

    std::optional<tt::tt_metal::Shape2D> physical_shard_size =
        sharded ? std::optional<tt::tt_metal::Shape2D>(memory_config.shard_spec().value().shape) : std::nullopt;
    auto page_shape = page_config.get_page_shape(physical_size, tensor.dtype(), memory_config, physical_shard_size);
    auto page_size = page_config.get_page_size_bytes(page_shape, tensor.dtype());
    return (uint32_t)page_size;
}

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) {
    auto BUFFER_ALIGNMENT = tensor.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                ? tt::tt_metal::hal::get_dram_alignment()
                                : tt::tt_metal::hal::get_l1_alignment();
    return tt::round_up(get_page_size(tensor), BUFFER_ALIGNMENT);
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
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
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
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    tt::tt_metal::Program program{};

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;
    auto output_tensor = tensor_return_value[0];
    auto metadata_tensor = tensor_return_value[1];
    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto mesh_device = input_tensor.mesh_device();
    const auto& mesh_view = mesh_device->get_view();
    auto src_device = mesh_device->get_device(mesh_coordinate);
    auto src_physical_device_id = src_device->id();
    // auto* control_plane = tt::tt_metal::MetalContext::instance().get_cluster().get_control_plane();
    // auto src_fabric_node_id = control_plane->get_fabric_node_id_from_physical_chip_id(src_physical_device_id);

    auto input_shape = input_tensor.get_tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.get_tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.get_tensor_spec().logical_shape();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t hidden_size = input_shape[-1];
    uint32_t batch_size = input_shape[0] * num_devices;
    uint32_t batches_per_device = input_shape[0];
    uint32_t selected_experts_k = indices_shape[-1];
    uint32_t experts = mapping_shape[-2];

    auto input_page_size = detail::get_aligned_page_size(input_tensor);
    auto indices_page_size = detail::get_aligned_page_size(indices_tensor);
    auto mapping_page_size = detail::get_aligned_page_size(mapping_tensor);
    auto output_page_size = detail::get_page_size(output_tensor);
    auto metadata_page_size = detail::get_page_size(metadata_tensor);

    auto input_pages = detail::get_num_pages(input_tensor);
    // tt::log_info(tt::LogAlways, "input shape: {}, input_pages: {}, input_page_size: {}",
    // input_tensor.logical_shape(), input_pages, input_page_size);
    auto indices_pages = detail::get_num_pages(indices_tensor);
    // tt::log_info(tt::LogAlways, "indices shape: {}, indices_pages: {}, indices_page_size: {}",
    // indices_tensor.logical_shape(), indices_pages, indices_page_size);
    auto mapping_pages = detail::get_num_pages(mapping_tensor);
    // tt::log_info(tt::LogAlways, "mapping shape: {}, mapping_pages: {}, mapping_page_size: {}",
    // mapping_tensor.logical_shape(), mapping_pages, mapping_page_size);
    auto output_pages = detail::get_num_pages(output_tensor);
    // tt::log_info(tt::LogAlways, "output shape: {}, output_pages: {}, output_page_size: {}",
    // output_tensor.logical_shape(), output_pages, output_page_size);
    auto metadata_pages = detail::get_num_pages(metadata_tensor);
    // tt::log_info(tt::LogAlways, "metadata shape: {}, metadata_pages: {}, metadata_page_size: {}",
    // metadata_tensor.logical_shape(), metadata_pages, metadata_page_size);

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.get_dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.get_dtype());

    constexpr uint32_t buffering_factor = 2;

    // input sharded buffer
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_1;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_2;
    // client interface
    uint32_t client_interface_cb_id = tt::CBIndex::c_3;
    // metadata buffer
    uint32_t send_preparation_buffer_id = tt::CBIndex::c_4;

    tt::tt_metal::CircularBufferConfig cb_input_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * input_page_size, {{input_tensor_cb_id, input_data_format}})
            .set_page_size(input_tensor_cb_id, input_page_size);

    tt::tt_metal::CircularBufferConfig cb_indices_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            indices_page_size * indices_pages, {{indices_tensor_cb_id, indices_data_format}})
            .set_page_size(indices_tensor_cb_id, indices_page_size);

    tt::tt_metal::CircularBufferConfig cb_mapping_tensor_config =
        tt::tt_metal::CircularBufferConfig(
            mapping_page_size * mapping_pages, {{mapping_tensor_cb_id, mapping_data_format}})
            .set_page_size(mapping_tensor_cb_id, mapping_page_size);

    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(client_interface_cb_id, tt::tt_fabric::CLIENT_INTERFACE_SIZE);

    uint32_t send_preparation_buffer_size = num_devices * batches_per_device * sizeof(uint16_t);

    tt::tt_metal::CircularBufferConfig send_preparation_buffer_config =
        tt::tt_metal::CircularBufferConfig(
            send_preparation_buffer_size, {{send_preparation_buffer_id, tt::DataFormat::UInt16}})
            .set_page_size(send_preparation_buffer_id, send_preparation_buffer_size);

    auto subdevice_core_range_set =
        mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, operation_attributes.subdevice_id);

    auto subdevice_cores = corerange_to_cores(subdevice_core_range_set);
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
    auto input_tensor_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_input_tensor_config);
    auto indices_tensor_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_indices_tensor_config);
    auto mapping_tensor_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, cb_mapping_tensor_config);
    auto client_interface_cb = tt::tt_metal::CreateCircularBuffer(program, sender_core, client_interface_cb_config);
    auto send_preparation_buffer =
        tt::tt_metal::CreateCircularBuffer(program, sender_core, send_preparation_buffer_config);

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_id,
        indices_tensor_cb_id,
        mapping_tensor_cb_id,
        client_interface_cb_id,
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
        input_page_size,
        indices_page_size,
        mapping_page_size,
        metadata_page_size,
        num_links,
        topology == tt::tt_fabric::Topology::Ring ? 1u : 0u,
    };

    tt::tt_metal::KernelHandle ternary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/reader_all_to_all_dispatch.cpp",
        sender_core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_compile_time_args = reader_compile_time_args;

    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ccl/all_to_all_dispatch/device/kernels/dataflow/writer_all_to_all_dispatch.cpp",
        sender_core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        mapping_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        (uint32_t)operation_attributes.cross_device_semaphore->address(),
    };

    std::vector<uint32_t> writer_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        mapping_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        (uint32_t)operation_attributes.cross_device_semaphore->address(),
    };

    tt::tt_metal::SetRuntimeArgs(program, ternary_reader_kernel_id, sender_cores.at(0), reader_runtime_args);
    tt::tt_metal::SetRuntimeArgs(program, binary_writer_kernel_id, sender_cores.at(0), writer_runtime_args);

    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .binary_writer_kernel_id = binary_writer_kernel_id,
         .core = sender_cores.at(0)}};
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
        auto& core = shared_variables.core;

        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, ternary_reader_kernel_id, core);
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, binary_writer_kernel_id, core);

        reader_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
        writer_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
    }
}

}  // namespace ttnn::operations::ccl
