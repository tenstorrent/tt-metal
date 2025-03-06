// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/hal_exp.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>

namespace ttnn::operations::experimental::ccl {

LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::cached_program_t
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    uint32_t num_devices = 2;

    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::IDevice* device = input_tensor.device();

    auto control_plane = tt::DevicePool::instance().get_control_plane();

    const auto& input_shape = input_tensor.get_logical_shape();
    const auto dim = operation_attributes.dim;
    uint32_t rank = input_shape.size();
    auto& output_tensor = tensor_return_value;
    auto& output_shape = output_tensor.get_logical_shape();
    auto& padded_output_shape = output_tensor.get_padded_shape();
    const auto& tile_shape = input_tensor.get_tensor_spec().tile().get_tile_shape();
    const auto& face_shape = input_tensor.get_tensor_spec().tile().get_face_shape();
    TT_FATAL(input_tensor.shard_spec().has_value(), "Shard spec is not present");
    auto shard_spec = input_tensor.shard_spec().value();
    auto output_shard_spec = output_tensor.shard_spec().value();

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    uint32_t tiles_per_core_width = shard_width / tile_shape[1];

    uint32_t shard_height_output = output_shard_spec.shape[0];
    uint32_t shard_width_output = output_shard_spec.shape[1];
    uint32_t tiles_per_core_width_output = shard_width_output / tile_shape[1];

    tt::tt_metal::Program program{};
    uint32_t element_size = input_tensor.element_size();

    uint32_t src_cb_index = tt::CBIndex::c_0;
    uint32_t dst_cb_index = tt::CBIndex::c_1;
    uint32_t client_interface_cb_index = tt::CBIndex::c_2;
    uint32_t fabric_sender_cb_index = tt::CBIndex::c_3;
    uint32_t fabric_receiver_cb_index = tt::CBIndex::c_4;

    uint32_t output_cb_index = src_cb_index;

    uint32_t num_input_pages_to_read = tiles_per_core_width;
    uint32_t num_output_pages_to_write = tiles_per_core_width_output;

    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t input_page_size = tile_size(cb_data_format);
    uint32_t output_page_size = tile_size(cb_data_format);

    tt::tt_metal::CircularBufferConfig cb_src_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages_to_read * input_page_size, {{src_cb_index, cb_data_format}})
            .set_page_size(src_cb_index, input_page_size)
            .set_globally_allocated_address(*src_buffer);

    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(num_input_pages_to_read * input_page_size, {{dst_cb_index, cb_data_format}})
            .set_page_size(dst_cb_index, input_page_size)
            .set_globally_allocated_address(*dst_buffer);

    // Allocate space for the client interface
    tt::tt_metal::CircularBufferConfig client_interface_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tt::tt_fabric::CLIENT_INTERFACE_SIZE, {{client_interface_cb_index, DataFormat::UInt32}})
            .set_page_size(client_interface_cb_index, tt::tt_fabric::CLIENT_INTERFACE_SIZE);

    // Add buffer for the packet, which is the whole shard (lol)
    uint32_t packet_size_bytes = (tiles_per_core_width * input_page_size) + PACKET_HEADER_SIZE_BYTES;
    tt::tt_metal::CircularBufferConfig fabric_sender_cb_config =
        tt::tt_metal::CircularBufferConfig(packet_size_bytes, {{fabric_sender_cb_index, cb_data_format}})
            .set_page_size(fabric_sender_cb_index, packet_size_bytes);

    // Add buffer for reduced shards from the other device(s)
    tt::tt_metal::CircularBufferConfig fabric_receiver_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_pages_to_read * input_page_size, {{fabric_receiver_cb_index, cb_data_format}})
            .set_page_size(fabric_receiver_cb_index, num_input_pages_to_read * input_page_size);

    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src_config);  // input buffer
    auto cb_dst = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);  // output buffer
    auto cb_client_interface =
        tt::tt_metal::CreateCircularBuffer(program, all_cores, client_interface_cb_config);  // client interface
    // fabric receiver - we put the reduced shards here after we receive them from the other device, then we reduce with
    // the input buffer shard and pack into the fabric sender buffer
    auto cb_fabric_receiver = tt::tt_metal::CreateCircularBuffer(program, all_cores, fabric_receiver_cb_config);
    // fabric sender - we put the payload and packet header here after we apply the reduction, and then send it to the
    // next device, or if it's the last device, we write it to the output buffer
    auto cb_fabric_sender = tt::tt_metal::CreateCircularBuffer(program, all_cores, fabric_sender_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {
        input_page_size,
        src_cb_index,
        dst_cb_index,
        0,
        tiles_per_core_width,
        fabric_sender_cb_index,
        fabric_receiver_cb_index,
        client_interface_cb_index};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
        "reader_llama_reduce_scatter.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {};

    std::vector<uint32_t> writer_compile_time_args = {
        input_page_size,
        src_cb_index,
        fabric_sender_cb_index,
        fabric_receiver_cb_index,
        client_interface_cb_index,
        0,
        tiles_per_core_width,
        PACKET_HEADER_SIZE_BYTES};

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
        "writer_llama_reduce_scatter.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    const auto [mesh_id, chip_id] = control_plane->get_mesh_chip_id_from_physical_chip_id(device->id());

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    uint32_t start_block = 0;
    uint32_t num_blocks_per_core = tiles_per_core_width;
    uint32_t num_cores = cores.size();
    uint32_t split_size = num_cores / num_devices;
    for (uint32_t i = 0; i < num_cores; i++) {
        auto core = cores[i];
        std::vector<uint32_t> reader_runtime_args = {src_buffer->address(), dst_buffer->address()};
        std::vector<uint32_t> writer_runtime_args = {dst_buffer->address()};
        uint32_t target_chip_id = i / split_size;  // 0 or 1 for 2 devices
        if (target_chip_id == chip_id) {
            // this is the target destination device
            bool write_output = true;
            writer_runtime_args.push_back((uint32_t)write_output);
            // add sharded addr gen stuff here or in compile time args
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
            continue;
        }
        // this is a source device
        bool write_output = false;
        writer_runtime_args.push_back((uint32_t)write_output);
        // get the router to the target device
        auto dir = target_chip_id > chip_id ? RoutingDirection::E : RoutingDirection::W;
        auto routers = control_plane->get_routers_to_chip(mesh_id, chip_id, mesh_id, target_chip_id);
        auto router_noc_xy = tt_metal::hal.noc_xy_encoding(routers[0].second.x, routers[0].second.y);
        auto my_noc_xy = tt_metal::hal.noc_xy_encoding(core.x, core.y);
        auto dst_noc_xy = tt_metal::hal.noc_xy_encoding(core.x, core.y);  // chip 0 core (x, y) -> core (x, y)
        // on chip 0, chip 1 writes to 160 tiles from the top for LLaMA 7B
        uint32_t offset_to_receiver = chip_id * (tiles_per_core_width * input_page_size);
        std::cout << "chip_id: " << chip_id << " target_chip_id: " << target_chip_id
                  << " offset_to_receiver: " << offset_to_receiver << " router_noc_xy: " << router_noc_xy << std::endl;

        writer_runtime_args.push_back(router_noc_xy);
        writer_runtime_args.push_back(offset_to_receiver);
        writer_runtime_args.push_back(my_noc_xy);
        writer_runtime_args.push_back(mesh_id);
        writer_runtime_args.push_back(target_chip_id);
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .cb_ids = {src_cb_index, dst_cb_index},
         .core_range = all_cores}};
}

void LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();
    auto& all_cores = cached_program.shared_variables.core_range;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_ids[0], *src_buffer);
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_ids[1], *dst_buffer);
    for (const auto& core : cores) {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, core);
        runtime_args[0] = src_buffer->address();
        auto& runtime_args_writer = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        runtime_args_writer[0] = dst_buffer->address();
    }
}

}  // namespace ttnn::operations::experimental::ccl
