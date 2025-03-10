// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "llama_reduce_scatter_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include <tt-metalium/hal_exp.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device_pool.hpp>
#include "ttnn/operations/ccl/erisc_datamover_builder.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::operations::experimental::ccl {

namespace detail {
std::string device_order_array_string(uint32_t ring_size, uint32_t ring_index) {
    ttnn::SmallVector<uint32_t> device_order;
    device_order.reserve(ring_size - 1);
    // Add all indices except ring_index
    for (uint32_t i = 0; i < ring_size; i++) {
        if (i != ring_index) {
            device_order.push_back(i);
        }
    }

    // Sort based on absolute difference from ring_index in descending order
    std::sort(device_order.begin(), device_order.end(), [ring_index](uint32_t a, uint32_t b) {
        return std::abs(static_cast<int>(a) - static_cast<int>(ring_index)) >
               std::abs(static_cast<int>(b) - static_cast<int>(ring_index));
    });

    // Convert to string format
    std::string result = "{";
    for (size_t i = 0; i < device_order.size(); i++) {
        result += std::to_string(device_order[i]);
        if (i < device_order.size() - 1) {
            result += ", ";
        }
    }
    result += "}";
    return result;
}

void append_fabric_connection_rt_args(
    const std::optional<ttnn::ccl::SenderWorkerAdapterSpec>& connection,
    const CoreCoord& core,
    tt::tt_metal::Program& program,
    std::vector<uint32_t>& writer_rt_args) {
    writer_rt_args.push_back(connection.has_value());
    if (connection.has_value()) {
        auto sender_worker_flow_control_semaphore_id = CreateSemaphore(program, {core}, 0);
        auto sender_worker_teardown_semaphore_id = CreateSemaphore(program, {core}, 0);
        auto sender_worker_buffer_index_semaphore_id = CreateSemaphore(program, {core}, 0);
        append_worker_to_fabric_edm_sender_rt_args(
            connection.value(),
            sender_worker_flow_control_semaphore_id,
            sender_worker_teardown_semaphore_id,
            sender_worker_buffer_index_semaphore_id,
            writer_rt_args);
    }
}
}  // namespace detail

LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::cached_program_t
LlamaReduceScatterDeviceOperation::LlamaReduceScatterAdd::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;
    uint32_t ring_size = operation_attributes.ring_devices;
    uint32_t num_devices = ring_size;

    const auto& input_tensor = tensor_args.input_tensor;
    tt::tt_metal::IDevice* device = input_tensor.device();
    bool enable_persistent_fabric = true;
    uint32_t num_links = 1;

    uint32_t ring_index = operation_attributes.ring_index;
    std::string device_order = detail::device_order_array_string(ring_size, ring_index);

    std::cout << "ring_index: " << operation_attributes.ring_index << " device_order: " << device_order << std::endl;

    std::map<std::string, std::string> reader_defines = {{"DEVICE_ORDER", device_order}};

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
    const auto& cross_device_semaphore = operation_attributes.cross_device_semaphore;
    // All of them should have the same address, noticed that the address value is uint64_t though, which we can't pass
    // into NOC
    TT_FATAL(cross_device_semaphore.has_value(), "Cross device semaphore is not present");
    uint32_t semaphore_address = cross_device_semaphore->address();

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    uint32_t tiles_per_core_width = shard_width / tile_shape[1];

    uint32_t shard_height_output = output_shard_spec.shape[0];
    uint32_t shard_width_output = output_shard_spec.shape[1];
    uint32_t tiles_per_core_width_output = shard_width_output / tile_shape[1];

    tt::tt_metal::Program program{};

    std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
        ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
            device,
            operation_attributes.forward_device.value_or(nullptr),
            operation_attributes.backward_device.value_or(nullptr),
            &program,
            enable_persistent_fabric,
            num_links);
    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};

    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, ttnn::ccl::Topology::Linear);
    LineTopology line_topology(ring_size, ring_index);
    const size_t num_targets_forward =
        line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::FORWARD);
    const size_t num_targets_backward =
        line_topology.get_distance_to_end_of_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] = choose_worker_cores(
        num_links, num_workers_per_link, enable_persistent_fabric, device, operation_attributes.subdevice_id);
    auto sender_core = sender_worker_cores.at(0);

    uint32_t element_size = input_tensor.element_size();

    // input sharded buffer
    uint32_t src_cb_index = tt::CBIndex::c_0;
    // output sharded buffer
    uint32_t dst_cb_index = tt::CBIndex::c_1;
    // client interface
    uint32_t packet_header_cb_index = tt::CBIndex::c_2;
    // fabric sender
    uint32_t fabric_sender_cb_index = tt::CBIndex::c_3;
    uint32_t fabric_receiver_cb_index = tt::CBIndex::c_4;

    uint32_t num_input_pages_to_read = tiles_per_core_width;
    uint32_t num_output_pages_per_core = tiles_per_core_width_output;

    auto all_cores = shard_spec.grid;
    uint32_t ncores = shard_spec.num_cores();
    uint32_t cores_per_device = ncores / num_devices;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t input_page_size = tile_size(
        cb_data_format);  // doesn't work for tiny tiles, there is likely some API somewhere but I don't know where
    uint32_t output_page_size = tile_size(cb_data_format);

    tt::tt_metal::CircularBufferConfig cb_src_config =
        tt::tt_metal::CircularBufferConfig(tiles_per_core_width * input_page_size, {{src_cb_index, cb_data_format}})
            .set_page_size(src_cb_index, input_page_size)
            .set_globally_allocated_address(*src_buffer);

    std::cout << "CB src config total size: " << tiles_per_core_width * input_page_size
              << " page size: " << input_page_size << std::endl;

    // CB to represent the output sharded buffer
    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(
            tiles_per_core_width_output * input_page_size, {{dst_cb_index, cb_data_format}})
            .set_page_size(dst_cb_index, input_page_size)
            .set_globally_allocated_address(*dst_buffer);

    std::cout << "CB dst config total size: " << tiles_per_core_width_output * input_page_size
              << " page size: " << input_page_size << std::endl;

    // Allocate space for the client interface
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes, {{packet_header_cb_index, DataFormat::UInt32}})
            .set_page_size(packet_header_cb_index, packet_header_size_bytes);

    std::cout << "CB packet header config total size: " << num_packet_headers_storable * packet_header_size_bytes * 2
              << " page size: " << packet_header_size_bytes << std::endl;
    // Add buffer for the packet + next packet (entire shard x 2)
    uint32_t buffering_factor = 2;
    // input shards are 5 tiles wide, and each goes to a different core on the next device
    tt::tt_metal::CircularBufferConfig fabric_sender_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * tiles_per_core_width * input_page_size, {{fabric_sender_cb_index, cb_data_format}})
            .set_page_size(fabric_sender_cb_index, tiles_per_core_width * input_page_size);

    std::cout << "CB fabric sender config total size: " << buffering_factor * tiles_per_core_width * input_page_size
              << " page size: " << tiles_per_core_width * input_page_size << std::endl;
    // buffer for receiving shards from the other device(s) - size is num_devices for simplicity, but could be reduced
    // to num_devicees - 1
    tt::tt_metal::CircularBufferConfig fabric_receiver_cb_config =
        tt::tt_metal::CircularBufferConfig(
            tiles_per_core_width_output * input_page_size * num_devices, {{fabric_receiver_cb_index, cb_data_format}})
            .set_page_size(fabric_receiver_cb_index, input_page_size);

    std::cout << "CB fabric receiver config total size: " << tiles_per_core_width_output * input_page_size * num_devices
              << " page size: " << input_page_size << std::endl;

    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_src_config);  // input buffer
    auto cb_dst =
        tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, cb_dst_config);  // output buffer
    auto cb_client_interface = tt::tt_metal::CreateCircularBuffer(
        program, sender_worker_core_range, packet_header_cb_config);  // client interface
    // fabric receiver - we put the reduced shards here after we receive them from the other device, then we reduce with
    // the input buffer shard and pack into the fabric sender buffer
    auto cb_fabric_receiver =
        tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, fabric_receiver_cb_config);
    // fabric sender - we put the payload and packet header here after we apply the reduction, and then send it to the
    // next device, or if it's the last device, we write it to the output buffer
    auto cb_fabric_sender =
        tt::tt_metal::CreateCircularBuffer(program, sender_worker_core_range, fabric_sender_cb_config);

    const uint32_t chip_id = ring_index;

    std::vector<uint32_t> reader_compile_time_args = {
        src_cb_index,
        fabric_sender_cb_index,
        (uint32_t)chip_id,
        tiles_per_core_width,
        cores_per_device,
        num_devices,
        input_page_size,
    };

    shard_builder::extend_sharding_compile_time_args(input_tensor, reader_compile_time_args);
    std::vector<uint32_t> reader_runtime_args;
    shard_builder::extend_sharding_run_time_args(input_tensor, reader_runtime_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
        "reader_llama_reduce_scatter.cpp",
        sender_core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, sender_core, reader_runtime_args);

    std::vector<uint32_t> compute_kernel_args = {};

    std::vector<uint32_t> writer_compile_time_args = {
        dst_cb_index,
        fabric_sender_cb_index,
        fabric_receiver_cb_index,
        packet_header_cb_index,
        (uint32_t)chip_id,
        tiles_per_core_width,
        cores_per_device,
        num_devices,
        input_page_size,
    };

    shard_builder::extend_sharding_compile_time_args(output_tensor, writer_compile_time_args);

    std::vector<uint32_t> writer_runtime_args = {semaphore_address};
    shard_builder::extend_sharding_run_time_args(output_tensor, writer_runtime_args);

    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> forward_fabric_connection =
        line_topology.is_first_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
            ? std::nullopt
            : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(
                  local_fabric_handle->uniquely_connect_worker(device, ttnn::ccl::EdmLineFabricOpInterface::FORWARD));
    std::optional<ttnn::ccl::SenderWorkerAdapterSpec> backward_fabric_connection =
        line_topology.is_last_device_in_line(ttnn::ccl::EdmLineFabricOpInterface::Direction::BACKWARD)
            ? std::nullopt
            : std::optional<ttnn::ccl::SenderWorkerAdapterSpec>(
                  local_fabric_handle->uniquely_connect_worker(device, ttnn::ccl::EdmLineFabricOpInterface::BACKWARD));

    auto writer_defines = reader_defines;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
        "writer_llama_reduce_scatter.cpp",
        sender_core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    detail::append_fabric_connection_rt_args(forward_fabric_connection, sender_core, program, writer_runtime_args);
    detail::append_fabric_connection_rt_args(backward_fabric_connection, sender_core, program, writer_runtime_args);

    std::cout << "Sender core: " << sender_core.x << ", " << sender_core.y << std::endl;
    tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, sender_core, writer_runtime_args);
    std::cout << "Made it to return " << chip_id << std::endl;
    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .cb_ids = {src_cb_index, dst_cb_index},
         .core_range = all_cores,
         .sender_core_range = sender_worker_core_range}};
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
    auto& sender_cores = cached_program.shared_variables.sender_core_range;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    auto sender_cores_list = corerange_to_cores(sender_cores, std::nullopt);

    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_ids[0], *src_buffer);
    UpdateDynamicCircularBufferAddress(program, cached_program.shared_variables.cb_ids[1], *dst_buffer);

    for (const auto& core : sender_cores_list) {
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        writer_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
    }
}

}  // namespace ttnn::operations::experimental::ccl
