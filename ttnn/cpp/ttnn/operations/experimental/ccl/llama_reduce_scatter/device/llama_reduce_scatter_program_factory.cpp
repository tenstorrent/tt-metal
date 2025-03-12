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
#include <"tt_metal/api/>tt-metalium/core_coord.hpp>

namespace ttnn::operations::experimental::ccl {

namespace detail {
CoreCoord next_core(const CoreCoord& current, const CoreCoord& grid_size, bool row_wise = true) {
    CoreCoord next = current;
    if (row_wise) {
        // Move right, then down
        next.x++;
        if (next.x >= grid_size.x) {
            next.x = 0;
            next.y++;
            if (next.y >= grid_size.y) {
                next.y = 0;
            }
        }
    } else {
        // Move down, then right
        next.y++;
        if (next.y >= grid_size.y) {
            next.y = 0;
            next.x++;
            if (next.x >= grid_size.x) {
                next.x = 0;
            }
        }
    }
    return next;
}

// Get the next core in the core range set that is outside of the bounding box
// This is used to get the next core to assign as a receiver worker
CoreCoord next_core_bounded(const CoreRangeSet& range, const CoreCoord& grid_size, bool row_wise = true) {
    // Get the bounding box of the range set and use its end coordinate
    CoreRange bbox = range.bounding_box();
    return next_core(bbox.end_coord, grid_size, row_wise);
}

CoreRangeSet next_core_range_set(
    const CoreRangeSet& range, const CoreCoord& grid_size, const uint32_t num_cores, bool row_wise = true) {
    TT_FATAL(num_cores > 0, "num_cores requested must be greater than 0");
    TT_FATAL(
        num_cores <= range.num_cores(),
        "num_cores requested must be less than or equal to the number of cores in the range set");
    auto first = next_core_bounded(range, grid_size, row_wise);
    CoreCoord last = first;
    for (uint32_t i = 0; i < num_cores - 1; i++) {
        last = next_core(last, grid_size, row_wise);
    }
    auto res = CoreRangeSet({CoreRange(first, last)});
    TT_FATAL(res.num_cores() == num_cores, "num_cores requested must be equal to the number of cores in the range set");
    return res;
}

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

std::string cores_to_string(std::vector<CoreCoord> cores) {
    std::string result = "{";
    for (const auto& core : cores) {
        result += "{" + std::to_string(core.x) + ", " + std::to_string(core.y) + "}, ";
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
    auto src_buffer = input_tensor.buffer();
    auto dst_buffer = output_tensor.buffer();

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};

    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, ttnn::ccl::Topology::Linear);
    LineTopology line_topology(ring_size, ring_index);

    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];
    uint32_t tiles_per_core_width = shard_width / tile_shape[1];

    uint32_t shard_height_output = output_shard_spec.shape[0];
    uint32_t shard_width_output = output_shard_spec.shape[1];
    uint32_t tiles_per_core_width_output = shard_width_output / tile_shape[1];
    auto input_grid = shard_spec.grid;
    auto output_grid = output_shard_spec.grid;
    auto receiver_grid = detail::next_core_range_set(
        output_grid,
        device->compute_with_storage_grid_size(),
        num_devices,
        shard_spec.orientation == ShardOrientation::ROW_MAJOR);

    TT_FATAL(output_grid.intersects(receiver_grid) == false, "output_grid and receiver_grid must not intersect");

    tt::tt_metal::Program program{};

    std::optional<ttnn::ccl::EdmLineFabricOpInterface> local_fabric_handle =
        ttnn::ccl::EdmLineFabricOpInterface::build_program_builder_worker_connection_fabric(
            device,
            operation_attributes.forward_device.value_or(nullptr),
            operation_attributes.backward_device.value_or(nullptr),
            &program,
            enable_persistent_fabric,
            num_links);

    // One sender per link - rn hardcoded to 6,6 for testing
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] = choose_worker_cores(
        num_links, num_workers_per_link, enable_persistent_fabric, device, operation_attributes.subdevice_id);

    auto sender_core = sender_worker_cores.at(0);

    auto all_cores = input_grid.merge(receiver_grid).merge(output_grid).merge(sender_core);

    std::cout << "input_grid: " << input_grid.str() << std::endl;
    std::cout << "output_grid: " << output_grid.str() << std::endl;
    std::cout << "receiver_grid: " << receiver_grid.str() << std::endl;
    std::cout << "sender_cores: " << CoreRange(sender_core).str() << std::endl;
    std::cout << "all_cores: " << all_cores.str() << std::endl;

    // input sharded buffer
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    // output sharded buffer
    uint32_t output_tensor_cb_id = tt::CBIndex::c_1;
    // client interface
    uint32_t packet_header_cb_index = tt::CBIndex::c_2;
    // fabric sender
    uint32_t fabric_sender_cb_index = tt::CBIndex::c_3;
    // fabric receiver where we receive the data from the other device
    uint32_t fabric_receiver_cb_index = tt::CBIndex::c_4;
    // accumulator before we perform the reduction
    uint32_t accumulator_cb_index = tt::CBIndex::c_5;

    uint32_t ncores_input = shard_spec.num_cores();
    uint32_t input_shard_cores_per_device = ncores_input / num_devices;
    uint32_t ncores_output = output_grid.num_cores();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    uint32_t input_page_size = tile_size(
        cb_data_format);  // doesn't work for tiny tiles, there is likely some API somewhere but I don't know where
    uint32_t output_page_size = tile_size(cb_data_format);

    tt::tt_metal::CircularBufferConfig cb_src_config =
        tt::tt_metal::CircularBufferConfig(
            tiles_per_core_width * input_page_size, {{input_tensor_cb_id, cb_data_format}})
            .set_page_size(input_tensor_cb_id, input_page_size)
            .set_globally_allocated_address(*src_buffer);

    std::cout << "CB src config total size: " << tiles_per_core_width * input_page_size
              << " page size: " << input_page_size << std::endl;

    // CB to represent the output sharded buffer
    tt::tt_metal::CircularBufferConfig cb_dst_config =
        tt::tt_metal::CircularBufferConfig(
            tiles_per_core_width_output * input_page_size, {{output_tensor_cb_id, cb_data_format}})
            .set_page_size(output_tensor_cb_id, input_page_size)
            .set_globally_allocated_address(*dst_buffer);

    std::cout << "CB dst config total size: " << tiles_per_core_width_output * input_page_size
              << " page size: " << input_page_size << std::endl;

    constexpr uint32_t buffering_factor = 2;
    // Allocate space for the client interface
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * buffering_factor,
            {{packet_header_cb_index, DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_index, packet_header_size_bytes);

    std::cout << "CB packet header config total size: "
              << num_packet_headers_storable * packet_header_size_bytes * buffering_factor
              << " page size: " << packet_header_size_bytes << std::endl;

    // input shards are 5 tiles wide, and each goes to a different core on the next device
    // max packet size is 4 tiles wide, so we only send 4/5 of the tiles at once
    // ideally we send 5/5 of the tiles at once, but we need to make sure we don't overflow the fabric buffer
    // so we send 0-3 of the first shard, then 4 of the first shard + 0-2 of the second shard, etc.
    const size_t packet_size_bytes = local_fabric_handle->get_edm_buffer_size_bytes();
    uint32_t num_pages_per_packet = packet_size_bytes / input_page_size;
    TT_FATAL(num_pages_per_packet <= tiles_per_core_width, "num_pages_per_packet is less than tiles_per_core_width");
    std::cout << "num_pages_per_packet: " << num_pages_per_packet << std::endl;
    std::cout << "tiles_per_core_width: " << tiles_per_core_width << std::endl;

    tt::tt_metal::CircularBufferConfig fabric_sender_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * num_pages_per_packet * input_page_size, {{fabric_sender_cb_index, cb_data_format}})
            .set_page_size(fabric_sender_cb_index, num_pages_per_packet * input_page_size);

    std::cout << "CB fabric sender config total size: " << buffering_factor * num_pages_per_packet * input_page_size
              << " page size: " << num_pages_per_packet * input_page_size << std::endl;

    // buffer for receiving shards from the another device. Each receiver core will take each device's shard. The
    // receiver core for the local device will do nothing.
    tt::tt_metal::CircularBufferConfig fabric_receiver_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * input_shard_cores_per_device * tiles_per_core_width * input_page_size,
            {{fabric_receiver_cb_index, cb_data_format}})
            .set_page_size(fabric_receiver_cb_index, input_page_size);

    std::cout << "CB fabric receiver config total size: " << buffering_factor * num_pages_per_packet * input_page_size
              << " page size: " << input_page_size << std::endl;

    tt::tt_metal::CircularBufferConfig accumulator_cb_config =
        tt::tt_metal::CircularBufferConfig(
            buffering_factor * tiles_per_core_width_output * input_page_size * num_devices,
            {{accumulator_cb_index, cb_data_format}})
            .set_page_size(accumulator_cb_index, input_page_size);

    std::cout << "CB accumulator config total size: "
              << buffering_factor * tiles_per_core_width_output * input_page_size
              << " page size: " << tiles_per_core_width_output * input_page_size << std::endl;

    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src_config);  // input buffer
    auto cb_dst = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_dst_config);  // output buffer
    auto cb_client_interface =
        tt::tt_metal::CreateCircularBuffer(program, all_cores, packet_header_cb_config);  // client interface
    auto cb_fabric_receiver = tt::tt_metal::CreateCircularBuffer(program, all_cores, fabric_receiver_cb_config);
    auto cb_fabric_sender = tt::tt_metal::CreateCircularBuffer(program, all_cores, fabric_sender_cb_config);
    auto cb_accumulator = tt::tt_metal::CreateCircularBuffer(program, all_cores, accumulator_cb_config);

    const uint32_t chip_id = ring_index;

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_id,
        fabric_sender_cb_index,
        packet_header_cb_index,
        fabric_receiver_cb_index,
        accumulator_cb_index,
        output_tensor_cb_id,
        (uint32_t)chip_id,
        tiles_per_core_width,
        tiles_per_core_width_output,
        num_pages_per_packet,
        input_shard_cores_per_device,
        num_devices,
        input_page_size,
        ncores_output,

    };
    reader_compile_time_args.push_back(fabric_receiver_cb_index);
    reader_compile_time_args.push_back(accumulator_cb_index);
    reader_compile_time_args.push_back(tiles_per_core_width_output);
    reader_compile_time_args.push_back(num_pages_per_packet);

    auto to_worker_cores = [device](const std::vector<CoreCoord>& cores) -> std::vector<CoreCoord> {
        std::vector<CoreCoord> worker_cores;
        for (const auto& core : cores) {
            worker_cores.push_back(device->worker_core_from_logical_core(core));
        }
        return worker_cores;
    };
    auto input_cores = corerange_to_cores(input_grid, std::nullopt);
    auto output_cores = corerange_to_cores(output_grid, std::nullopt);
    auto receiver_cores = corerange_to_cores(receiver_grid, std::nullopt);
    auto sender_cores = corerange_to_cores(sender_worker_core_range, std::nullopt);

    reader_defines["INPUT_CORE_XY"] = detail::cores_to_string(to_worker_cores(input_cores));
    reader_defines["OUTPUT_CORE_XY"] = detail::cores_to_string(to_worker_cores(output_cores));
    reader_defines["RECEIVER_CORE_XY"] = detail::cores_to_string(to_worker_cores(receiver_cores));

    std::cout << "input_cores: " << reader_defines["INPUT_CORE_XY"] << std::endl;
    std::cout << "output_cores: " << reader_defines["OUTPUT_CORE_XY"] << std::endl;

    // create local semaphore
    auto local_semaphore = tt::tt_metal::CreateSemaphore(program, all_cores, INVALID);

    std::vector<uint32_t> reader_runtime_args = {cross_device_semaphore->address(), local_semaphore, false, false};
    uint32_t is_reader_sender_core_idx = reader_runtime_args.size() - 2;
    uint32_t is_worker_core_idx = reader_runtime_args.size() - 1;

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
        "reader_llama_reduce_scatter.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, reader_defines));

    for (auto core : sender_cores) {
        reader_runtime_args[is_reader_sender_core_idx] = true;
        reader_runtime_args[is_worker_core_idx] = false;
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
    }

    for (auto core : input_cores) {
        reader_runtime_args[is_reader_sender_core_idx] = false;
        reader_runtime_args[is_worker_core_idx] = true;
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);
    }

    std::vector<uint32_t> compute_kernel_args = {};

    std::vector<uint32_t> writer_compile_time_args = {
        input_tensor_cb_id,
        fabric_sender_cb_index,
        packet_header_cb_index,
        fabric_receiver_cb_index,
        accumulator_cb_index,
        output_tensor_cb_id,
        (uint32_t)chip_id,
        tiles_per_core_width,
        tiles_per_core_width_output,
        num_pages_per_packet,
        input_shard_cores_per_device,
        num_devices,
        input_page_size,
    };

    auto writer_defines = reader_defines;
    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_reduce_scatter/device/kernels/dataflow/"
        "writer_llama_reduce_scatter.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, writer_defines));

    std::vector<uint32_t> writer_runtime_args = {cross_device_semaphore->address(), local_semaphore, false, false};
    uint32_t is_writer_sender_core_idx = writer_runtime_args.size() - 2;
    uint32_t is_receiver_core_idx = writer_runtime_args.size() - 1;

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

    detail::append_fabric_connection_rt_args(forward_fabric_connection, sender_core, program, writer_runtime_args);
    std::cout << "Writer runtime args after appending forward fabric connection: " << writer_runtime_args.size()
              << std::endl;
    detail::append_fabric_connection_rt_args(backward_fabric_connection, sender_core, program, writer_runtime_args);
    std::cout << "Writer runtime args after appending backward fabric connection: " << writer_runtime_args.size()
              << std::endl;

    std::cout << "Sender core: " << sender_core.x << ", " << sender_core.y << std::endl;

    for (auto core : sender_cores) {
        writer_runtime_args[is_writer_sender_core_idx] = true;
        writer_runtime_args[is_receiver_core_idx] = false;
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
    }

    for (auto core : receiver_cores) {
        writer_runtime_args[is_writer_sender_core_idx] = false;
        writer_runtime_args[is_receiver_core_idx] = true;
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_runtime_args);
    }

    std::cout << "Made it to return " << chip_id << std::endl;
    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id,
         .unary_writer_kernel_id = unary_writer_kernel_id,
         .cb_ids = {input_tensor_cb_id, output_tensor_cb_id},
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

    for (const auto& core : cores) {
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, core);
        writer_runtime_args[0] = (uint32_t)operation_attributes.cross_device_semaphore->address();
    }
}

}  // namespace ttnn::operations::experimental::ccl
