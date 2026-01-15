// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_program_factory.hpp"
#include "all_to_all_async_device_operation_types.hpp"

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/constants.hpp>
#include <algorithm>
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
#include <tuple>

using namespace tt::constants;

namespace ttnn::operations::experimental::ccl::all_to_all_async {

namespace detail {
// Configuration constants
constexpr uint32_t MIN_CHUNK_GRANULARITY = 4;
constexpr uint32_t MAX_CHUNKS_PER_SHARD = 30;
constexpr uint32_t TRIPLE_BUFFER_MULTIPLIER = 3;
constexpr uint32_t PACKET_HEADER_BUFFER_SIZE = 8;

// Calculate optimal chunk parameters to keep num_chunks < MAX_CHUNKS_PER_SHARD
std::tuple<uint32_t, uint32_t, uint32_t> calculate_chunk_params(uint32_t pages_per_shard, uint32_t pages_per_packet) {
    uint32_t chunk_granularity = MIN_CHUNK_GRANULARITY;

    while (true) {
        const uint32_t chunk_tiles = chunk_granularity * pages_per_packet;
        const uint32_t num_chunks = tt::div_up(pages_per_shard, chunk_tiles);

        if (num_chunks < MAX_CHUNKS_PER_SHARD) {
            return {chunk_granularity, chunk_tiles, num_chunks};
        }
        chunk_granularity *= 2;
    }
}

// Create circular buffers for sender cores
auto create_sender_buffers(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& sender_core_range,
    uint32_t cb_num_pages,
    uint32_t page_size,
    tt::DataFormat data_format) {
    // Main data buffer
    auto cb_src0_config = tt::tt_metal::CircularBufferConfig(cb_num_pages * page_size, {{tt::CB::c_in0, data_format}})
                              .set_page_size(tt::CB::c_in0, page_size);

    auto cb_src0_handle = CreateCircularBuffer(program, sender_core_range, cb_src0_config);

    // Packet header buffer
    auto header_buffer_config =
        tt::tt_metal::CircularBufferConfig(
            PACKET_HEADER_BUFFER_SIZE * tt::tt_fabric::get_tt_fabric_packet_header_size_bytes() * 2,
            {{tt::CB::c_in1, tt::DataFormat::RawUInt32}})
            .set_page_size(tt::CB::c_in1, tt::tt_fabric::get_tt_fabric_packet_header_size_bytes());

    auto header_buffer_handle = CreateCircularBuffer(program, sender_core_range, header_buffer_config);

    return std::make_tuple(cb_src0_handle, header_buffer_handle);
}

// Create circular buffers for receiver cores
auto create_receiver_buffer(
    tt::tt_metal::Program& program,
    const tt::tt_metal::CoreRangeSet& receiver_core_range,
    uint32_t pages_per_packet,
    uint32_t page_size,
    tt::DataFormat data_format) {
    const uint32_t receiver_pages = pages_per_packet * TRIPLE_BUFFER_MULTIPLIER;

    auto config = tt::tt_metal::CircularBufferConfig(receiver_pages * page_size, {{tt::CB::c_in0, data_format}})
                      .set_page_size(tt::CB::c_in0, page_size);

    return CreateCircularBuffer(program, receiver_core_range, config);
}

static std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> calculate_strides_and_offsets(
    uint32_t in_row_tiles, uint32_t in_col_tiles, uint32_t ring_size, uint32_t ring_index, uint32_t in_dim) {
    uint32_t input_row_device_stride = 0;
    uint32_t input_col_device_stride = 0;
    uint32_t out_row_start = 0;
    uint32_t out_col_start = 0;
    uint32_t input_shard_row_tiles = in_row_tiles;
    uint32_t input_shard_col_tiles = in_col_tiles;

    if (in_dim == 2) {
        // out_dim == 3
        input_col_device_stride = in_col_tiles / ring_size;
        out_row_start = ring_index * in_row_tiles;
        input_shard_col_tiles = input_col_device_stride;
    } else if (in_dim == 3) {
        // out_dim == 2
        input_row_device_stride = in_row_tiles / ring_size;
        out_col_start = ring_index * in_col_tiles;
        input_shard_row_tiles = input_row_device_stride;
    }

    return std::make_tuple(
        input_row_device_stride,
        input_col_device_stride,
        out_row_start,
        out_col_start,
        input_shard_row_tiles,
        input_shard_col_tiles);
}

}  // namespace detail

AllToAllAsyncProgram::cached_mesh_workload_t AllToAllAsyncProgram::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<AllToAllAsyncProgram::shared_variables_t> AllToAllAsyncProgram::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    Tensor& /*tensor_return_value*/) {
    log_debug(tt::LogOp, "DEBUG: create_at is called");
    auto* mesh_device = tensor_args.input_tensor.device();
    IDevice* target_device = mesh_device ? mesh_device->get_device(mesh_coordinate) : tensor_args.input_tensor.device();

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;
    uint32_t ring_index = operation_attributes.ring_size;  // Initialize device index

    TT_FATAL(
        operation_attributes.topology == ttnn::ccl::Topology::Ring,
        "DEBUG: topology: {}",
        operation_attributes.topology);

    std::vector<IDevice*> devices_to_use = tensor_args.input_tensor.device()->get_view().get_ring_devices();

    for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            ring_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(operation_attributes.ring_size - 1);
            }
            if (i != operation_attributes.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    TT_FATAL(ring_index < operation_attributes.ring_size, "DEBUG: ring_index: {}", ring_index);
    TT_FATAL(
        forward_device.value()->id() != backward_device.value()->id(),
        "DEBUG: forward and backward devices are the same: {}, {}",
        forward_device.value()->id(),
        backward_device.value()->id());
    TT_FATAL(
        forward_device.value()->id() != target_device->id(),
        "DEBUG: forward device is the same as target device: {}, {}",
        forward_device.value()->id(),
        target_device->id());
    TT_FATAL(
        backward_device.value()->id() != target_device->id(),
        "DEBUG: backward device is the same as target device: {}, {}",
        backward_device.value()->id(),
        target_device->id());

    // Implementation moved from all_to_all_async_minimal
    const auto& semaphore = operation_attributes.semaphore;

    tt::tt_metal::Program program{};
    IDevice* device = tensor_args.input_tensor.device();

    // Basic configuration
    const bool enable_async_output = false;
    [[maybe_unused]] const bool is_first_chip = ring_index == 0;
    [[maybe_unused]] const bool is_last_chip = ring_index == operation_attributes.ring_size - 1;

    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        tensor_args.input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {tensor_args.input_tensor};
    std::vector<Tensor> output_tensors = {
        tensor_args.persistent_intermediate_buffer, tensor_args.persistent_output_buffer};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, operation_attributes.topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] = ttnn::ccl::get_forward_backward_configuration(
        operation_attributes.ring_size, ring_index, operation_attributes.topology);

    log_trace(
        tt::LogOp,
        "ring_index: {}, num_targets_forward: {}, num_targets_backward: {}, dynamic_alternate: {}",
        ring_index,
        num_targets_forward,
        num_targets_backward,
        dynamic_alternate);

    TT_FATAL(!dynamic_alternate, "Dynamic alternate is not supported");

    /**
     * Get worker cores. 1 sender per link, ring_size receivers per link.
     *
     * It's ring_size rather than (ring_size-1) since this simplifies receiver
     * core coordinates. Device D always targets receiver core D on remote devices.
     */
    uint32_t num_senders_per_link = 1;
    uint32_t num_receivers_per_link = operation_attributes.ring_size;
    const auto [sender_worker_core_range, sender_worker_cores] = ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links, num_senders_per_link, device, operation_attributes.sub_device_id);

    const auto [total_workers_core_range, total_workers_cores] = ttnn::ccl::choose_worker_cores(
        operation_attributes.num_links,
        (num_senders_per_link + num_receivers_per_link),
        device,
        operation_attributes.sub_device_id);

    const auto receiver_worker_core_range = total_workers_core_range.subtract(sender_worker_core_range);
    const auto receiver_worker_cores = corerange_to_cores(receiver_worker_core_range, std::nullopt, true);
    TT_FATAL(
        receiver_worker_cores.size() == operation_attributes.num_links * num_receivers_per_link,
        "Expected {} receiver cores but got {}",
        operation_attributes.num_links * num_receivers_per_link,
        receiver_worker_cores.size());

    // Calculate buffer parameters
    const size_t packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t page_size = op_config.get_page_size();
    const uint32_t pages_per_packet = packet_size / page_size;
    const uint32_t cb_pages = detail::TRIPLE_BUFFER_MULTIPLIER * pages_per_packet;

    // Create buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    detail::create_sender_buffers(program, sender_worker_core_range, cb_pages, page_size, data_format);
    detail::create_receiver_buffer(program, receiver_worker_core_range, pages_per_packet, page_size, data_format);

    const auto [chunk_granularity, chunk_num_tiles, num_chunks_per_shard] = detail::calculate_chunk_params(
        tensor_args.input_tensor.buffer()->num_pages() /
            operation_attributes.ring_size,  // number of pages sent between each pair of devices
        pages_per_packet);

    const uint32_t contig_pages_advanced = pages_per_packet;  // Write 2 tiles per packet

    log_trace(
        tt::LogOp,
        "chunk_granularity: {}, chunk_num_tiles: {}, num_chunks_per_shard: {}",
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard);

    // Verify constraints
    TT_FATAL(num_chunks_per_shard < 32, "num_chunks_per_shard must be < 32, got {}", num_chunks_per_shard);
    TT_FATAL(chunk_granularity >= 4, "chunk_granularity must be >= 4, got {}", chunk_granularity);

    const uint32_t receiver_cb_index = tt::CB::c_in0;

    // Tensor Info
    const auto input_tensor_num_pages = tensor_args.input_tensor.buffer()->num_pages();

    const uint32_t N_DRAM_BANKS = device->num_dram_channels();

    TT_FATAL(
        num_chunks_per_shard < 31,
        "num_chunks_per_shard: {}, chunk_num_tiles: {}, input_tensor_num_pages: {}",
        num_chunks_per_shard,
        chunk_num_tiles,
        input_tensor_num_pages);

    // KERNEL CREATION
    // Reader
    auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    reader_kernel_config.compile_args = {
        ring_index,                      // my_chip_id
        operation_attributes.ring_size,  // num_chips
        tt::CB::c_in0,                   // cb0_id
        pages_per_packet,                // packet_size_in_pages
        op_config.get_page_size(),       // tensor0_page_size
        num_targets_forward,             // num_targets_forward_direction
        num_targets_backward             // num_targets_backward_direction
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer()).append_to(reader_kernel_config.compile_args);
    log_trace(tt::LogOp, "Reader Compile Args:");
    for ([[maybe_unused]] const auto& arg : reader_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_reader.cpp",
        sender_worker_core_range,
        reader_kernel_config);

    // Writer
    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    writer_kernel_config.compile_args = {
        ring_index,                         // my_chip_id
        operation_attributes.ring_size,     // num_chips
        tt::CB::c_in1,                      // reserved_packet_header_cb_id
        detail::PACKET_HEADER_BUFFER_SIZE,  // num_packet_headers_storable
        tt::CB::c_in0,                      // cb0_id
        pages_per_packet,                   // packet_size_in_pages
        op_config.get_page_size(),          // tensor0_page_size
        num_targets_forward,                // num_targets_forward_direction
        num_targets_backward,               // num_targets_backward_direction
        dynamic_alternate,                  // alternate
        chunk_granularity,                  // granularity of signaling to receiver
        contig_pages_advanced,              // contig_pages_advanced
        N_DRAM_BANKS                        // num_dram_banks
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.persistent_intermediate_buffer.buffer())
        .append_to(writer_kernel_config.compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.persistent_output_buffer.buffer())
        .append_to(writer_kernel_config.compile_args);
    for ([[maybe_unused]] const auto& arg : writer_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_writer.cpp",
        sender_worker_core_range,
        writer_kernel_config);

    // Create receiver kernels
    auto receiver_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    receiver_writer_kernel_config.compile_args = {
        ring_index,
        operation_attributes.ring_size,
        pages_per_packet,
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard,
        op_config.get_page_size(),
        receiver_cb_index};
    tt::tt_metal::TensorAccessorArgs(tensor_args.persistent_output_buffer.buffer())
        .append_to(receiver_writer_kernel_config.compile_args);

    auto receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_receiver_writer.cpp",
        receiver_worker_core_range,
        receiver_writer_kernel_config);

    auto receiver_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    receiver_reader_kernel_config.compile_args = {
        ring_index,
        operation_attributes.ring_size,
        pages_per_packet,
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard,
        op_config.get_page_size(),
        receiver_cb_index,
        pages_per_packet,
        N_DRAM_BANKS};
    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer())
        .append_to(receiver_reader_kernel_config.compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.persistent_intermediate_buffer.buffer())
        .append_to(receiver_reader_kernel_config.compile_args);
    auto receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_receiver_reader.cpp",
        receiver_worker_core_range,
        receiver_reader_kernel_config);

    // Determine output shape and fracturing
    const auto& input_shape = tensor_args.input_tensor.padded_shape();
    const auto in_row_tiles = input_shape[2] / tt::constants::TILE_HEIGHT;
    const auto in_col_tiles = input_shape[3] / tt::constants::TILE_WIDTH;
    auto output_shape = tensor_args.persistent_output_buffer.padded_shape();
    const auto out_row_tiles = output_shape[2] / tt::constants::TILE_HEIGHT;
    const auto out_col_tiles = output_shape[3] / tt::constants::TILE_WIDTH;

    auto
        [input_row_device_stride,
         input_col_device_stride,
         out_row_start,
         out_col_start,
         input_shard_row_tiles,
         input_shard_col_tiles] =
            detail::calculate_strides_and_offsets(
                in_row_tiles, in_col_tiles, operation_attributes.ring_size, ring_index, operation_attributes.in_dim);

    // Check that with this static partitioning, the final packet does not overflow the intermediate buffer
    const uint32_t packets_per_row = tt::div_up(input_shard_col_tiles, contig_pages_advanced);
    const uint32_t packets_per_device_shard = packets_per_row * input_shard_row_tiles;
    const uint32_t final_packet_id = packets_per_device_shard - 1;
    const uint32_t final_packet_global_id =
        (operation_attributes.ring_size - 2) + (final_packet_id * (operation_attributes.ring_size - 1));

    const uint32_t final_packet_first_tile_id =
        (final_packet_global_id % N_DRAM_BANKS) +
        (contig_pages_advanced * N_DRAM_BANKS * (final_packet_global_id / N_DRAM_BANKS));
    const uint32_t final_packet_second_tile_id = final_packet_first_tile_id + N_DRAM_BANKS;
    TT_FATAL(
        final_packet_second_tile_id < tensor_args.persistent_intermediate_buffer.buffer()->num_pages(),
        "Final packet would overflow intermediate buffer. This is unexpected and dangerous. "
        "final_packet_first_tile_id: {}, final_packet_second_tile_id: {}, intermediate_buffer.buffer()->num_pages(): "
        "{}",
        final_packet_first_tile_id,
        final_packet_second_tile_id,
        tensor_args.persistent_intermediate_buffer.buffer()->num_pages());

    // Kernel Runtime Args
    const uint32_t receiver_core_x = device->worker_core_from_logical_core(receiver_worker_cores[ring_index]).x;
    const uint32_t receiver_core_y = device->worker_core_from_logical_core(receiver_worker_cores[ring_index]).y;

    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore

    for (uint32_t link = 0; link < operation_attributes.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = device->worker_core_from_logical_core(core);
        }

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            tensor_args.input_tensor.buffer()->address(),  // tensor_address0
            in_row_tiles,
            in_col_tiles,
            input_row_device_stride,
            input_col_device_stride,
            input_shard_row_tiles,
            input_shard_col_tiles,
            out_row_start,
            out_col_start,
        };
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for ([[maybe_unused]] const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output;
        bool reset_global_semaphore = (link == 0) && !enable_async_output;
        std::vector<uint32_t> writer_rt_args = {
            tensor_args.persistent_intermediate_buffer.buffer()->address(),
            tensor_args.persistent_output_buffer.buffer()->address(),
            semaphore.address(),
            out_row_tiles,
            out_col_tiles,
            out_row_start,
            out_col_start,
            input_shard_row_tiles,
            input_shard_col_tiles,
            wait_output_semaphore,
            reset_global_semaphore,
            receiver_core_x,
            receiver_core_y,
        };

        log_trace(tt::LogOp, "Writer Runtime Args:");
        for ([[maybe_unused]] const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            const auto sender_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
            const auto forward_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_fabric_node_id, forward_device_fabric_node_id, link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            const auto sender_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(target_device->id());
            const auto backward_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(backward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_fabric_node_id, backward_device_fabric_node_id, link, program, {core}, writer_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

        for (uint32_t i = 0; i < receiver_worker_cores.size(); i++) {
            const auto core = receiver_worker_cores[i];
            // Compute strides and offsets for receiver, as if we are device at ring_index i.
            // This lets receiver mimic sender logic.
            auto
                [receiver_input_row_device_stride,
                 receiver_input_col_device_stride,
                 receiver_out_row_start,
                 receiver_out_col_start,
                 receiver_input_shard_row_tiles,
                 receiver_input_shard_col_tiles] =
                    detail::calculate_strides_and_offsets(
                        in_row_tiles, in_col_tiles, operation_attributes.ring_size, i, operation_attributes.in_dim);

            // Set receiver runtime args
            std::vector<uint32_t> receiver_reader_rt_args = {
                tensor_args.persistent_intermediate_buffer.buffer()->address(),
                tensor_args.input_tensor.buffer()->address(),
                semaphore.address(),  // Global semaphore for sender i
                in_row_tiles,
                in_col_tiles,
                receiver_input_row_device_stride,
                receiver_input_col_device_stride,
                receiver_input_shard_row_tiles,
                receiver_input_shard_col_tiles,
                receiver_out_row_start,
                receiver_out_col_start,
                out_row_tiles,
                out_col_tiles,
                pages_per_packet,
                i  // Receiver of device at ring_index i
            };
            tt::tt_metal::SetRuntimeArgs(program, receiver_reader_kernel_id, {core}, receiver_reader_rt_args);

            std::vector<uint32_t> receiver_writer_rt_args = {
                tensor_args.persistent_output_buffer.buffer()->address(),
                in_row_tiles,
                in_col_tiles,
                receiver_input_row_device_stride,
                receiver_input_col_device_stride,
                receiver_input_shard_row_tiles,
                receiver_input_shard_col_tiles,
                receiver_out_row_start,
                receiver_out_col_start,
                out_row_tiles,
                out_col_tiles,
                pages_per_packet,
                i  // Receiver of device at ring_index i
            };
            tt::tt_metal::SetRuntimeArgs(program, receiver_writer_kernel_id, {core}, receiver_writer_rt_args);
        }
    }

    // Store shared variables
    shared_variables_t shared_vars{
        .worker_sender_reader_kernel_id = worker_sender_reader_kernel_id,
        .worker_sender_writer_kernel_id = worker_sender_writer_kernel_id,
        .receiver_reader_kernel_id = receiver_reader_kernel_id,
        .receiver_writer_kernel_id = receiver_writer_kernel_id,
        .sender_worker_cores = sender_worker_cores,
        .receiver_worker_cores = receiver_worker_cores};

    return {std::move(program), std::move(shared_vars)};
}

void AllToAllAsyncProgram::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    // Update runtime arguments for each program in the mesh workload
    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        (void)coordinate_range;     // Suppress unused variable warning
        (void)tensor_return_value;  // Suppress unused variable warning
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        // Get runtime args for each kernel and update buffer addresses
        auto& worker_reader_sender_runtime_args_by_core =
            tt::tt_metal::GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            tt::tt_metal::GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);
        auto& receiver_writer_runtime_args_by_core =
            tt::tt_metal::GetRuntimeArgs(program, shared_vars.receiver_writer_kernel_id);
        auto& receiver_reader_runtime_args_by_core =
            tt::tt_metal::GetRuntimeArgs(program, shared_vars.receiver_reader_kernel_id);

        // Update sender runtime args
        for (const auto& core : shared_vars.sender_worker_cores) {
            // Update reader runtime args
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = tensor_args.input_tensor.buffer()->address();

            // Update writer runtime args
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = tensor_args.persistent_intermediate_buffer.buffer()->address();
            worker_writer_sender_runtime_args[1] = tensor_args.persistent_output_buffer.buffer()->address();
            worker_writer_sender_runtime_args[2] = operation_attributes.semaphore.address();
        }

        // Update receiver runtime args
        for (const auto& core : shared_vars.receiver_worker_cores) {
            auto& receiver_writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];
            receiver_writer_runtime_args[0] = tensor_args.persistent_output_buffer.buffer()->address();

            auto& receiver_reader_runtime_args = receiver_reader_runtime_args_by_core[core.x][core.y];
            receiver_reader_runtime_args[0] = tensor_args.persistent_intermediate_buffer.buffer()->address();
            receiver_reader_runtime_args[1] = tensor_args.input_tensor.buffer()->address();
            receiver_reader_runtime_args[2] = operation_attributes.semaphore.address();
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl::all_to_all_async
