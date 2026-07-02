// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_async_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "all_to_all_async_device_operation_types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/math.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
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

// Append sender CBs to the ProgramDescriptor (mirrors legacy create_sender_buffers).
void append_sender_buffers(
    ProgramDescriptor& desc,
    const tt::tt_metal::CoreRangeSet& sender_core_range,
    uint32_t cb_num_pages,
    uint32_t page_size,
    tt::DataFormat data_format) {
    // Main data buffer
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * page_size,
        .core_ranges = sender_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CB::c_in0), .data_format = data_format, .page_size = page_size}},
    });

    // Packet header buffer
    desc.cbs.push_back(CBDescriptor{
        .total_size = PACKET_HEADER_BUFFER_SIZE * tt::tt_fabric::get_tt_fabric_packet_header_size_bytes() * 2,
        .core_ranges = sender_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CB::c_in1),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes()}},
    });
}

// Append receiver CB to the ProgramDescriptor (mirrors legacy create_receiver_buffer).
void append_receiver_buffer(
    ProgramDescriptor& desc,
    const tt::tt_metal::CoreRangeSet& receiver_core_range,
    uint32_t pages_per_packet,
    uint32_t page_size,
    tt::DataFormat data_format) {
    const uint32_t receiver_pages = pages_per_packet * TRIPLE_BUFFER_MULTIPLIER;

    desc.cbs.push_back(CBDescriptor{
        .total_size = receiver_pages * page_size,
        .core_ranges = receiver_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(tt::CB::c_in0), .data_format = data_format, .page_size = page_size}},
    });
}

std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t> calculate_strides_and_offsets(
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

// Builds the ProgramDescriptor for one coord.  ring_index, forward/backward
// neighbours, and per-link/per-receiver strides vary with the coord; the rest
// mirrors the legacy create_at body verbatim.
ProgramDescriptor build_program_descriptor(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    Tensor& /*tensor_return_value*/,
    const ttnn::MeshCoordinate& mesh_coordinate) {
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

    const auto& mesh_view = tensor_args.input_tensor.device()->get_view();
    std::vector<IDevice*> devices_to_use = mesh_view.get_ring_devices();
    const auto fabric_node_ids = mesh_view.get_ring_fabric_node_ids();

    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    for (uint32_t i = 0; i < operation_attributes.ring_size; ++i) {
        if (devices_to_use.at(i) == target_device) {
            ring_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
                backward_fabric_node_id = fabric_node_ids.at(i - 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(operation_attributes.ring_size - 1);
                backward_fabric_node_id = fabric_node_ids.at(operation_attributes.ring_size - 1);
            }
            if (i != operation_attributes.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
                forward_fabric_node_id = fabric_node_ids.at(i + 1);
            } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
                forward_fabric_node_id = fabric_node_ids.at(0);
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
    const uint32_t cb_pages = TRIPLE_BUFFER_MULTIPLIER * pages_per_packet;

    ProgramDescriptor desc;

    // Create buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.input_tensor.dtype());
    append_sender_buffers(desc, sender_worker_core_range, cb_pages, page_size, data_format);
    append_receiver_buffer(desc, receiver_worker_core_range, pages_per_packet, page_size, data_format);

    const auto [chunk_granularity, chunk_num_tiles, num_chunks_per_shard] = calculate_chunk_params(
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
    std::vector<uint32_t> sender_reader_compile_args = {
        ring_index,                      // my_chip_id
        operation_attributes.ring_size,  // num_chips
        tt::CB::c_in0,                   // cb0_id
        pages_per_packet,                // packet_size_in_pages
        op_config.get_page_size(),       // tensor0_page_size
        num_targets_forward,             // num_targets_forward_direction
        num_targets_backward             // num_targets_backward_direction
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer()).append_to(sender_reader_compile_args);
    log_trace(tt::LogOp, "Reader Compile Args:");
    for ([[maybe_unused]] const auto& arg : sender_reader_compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    KernelDescriptor sender_reader_kernel_desc;
    sender_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_reader.cpp";
    sender_reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_reader_kernel_desc.core_ranges = sender_worker_core_range;
    sender_reader_kernel_desc.compile_time_args = std::move(sender_reader_compile_args);
    sender_reader_kernel_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(sender_reader_kernel_desc));
    const KernelHandle worker_sender_reader_kernel_id = desc.kernels.size() - 1;

    // Writer
    std::vector<uint32_t> sender_writer_compile_args = {
        ring_index,                      // my_chip_id
        operation_attributes.ring_size,  // num_chips
        tt::CB::c_in1,                   // reserved_packet_header_cb_id
        PACKET_HEADER_BUFFER_SIZE,       // num_packet_headers_storable
        tt::CB::c_in0,                   // cb0_id
        pages_per_packet,                // packet_size_in_pages
        op_config.get_page_size(),       // tensor0_page_size
        num_targets_forward,             // num_targets_forward_direction
        num_targets_backward,            // num_targets_backward_direction
        dynamic_alternate,               // alternate
        chunk_granularity,               // granularity of signaling to receiver
        contig_pages_advanced,           // contig_pages_advanced
        N_DRAM_BANKS                     // num_dram_banks
    };
    tt::tt_metal::TensorAccessorArgs(tensor_args.persistent_intermediate_buffer.buffer())
        .append_to(sender_writer_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.persistent_output_buffer.buffer())
        .append_to(sender_writer_compile_args);
    for ([[maybe_unused]] const auto& arg : sender_writer_compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    KernelDescriptor sender_writer_kernel_desc;
    sender_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_writer.cpp";
    sender_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_writer_kernel_desc.core_ranges = sender_worker_core_range;
    sender_writer_kernel_desc.compile_time_args = std::move(sender_writer_compile_args);
    sender_writer_kernel_desc.config = WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(sender_writer_kernel_desc));
    const KernelHandle worker_sender_writer_kernel_id = desc.kernels.size() - 1;

    // Create receiver kernels
    std::vector<uint32_t> receiver_writer_compile_args = {
        ring_index,
        operation_attributes.ring_size,
        pages_per_packet,
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard,
        op_config.get_page_size(),
        receiver_cb_index};
    tt::tt_metal::TensorAccessorArgs(tensor_args.persistent_output_buffer.buffer())
        .append_to(receiver_writer_compile_args);

    KernelDescriptor receiver_writer_kernel_desc;
    receiver_writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_receiver_writer.cpp";
    receiver_writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    receiver_writer_kernel_desc.core_ranges = receiver_worker_core_range;
    receiver_writer_kernel_desc.compile_time_args = std::move(receiver_writer_compile_args);
    receiver_writer_kernel_desc.config = WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(receiver_writer_kernel_desc));
    const KernelHandle receiver_writer_kernel_id = desc.kernels.size() - 1;

    std::vector<uint32_t> receiver_reader_compile_args = {
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
    tt::tt_metal::TensorAccessorArgs(tensor_args.input_tensor.buffer()).append_to(receiver_reader_compile_args);
    tt::tt_metal::TensorAccessorArgs(tensor_args.persistent_intermediate_buffer.buffer())
        .append_to(receiver_reader_compile_args);

    KernelDescriptor receiver_reader_kernel_desc;
    receiver_reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_receiver_reader.cpp";
    receiver_reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    receiver_reader_kernel_desc.core_ranges = receiver_worker_core_range;
    receiver_reader_kernel_desc.compile_time_args = std::move(receiver_reader_compile_args);
    receiver_reader_kernel_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(receiver_reader_kernel_desc));
    const KernelHandle receiver_reader_kernel_id = desc.kernels.size() - 1;

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
            calculate_strides_and_offsets(
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

        // Reader: input_tensor is a tensor buffer → BufferBinding.
        KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(tensor_args.input_tensor.buffer());  // Buffer* binding
        reader_rt_args.push_back(in_row_tiles);
        reader_rt_args.push_back(in_col_tiles);
        reader_rt_args.push_back(input_row_device_stride);
        reader_rt_args.push_back(input_col_device_stride);
        reader_rt_args.push_back(input_shard_row_tiles);
        reader_rt_args.push_back(input_shard_col_tiles);
        reader_rt_args.push_back(out_row_start);
        reader_rt_args.push_back(out_col_start);
        desc.kernels[worker_sender_reader_kernel_id].emplace_runtime_args(core, reader_rt_args);

        // Writer: persistent_intermediate/output are tensor buffers → bindings.
        // Build fabric tail in a raw vector; final RTArgList registers Buffer*.
        bool wait_output_semaphore = (link == 0) && !enable_async_output;
        bool reset_global_semaphore = (link == 0) && !enable_async_output;
        std::vector<uint32_t> writer_tail;
        writer_tail.push_back(forward_fabric_node_id.has_value());
        if (forward_fabric_node_id.has_value()) {
            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                sender_fabric_node_id, forward_fabric_node_id.value(), link, desc, core, writer_tail);
        }
        writer_tail.push_back(backward_fabric_node_id.has_value());
        if (backward_fabric_node_id.has_value()) {
            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                sender_fabric_node_id, backward_fabric_node_id.value(), link, desc, core, writer_tail);
        }
        KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(tensor_args.persistent_intermediate_buffer.buffer());  // binding
        writer_rt_args.push_back(tensor_args.persistent_output_buffer.buffer());        // binding
        writer_rt_args.push_back(semaphore.address());                                  // workload semaphore
        writer_rt_args.push_back(out_row_tiles);
        writer_rt_args.push_back(out_col_tiles);
        writer_rt_args.push_back(out_row_start);
        writer_rt_args.push_back(out_col_start);
        writer_rt_args.push_back(input_shard_row_tiles);
        writer_rt_args.push_back(input_shard_col_tiles);
        writer_rt_args.push_back(static_cast<uint32_t>(wait_output_semaphore));
        writer_rt_args.push_back(static_cast<uint32_t>(reset_global_semaphore));
        writer_rt_args.push_back(receiver_core_x);
        writer_rt_args.push_back(receiver_core_y);
        writer_rt_args.append(writer_tail);
        desc.kernels[worker_sender_writer_kernel_id].emplace_runtime_args(core, writer_rt_args);

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
                    calculate_strides_and_offsets(
                        in_row_tiles, in_col_tiles, operation_attributes.ring_size, i, operation_attributes.in_dim);

            // Receiver reader: persistent_intermediate + input_tensor are tensor buffers.
            KernelDescriptor::RTArgList receiver_reader_rt_args;
            receiver_reader_rt_args.push_back(tensor_args.persistent_intermediate_buffer.buffer());  // binding
            receiver_reader_rt_args.push_back(tensor_args.input_tensor.buffer());                    // binding
            receiver_reader_rt_args.push_back(semaphore.address());  // workload semaphore
            receiver_reader_rt_args.push_back(in_row_tiles);
            receiver_reader_rt_args.push_back(in_col_tiles);
            receiver_reader_rt_args.push_back(receiver_input_row_device_stride);
            receiver_reader_rt_args.push_back(receiver_input_col_device_stride);
            receiver_reader_rt_args.push_back(receiver_input_shard_row_tiles);
            receiver_reader_rt_args.push_back(receiver_input_shard_col_tiles);
            receiver_reader_rt_args.push_back(receiver_out_row_start);
            receiver_reader_rt_args.push_back(receiver_out_col_start);
            receiver_reader_rt_args.push_back(out_row_tiles);
            receiver_reader_rt_args.push_back(out_col_tiles);
            receiver_reader_rt_args.push_back(pages_per_packet);
            receiver_reader_rt_args.push_back(i);  // Receiver of device at ring_index i
            desc.kernels[receiver_reader_kernel_id].emplace_runtime_args(core, receiver_reader_rt_args);

            KernelDescriptor::RTArgList receiver_writer_rt_args;
            receiver_writer_rt_args.push_back(tensor_args.persistent_output_buffer.buffer());  // binding
            receiver_writer_rt_args.push_back(in_row_tiles);
            receiver_writer_rt_args.push_back(in_col_tiles);
            receiver_writer_rt_args.push_back(receiver_input_row_device_stride);
            receiver_writer_rt_args.push_back(receiver_input_col_device_stride);
            receiver_writer_rt_args.push_back(receiver_input_shard_row_tiles);
            receiver_writer_rt_args.push_back(receiver_input_shard_col_tiles);
            receiver_writer_rt_args.push_back(receiver_out_row_start);
            receiver_writer_rt_args.push_back(receiver_out_col_start);
            receiver_writer_rt_args.push_back(out_row_tiles);
            receiver_writer_rt_args.push_back(out_col_tiles);
            receiver_writer_rt_args.push_back(pages_per_packet);
            receiver_writer_rt_args.push_back(i);  // Receiver of device at ring_index i
            desc.kernels[receiver_writer_kernel_id].emplace_runtime_args(core, receiver_writer_rt_args);
        }
    }

    return desc;
}

}  // anonymous namespace

WorkloadDescriptor AllToAllAsyncProgram::create_workload_descriptor(
    const AllToAllAsyncParams& operation_attributes,
    const AllToAllAsyncInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());

    for (const auto& coord : coords) {
        ProgramDescriptor desc =
            build_program_descriptor(operation_attributes, tensor_args, tensor_return_value, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return wd;
}

}  // namespace ttnn::experimental::prim
