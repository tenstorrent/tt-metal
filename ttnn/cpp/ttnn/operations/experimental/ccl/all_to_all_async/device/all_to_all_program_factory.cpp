// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_to_all_async/device/all_to_all_async_op.hpp"
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
#include <tuple>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

namespace all_to_all_detail {
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

}  // namespace all_to_all_detail

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

tt::tt_metal::operation::ProgramWithCallbacks all_to_all_async_minimal(
    const Tensor& input_tensor,
    Tensor& intermediate_buffer,
    Tensor& output_buffer,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const uint32_t in_dim,
    const uint32_t out_dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};
    IDevice* device = input_tensor.device();

    // Basic configuration
    const bool enable_async_output = false;
    const bool is_first_chip = ring_index == 0;
    const bool is_last_chip = ring_index == ring_size - 1;

    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {intermediate_buffer, output_buffer};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

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
    uint32_t num_receivers_per_link = ring_size;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_senders_per_link, device, sub_device_id);

    const auto [total_workers_core_range, total_workers_cores] =
        choose_worker_cores(num_links, (num_senders_per_link + num_receivers_per_link), device, sub_device_id);

    const auto receiver_worker_core_range = total_workers_core_range.subtract(sender_worker_core_range);
    const auto receiver_worker_cores = corerange_to_cores(receiver_worker_core_range, std::nullopt, true);
    TT_FATAL(
        receiver_worker_cores.size() == num_links * num_receivers_per_link,
        "Expected {} receiver cores but got {}",
        num_links * num_receivers_per_link,
        receiver_worker_cores.size());

    // Calculate buffer parameters
    const size_t packet_size = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    ;
    const uint32_t page_size = op_config.get_page_size();
    const uint32_t pages_per_packet = packet_size / page_size;
    const uint32_t cb_pages = all_to_all_detail::TRIPLE_BUFFER_MULTIPLIER * pages_per_packet;

    // Create buffers
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto [sender_buffer, header_buffer] =
        all_to_all_detail::create_sender_buffers(program, sender_worker_core_range, cb_pages, page_size, data_format);
    auto receiver_buffer = all_to_all_detail::create_receiver_buffer(
        program, receiver_worker_core_range, pages_per_packet, page_size, data_format);

    const auto [chunk_granularity, chunk_num_tiles, num_chunks_per_shard] = all_to_all_detail::calculate_chunk_params(
        input_tensor.buffer()->num_pages() / ring_size,  // number of pages sent between each pair of devices
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

    const uint32_t receiver_num_pages = pages_per_packet * 3;  // triple buffering
    const uint32_t receiver_cb_index = tt::CB::c_in0;

    /**
     * Syncrhonization and Algorithm
     *
     * Sender writer has 2 jobs:
     * 1. Write packed payloads to DRAM, 2 tiles per packet.
     * 2. Signal to receivers that chunks have arrived.
     *
     * The intermediate DRAM buffer is large enough to store the full output tensor.
     * This lets us avoid credits.
     *
     * One problem is: how do you ensure that a sender device can write contiguous tiles to DRAM?
     *
     * There are about `output_tensor_num_pages//2` pairs of contiguous tiles. These pairs
     * need to be statically partitioned between `ring_size-1` sender devices.
     *
     * A sender will determine its relative index, subtracting 1 from ring_index if ring_index > receiver_ring_index.
     * d0           d1           d2          d3          d4            d5          d6
     * (0, 12),     (1, 13),    (2, 14),    (3, 15),    (4, 16),      (5, 17),    (6, 18)
     * (7, 19),     (8, 20),    (9, 21),    (10, 22),   (11, 23),     (24, 36),   (25, 37)
     * (26, 38),    (27, 39),   (28, 40),   (29, 41),   (30, 42),     (31, 43),   (32, 44)
     * (33, 45),    (34, 46),   (35, 47),   (48, 60),   (49, 61),     (50, 62),   (51, 63)
     *
     * Pair calculation looks like this:
     *  N_BANKS = 12
     *  global_id = relative_ring_idx + local_tile_id * N_SENDERS
     *  first_id = (global_id % N_BANKS) + 2 * N_BANKS * (global_id // N_BANKS)
     *  second_id = first_id + N_BANKS
     *
     * Synchronization logic:
     * 1. sender: write `chunk_granularity` packets to dest_idx intermediate buffer
     * 2. sender: write semaphore increment to receiver_core[remote_index]
     * 3. receiver: read `chunk_granularity` packets from intermediate buffer and write to output tensor
     * 4. op completes when all receivers have received `num_chunks` chunks.
     *
     */

    // Tensor Info
    const auto input_tensor_layout = input_tensor.buffer()->buffer_layout();
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto input_tensor_page_layout = input_tensor.layout();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto output_tensor_layout = output_buffer.buffer()->buffer_layout();
    const auto output_tensor_buffer_type = output_buffer.buffer()->buffer_type();
    const auto output_tensor_page_layout = output_buffer.layout();
    const auto output_tensor_num_pages = output_buffer.buffer()->num_pages();

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
        ring_index,                                       // my_chip_id
        ring_size,                                        // num_chips
        static_cast<uint32_t>(input_tensor_buffer_type),  // buffer0_type
        tt::CB::c_in0,                                    // cb0_id
        pages_per_packet,                                 // packet_size_in_pages
        op_config.get_page_size(),                        // tensor0_page_size
        num_targets_forward,                              // num_targets_forward_direction
        num_targets_backward                              // num_targets_backward_direction
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    for (const auto& arg : reader_kernel_config.compile_args) {
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
        ring_index,                                        // my_chip_id
        ring_size,                                         // num_chips
        tt::CB::c_in1,                                     // reserved_packet_header_cb_id
        all_to_all_detail::PACKET_HEADER_BUFFER_SIZE,      // num_packet_headers_storable
        static_cast<uint32_t>(output_tensor_buffer_type),  // buffer0_type
        tt::CB::c_in0,                                     // cb0_id
        pages_per_packet,                                  // packet_size_in_pages
        op_config.get_page_size(),                         // tensor0_page_size
        num_targets_forward,                               // num_targets_forward_direction
        num_targets_backward,                              // num_targets_backward_direction
        dynamic_alternate,                                 // alternate
        chunk_granularity,                                 // granularity of signaling to receiver
        contig_pages_advanced,                             // contig_pages_advanced
        N_DRAM_BANKS                                       // num_dram_banks
    };
    for (const auto& arg : writer_kernel_config.compile_args) {
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
        ring_size,
        pages_per_packet,
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard,
        op_config.get_page_size(),
        receiver_cb_index};

    auto receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_receiver_writer.cpp",
        receiver_worker_core_range,
        receiver_writer_kernel_config);

    auto receiver_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    receiver_reader_kernel_config.compile_args = {
        ring_index,
        ring_size,
        pages_per_packet,
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard,
        op_config.get_page_size(),
        receiver_cb_index,
        pages_per_packet,
        N_DRAM_BANKS};
    auto receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_async/device/kernels/"
        "interleaved_all_to_all_receiver_reader.cpp",
        receiver_worker_core_range,
        receiver_reader_kernel_config);

    // Determine output shape and fracturing
    const auto& input_shape = input_tensor.padded_shape();
    const auto in_row_tiles = input_shape[2] / tt::constants::TILE_HEIGHT;
    const auto in_col_tiles = input_shape[3] / tt::constants::TILE_WIDTH;
    auto output_shape = output_buffer.padded_shape();
    const auto out_row_tiles = output_shape[2] / tt::constants::TILE_HEIGHT;
    const auto out_col_tiles = output_shape[3] / tt::constants::TILE_WIDTH;

    /*
        When reading a slice of my input for a certain destination device, how do I
        offset my local row/column indices to get that slice?
        NOTE: gives a stride which can be multiplied by destination ring index.
            - input_row_device_stride
            - input_col_device_stride

        When writing a slice to a certain destination device, how do I offset
        the remote row/column indices to index into the output?
        NOTE: gives an offset which only depends on my ring index.
            - out_row_start
            - out_col_start
    */
    auto
        [input_row_device_stride,
         input_col_device_stride,
         out_row_start,
         out_col_start,
         input_shard_row_tiles,
         input_shard_col_tiles] =
            calculate_strides_and_offsets(in_row_tiles, in_col_tiles, ring_size, ring_index, in_dim);

    // Check that with this static partitioning, the final packet does not overflow the intermediate buffer
    const uint32_t packets_per_row = tt::div_up(input_shard_col_tiles, contig_pages_advanced);
    const uint32_t packets_per_device_shard = packets_per_row * input_shard_row_tiles;
    const uint32_t final_packet_id = packets_per_device_shard - 1;
    const uint32_t final_packet_global_id = (ring_size - 2) + final_packet_id * (ring_size - 1);

    const uint32_t final_packet_first_tile_id =
        (final_packet_global_id % N_DRAM_BANKS) +
        contig_pages_advanced * N_DRAM_BANKS * (final_packet_global_id / N_DRAM_BANKS);
    const uint32_t final_packet_second_tile_id = final_packet_first_tile_id + N_DRAM_BANKS;
    TT_FATAL(
        final_packet_second_tile_id < intermediate_buffer.buffer()->num_pages(),
        "Final packet would overflow intermediate buffer. This is unexpected and dangerous. "
        "final_packet_first_tile_id: {}, final_packet_second_tile_id: {}, intermediate_buffer.buffer()->num_pages(): "
        "{}",
        final_packet_first_tile_id,
        final_packet_second_tile_id,
        intermediate_buffer.buffer()->num_pages());

    // Kernel Runtime Args

    const uint32_t receiver_core_x = device->worker_core_from_logical_core(receiver_worker_cores[ring_index]).x;
    const uint32_t receiver_core_y = device->worker_core_from_logical_core(receiver_worker_cores[ring_index]).y;

    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore

    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = device->worker_core_from_logical_core(core);
        }

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // tensor_address0
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
        for (const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output;
        bool reset_global_semaphore = (link == 0) && !enable_async_output;
        std::vector<uint32_t> writer_rt_args = {
            intermediate_buffer.buffer()->address(),
            output_buffer.buffer()->address(),
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
        for (const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            const auto sender_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
            const auto forward_device_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(forward_device.value()->id());
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_fabric_node_id, forward_device_fabric_node_id, link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            const auto sender_fabric_node_id =
                tt::tt_fabric::get_fabric_node_id_from_physical_chip_id(sender_device->id());
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
                    calculate_strides_and_offsets(in_row_tiles, in_col_tiles, ring_size, i, in_dim);

            // Set receiver runtime args
            std::vector<uint32_t> receiver_reader_rt_args = {
                intermediate_buffer.buffer()->address(),
                input_tensor.buffer()->address(),
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
                output_buffer.buffer()->address(),
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

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         receiver_writer_kernel_id,
         receiver_reader_kernel_id,
         semaphore,
         sender_worker_cores,
         receiver_worker_cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& intermediate_buffer = output_tensors[0];
            const auto& output_buffer = output_tensors[1];

            auto semaphore = static_cast<const ttnn::AllToAllAsync*>(operation)->semaphore;

            log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            auto& receiver_writer_runtime_args_by_core = GetRuntimeArgs(program, receiver_writer_kernel_id);
            auto& receiver_reader_runtime_args_by_core = GetRuntimeArgs(program, receiver_reader_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermediate_buffer.buffer()->address();
                worker_writer_sender_runtime_args[1] = output_buffer.buffer()->address();
                worker_writer_sender_runtime_args[2] = semaphore.address();
            }
            // receiver
            for (uint32_t i = 0; i < receiver_worker_cores.size(); i++) {
                const auto core = receiver_worker_cores[i];
                auto& receiver_writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];
                receiver_writer_runtime_args[0] = output_buffer.buffer()->address();
                auto& receiver_reader_runtime_args = receiver_reader_runtime_args_by_core[core.x][core.y];
                receiver_reader_runtime_args[0] = intermediate_buffer.buffer()->address();
                receiver_reader_runtime_args[1] = input.buffer()->address();
                receiver_reader_runtime_args[2] = semaphore.address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
