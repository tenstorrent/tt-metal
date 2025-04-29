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
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/host_api.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>
#include <tuple>

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

std::tuple<uint32_t, uint32_t, uint32_t> calculate_chunk_params(
    uint32_t num_pages_per_device_shard, uint32_t num_pages_per_packet) {
    uint32_t chunk_granularity = 4;  // Start with minimum of 4

    while (true) {
        const uint32_t chunk_num_tiles = chunk_granularity * num_pages_per_packet;
        const uint32_t num_chunks = tt::div_up(num_pages_per_device_shard, chunk_num_tiles);

        if (num_chunks < 30) {
            return {chunk_granularity, chunk_num_tiles, num_chunks};
        }
        chunk_granularity *= 2;
    }
}

tt::tt_metal::operation::ProgramWithCallbacks all_to_all_async_minimal(
    const Tensor& input_tensor,
    Tensor& persistent_intermediate_buffer,
    Tensor& persistent_output_buffer,
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
    const bool enable_async_output_tensor = false;
    const bool enable_persistent_fabric_mode = true;
    IDevice* device = input_tensor.device();
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {persistent_intermediate_buffer, persistent_output_buffer};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    log_info(
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
        choose_worker_cores(num_links, num_senders_per_link, enable_persistent_fabric_mode, device, sub_device_id);

    const auto [total_workers_core_range, total_workers_cores] = choose_worker_cores(
        num_links,
        (num_senders_per_link + num_receivers_per_link),
        enable_persistent_fabric_mode,
        device,
        sub_device_id);

    const auto receiver_worker_core_range = total_workers_core_range.subtract(sender_worker_core_range);
    const auto receiver_worker_cores = corerange_to_cores(receiver_worker_core_range, std::nullopt, true);
    TT_FATAL(
        receiver_worker_cores.size() == num_links * num_receivers_per_link,
        "Expected {} receiver cores but got {}",
        num_links * num_receivers_per_link,
        receiver_worker_cores.size());

    // log_info(tt::LogOp, "sender_worker_core_range: {}", sender_worker_core_range);
    // log_info(tt::LogOp, "sender_worker_cores: {}", sender_worker_cores);
    // log_info(tt::LogOp, "total_workers_core_range: {}", total_workers_core_range);
    // log_info(tt::LogOp, "total_workers_cores: {}", total_workers_cores);
    // log_info(tt::LogOp, "receiver_worker_core_range: {}", receiver_worker_core_range);
    // log_info(tt::LogOp, "receiver_worker_cores: {}", receiver_worker_cores);
    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_1d_fabric_config().channel_buffer_size_bytes;
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    static constexpr auto packet_header_size_bytes = sizeof(tt::tt_fabric::PacketHeader);
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    /**
     * Receiver CBs
     */
    // Calculate chunk parameters to ensure num_chunks_per_shard < 32
    const auto [chunk_granularity, chunk_num_tiles, num_chunks_per_shard] = calculate_chunk_params(
        input_tensor.buffer()->num_pages() / ring_size,  // number of pages sent between each pair of devices
        num_pages_per_packet);

    // Verify constraints
    TT_FATAL(num_chunks_per_shard < 32, "num_chunks_per_shard must be < 32, got {}", num_chunks_per_shard);
    TT_FATAL(chunk_granularity >= 4, "chunk_granularity must be >= 4, got {}", chunk_granularity);

    const uint32_t receiver_num_pages = num_pages_per_packet * 3;  // triple buffering
    const uint32_t receiver_cb_index = tt::CB::c_in0;

    tt::tt_metal::CircularBufferConfig cb_receiver_config =
        tt::tt_metal::CircularBufferConfig(receiver_num_pages * op_config.get_page_size(), {{receiver_cb_index, df}})
            .set_page_size(receiver_cb_index, op_config.get_page_size());
    auto receiver_CB_handle = CreateCircularBuffer(program, receiver_worker_core_range, cb_receiver_config);

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
    const auto output_tensor_layout = persistent_output_buffer.buffer()->buffer_layout();
    const auto output_tensor_buffer_type = persistent_output_buffer.buffer()->buffer_type();
    const auto output_tensor_page_layout = persistent_output_buffer.layout();
    const auto output_tensor_num_pages = persistent_output_buffer.buffer()->num_pages();

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
        src0_cb_index,                                    // cb0_id
        num_pages_per_packet,                             // packet_size_in_pages
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
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_all_to_all_reader.cpp",
        sender_worker_core_range,
        reader_kernel_config);

    // Writer
    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    writer_kernel_config.compile_args = {
        ring_index,                                        // my_chip_id
        ring_size,                                         // num_chips
        reserved_packet_header_CB_index,                   // reserved_packet_header_cb_id
        num_packet_headers_storable,                       // num_packet_headers_storable
        static_cast<uint32_t>(output_tensor_buffer_type),  // buffer0_type
        src0_cb_index,                                     // cb0_id
        num_pages_per_packet,                              // packet_size_in_pages
        op_config.get_page_size(),                         // tensor0_page_size
        num_targets_forward,                               // num_targets_forward_direction
        num_targets_backward,                              // num_targets_backward_direction
        dynamic_alternate,                                 // alternate
        chunk_granularity,                                 // granularity of signaling to receiver
    };
    for (const auto& arg : writer_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_all_to_all_writer.cpp",
        sender_worker_core_range,
        writer_kernel_config);

    // Create receiver kernels
    auto receiver_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    receiver_writer_kernel_config.compile_args = {
        ring_index,
        ring_size,
        num_pages_per_packet,
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard,
        op_config.get_page_size(),
        receiver_cb_index};

    auto receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_all_to_all_receiver_writer.cpp",
        receiver_worker_core_range,
        receiver_writer_kernel_config);

    auto receiver_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    receiver_reader_kernel_config.compile_args = {
        ring_index,
        ring_size,
        num_pages_per_packet,
        chunk_granularity,
        chunk_num_tiles,
        num_chunks_per_shard,
        op_config.get_page_size(),
        receiver_cb_index};
    auto receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_all_to_all_receiver_reader.cpp",
        receiver_worker_core_range,
        receiver_reader_kernel_config);

    // Determine output shape and fracturing
    const auto input_shape = input_tensor.get_padded_shape();
    const auto in_row_tiles = input_shape[2] / tt::constants::TILE_HEIGHT;
    const auto in_col_tiles = input_shape[3] / tt::constants::TILE_WIDTH;
    auto output_shape = persistent_output_buffer.get_padded_shape();
    const auto out_row_tiles = output_shape[2] / tt::constants::TILE_HEIGHT;
    const auto out_col_tiles = output_shape[3] / tt::constants::TILE_WIDTH;
    /*
        When iterating over devices to unicast to, start with the next and iterate forwards.
        Finally, do a local copy (independent core could do this).
    */

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
    auto calculate_strides_and_offsets =
        [](uint32_t in_row_tiles, uint32_t in_col_tiles, uint32_t ring_size, uint32_t ring_index, uint32_t in_dim) {
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

            return std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(
                input_row_device_stride,
                input_col_device_stride,
                out_row_start,
                out_col_start,
                input_shard_row_tiles,
                input_shard_col_tiles);
        };

    auto
        [input_row_device_stride,
         input_col_device_stride,
         out_row_start,
         out_col_start,
         input_shard_row_tiles,
         input_shard_col_tiles] =
            calculate_strides_and_offsets(in_row_tiles, in_col_tiles, ring_size, ring_index, in_dim);

    // Kernel Runtime Args

    std::vector<uint32_t> receiver_cores_x;
    std::vector<uint32_t> receiver_cores_y;
    for (uint32_t i = 0; i < ring_size; i++) {
        const auto core_coord = device->worker_core_from_logical_core(receiver_worker_cores[i]);
        receiver_cores_x.push_back(core_coord.x);
        receiver_cores_y.push_back(core_coord.y);
    }

    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore

    uint32_t global_semaphore_args_idx;
    // log_info(tt::LogOp, "num_links: {}", num_links); // 1
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
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        std::vector<uint32_t> writer_rt_args = {
            persistent_intermediate_buffer.buffer()->address(),
            persistent_output_buffer.buffer()->address(),
            out_row_tiles,
            out_col_tiles,
            out_row_start,
            out_col_start,
            input_shard_row_tiles,
            input_shard_col_tiles,
            wait_output_semaphore,
            reset_global_semaphore,
            drain_sync_core.x,
            drain_sync_core.y,
            semaphore.address(),
        };
        global_semaphore_args_idx = writer_rt_args.size() - 1;

        writer_rt_args.insert(writer_rt_args.end(), receiver_cores_x.begin(), receiver_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), receiver_cores_y.begin(), receiver_cores_y.end());

        log_info(tt::LogOp, "writer_rt_args size: {}", writer_rt_args.size());
        log_trace(tt::LogOp, "Writer Runtime Args:");
        for (const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_device->id(), forward_device.value()->id(), link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_device->id(), backward_device.value()->id(), link, program, {core}, writer_rt_args);
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
                persistent_intermediate_buffer.buffer()->address(),
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
                num_pages_per_packet,
                i  // Receiver of device at ring_index i
            };
            tt::tt_metal::SetRuntimeArgs(program, receiver_reader_kernel_id, {core}, receiver_reader_rt_args);

            std::vector<uint32_t> receiver_writer_rt_args = {
                persistent_output_buffer.buffer()->address(),
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
                num_pages_per_packet,
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
         global_semaphore_args_idx,
         receiver_worker_cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& persistent_intermediate_buffer = output_tensors[0];
            const auto& persistent_output_buffer = output_tensors[1];

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
                worker_writer_sender_runtime_args[0] = persistent_intermediate_buffer.buffer()->address();
                worker_writer_sender_runtime_args[1] = persistent_output_buffer.buffer()->address();
                worker_writer_sender_runtime_args[global_semaphore_args_idx] = semaphore.address();
            }
            // receiver
            for (uint32_t i = 0; i < receiver_worker_cores.size(); i++) {
                const auto core = receiver_worker_cores[i];
                auto& receiver_writer_runtime_args = receiver_writer_runtime_args_by_core[core.x][core.y];
                receiver_writer_runtime_args[0] = persistent_output_buffer.buffer()->address();
                auto& receiver_reader_runtime_args = receiver_reader_runtime_args_by_core[core.x][core.y];
                receiver_reader_runtime_args[0] = persistent_intermediate_buffer.buffer()->address();
                receiver_reader_runtime_args[1] = input.buffer()->address();
                receiver_reader_runtime_args[2] = semaphore.address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
