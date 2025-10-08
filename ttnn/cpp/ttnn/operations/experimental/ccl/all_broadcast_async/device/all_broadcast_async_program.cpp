// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_broadcast_async/device/all_broadcast_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"

#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"

#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <sstream>
#include <type_traits>
#include <ranges>
#include <optional>

#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks all_broadcast_async_multicore(
    const Tensor& input_tensor,
    const MeshCoordinate& sender_device_coord,
    const std::optional<MeshCoordinate>& forward_coord,
    const std::optional<MeshCoordinate>& backward_coord,
    std::vector<Tensor>& output_tensors,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const GlobalSemaphore& barrier_semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};

    auto mesh_device = input_tensor.device();
    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    bool sharded = input_tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    bool tilized = input_tensor.layout() == ttnn::TILE_LAYOUT;

    uint32_t num_width_shards = 1;
    if (!tilized && (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                     input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED)) {
        num_width_shards = input_tensor.padded_shape()[-1] / input_tensor.memory_config().shard_spec()->shape[1];
    }

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    auto [num_targets_forward, num_targets_backward] =
        ccl::get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, true);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_workers_per_link, mesh_device, sub_device_id);

    // Info for RM tensors
    uint32_t row_size = input_tensor.logical_shape()[-1] * input_tensor.element_size();
    uint32_t page_size = input_tensor.buffer()->aligned_page_size();

    uint32_t num_rows = input_tensor.logical_shape().size() > 2
                            ? input_tensor.logical_shape()[-2] * input_tensor.logical_shape()[-3]
                            : input_tensor.logical_shape()[-2];
    if (input_tensor.logical_shape().size() == 4) {
        num_rows *= input_tensor.logical_shape()[0];
    }

    // L1 Scratch CB Creation
    DataType dtype = input_tensor.dtype();
    const uint32_t fabric_max_packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t MAX_PACKET_SIZE_BYTES =
        dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;
    const size_t packet_size_bytes =
        tilized ? tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes() : MAX_PACKET_SIZE_BYTES;
    size_t max_packet_size = packet_size_bytes;
    uint32_t l1_scratch_cb_page_size_bytes = input_tensor.buffer()->aligned_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);

    uint32_t buffer_page_size = page_size;
    uint32_t num_packets_per_page =
        static_cast<uint32_t>(std::ceil(static_cast<double>(buffer_page_size) / max_packet_size));
    if (!tilized) {
        if (num_width_shards > 1) {
            buffer_page_size = input_tensor.memory_config().shard_spec()->shape[1] * input_tensor.element_size();
        }

        uint32_t num_rows_per_packet = (max_packet_size / buffer_page_size >= 2) ? 2 : 1;
        cb_src0_config =
            tt::tt_metal::CircularBufferConfig(3 * buffer_page_size * num_rows_per_packet, {{src0_cb_index, df}})
                .set_page_size(src0_cb_index, buffer_page_size);
    }
    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        src0_cb_index,                               // cb0_id
        num_pages_per_packet,                        // packet_size_in_pages
        input_tensor.buffer()->aligned_page_size(),  // tensor0_page_size
    };

    if (!tilized) {
        reader_compile_args = {
            src0_cb_index,                                      // cb0_id
            buffer_page_size,                                   // page_size
            row_size,                                           // row_size
            (max_packet_size / buffer_page_size >= 2) ? 2 : 1,  // num_rows_per_packet
            num_packets_per_page,                               // num_packets_per_page
            max_packet_size};
    }

    std::vector<uint32_t> writer_compile_args = {
        src0_cb_index,                               // cb0_id
        num_pages_per_packet,                        // packet_size_in_pages
        input_tensor.buffer()->aligned_page_size(),  // tensor0_page_size
        num_targets_forward,                         // num_targets_forward_direction
        num_targets_backward,                        // num_targets_backward_direction
    };

    if (!tilized) {
        writer_compile_args = {
            src0_cb_index,  // cb0_id
            buffer_page_size,
            row_size,
            max_packet_size,
            (max_packet_size / buffer_page_size >= 2) ? 2 : 1,  // num_rows_per_packet
            num_packets_per_page,                               // num_packets_per_row
            num_targets_forward,                                // num_targets_forward_direction
            num_targets_backward,                               // num_targets_backward_direction
        };
    }
    std::vector<uint32_t> mcast_forward_args(2, 0);
    std::vector<uint32_t> mcast_backward_args(2, 0);
    if (forward_coord.has_value()) {
        mcast_forward_args[0] = 1;
        mcast_forward_args[1] = num_targets_forward;
    }
    if (backward_coord.has_value()) {
        mcast_backward_args[0] = 1;
        mcast_backward_args[1] = num_targets_backward;
    }
    writer_compile_args.insert(writer_compile_args.end(), mcast_forward_args.begin(), mcast_forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), mcast_backward_args.begin(), mcast_backward_args.end());
    std::map<std::string, std::string> kernel_defines;
    if (sharded) {
        kernel_defines["SHARDED"] = "1";
        shard_builder::extend_sharding_compile_time_args(input_tensor, reader_compile_args);
        shard_builder::extend_sharding_compile_time_args(input_tensor, writer_compile_args);
    } else {
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
        tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(writer_compile_args);
    }
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/experimental/ccl/all_broadcast_async/device/kernels/"
                  "all_broadcast_tile_reader.cpp"
                : "ttnn/cpp/ttnn/operations/experimental/ccl/all_broadcast_async/device/kernels/"
                  "all_broadcast_rm_reader.cpp",
        sender_worker_core_range,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_args, kernel_defines));

    // Writer
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        tilized ? "ttnn/cpp/ttnn/operations/experimental/ccl/all_broadcast_async/device/kernels/"
                  "all_broadcast_tile_writer.cpp"
                : "ttnn/cpp/ttnn/operations/experimental/ccl/all_broadcast_async/device/kernels/"
                  "all_broadcast_rm_writer.cpp",
        sender_worker_core_range,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_args, kernel_defines));

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // Set reader runtime args
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        if (!tilized) {
            base_pages_per_worker = num_rows / num_links;
        }
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),        // tensor_address0
            input_tile_id_start * num_width_shards,  // tile_id_start
            input_tile_id_end * num_width_shards,    // tile_id_end
        };

        if (sharded) {
            shard_builder::extend_sharding_run_time_args(input_tensor, reader_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0);
        bool reset_global_semaphore = (link == 0);
        uint32_t out_ready_sem_wait_value = ring_size * num_links;
        uint32_t output_tile_id_start = input_tile_id_start;
        uint32_t output_tile_id_end = input_tile_id_end;
        std::vector<uint32_t> writer_rt_args = {
            output_tensors[ring_index].buffer()->address(),  // tensor_address0  //HERE
            semaphore.address(),                             // out_ready_sem_bank_addr (absolute address)
            output_tile_id_start * num_width_shards,         // tile_id_start
            output_tile_id_end * num_width_shards,           // tile_id_end
            wait_output_semaphore,                           // wait_output_semaphore
            reset_global_semaphore,                          // reset_global_semaphore
            drain_sync_core.x,                               // out_ready_sem_noc0_x
            drain_sync_core.y,                               // out_ready_sem_noc0_y
            out_ready_sem_wait_value,                        // out_ready_sem_wait_value
            barrier_semaphore.address(),                     // barrier_sem
            barrier_core.x,                                  // barrier_sem_noc0_x
            barrier_core.y                                   // barrier_sem_noc0_y
        };
        auto num_connections = (int)forward_coord.has_value() + (int)backward_coord.has_value();
        writer_rt_args.push_back(num_connections);
        if (sharded) {
            shard_builder::extend_sharding_run_time_args(input_tensor, writer_rt_args);
        }

        const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
        std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
        dst_nodes.reserve(num_connections);
        if (forward_coord.has_value()) {
            const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            dst_nodes.push_back(forward_coord_fabric_node_id);
        }
        if (backward_coord.has_value()) {
            const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            dst_nodes.push_back(backward_coord_fabric_node_id);
        }

        append_routing_plane_connection_manager_rt_args(
            sender_fabric_node_id, dst_nodes, {link}, program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         semaphore,
         barrier_semaphore,
         sender_worker_cores,
         ring_index](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];

            log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());
            log_trace(tt::LogOp, "DEBUG: barrier_semaphore: {}", barrier_semaphore.address());

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[0] = output_tensors[ring_index].buffer()->address();
                worker_writer_sender_runtime_args[1] = semaphore.address();
                worker_writer_sender_runtime_args[9] = barrier_semaphore.address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
