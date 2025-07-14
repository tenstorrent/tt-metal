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
#include "ttnn/operations/experimental/ccl/all_gather_async/device/all_gather_async_op.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
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

#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks all_broadcast_async_multicore(
    const Tensor& input_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    std::vector<Tensor>& output_tensors,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};

    auto mesh_device = input_tensor.mesh_device();
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device->id(),
        is_first_chip,
        is_last_chip);

    bool sharded = input_tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;
    bool tilized = input_tensor.get_layout() == ttnn::TILE_LAYOUT;

    uint32_t num_width_shards = 1;
    if (!tilized && (input_tensor.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                     input_tensor.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED)) {
        num_width_shards = input_tensor.get_padded_shape()[-1] / input_tensor.memory_config().shard_spec()->shape[1];
    }

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    auto output_tensor = output_tensors[0];
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_workers_per_link, mesh_device, sub_device_id);

    // Info for RM tensors
    uint32_t row_size = input_tensor.get_logical_shape()[-1] * input_tensor.element_size();
    uint32_t page_size = round_up_to_mul32(row_size);

    uint32_t num_rows = input_tensor.get_logical_shape().size() > 2
                            ? input_tensor.get_logical_shape()[-2] * input_tensor.get_logical_shape()[-3]
                            : input_tensor.get_logical_shape()[-2];
    if (input_tensor.get_logical_shape().size() == 4) {
        num_rows *= input_tensor.get_logical_shape()[0];
    }

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tilized ? tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes() : 4096;
    size_t max_packet_size = packet_size_bytes;
    uint32_t num_packets_per_row = std::ceil(static_cast<double>(row_size) / max_packet_size);
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_num_pages * l1_scratch_cb_page_size_bytes, {{src0_cb_index, df}})
            .set_page_size(src0_cb_index, l1_scratch_cb_page_size_bytes);

    uint32_t buffer_page_size = page_size;
    if (!tilized) {
        if (num_width_shards > 1) {
            buffer_page_size = input_tensor.memory_config().shard_spec()->shape[1] * input_tensor.element_size();
        }
        cb_src0_config = tt::tt_metal::CircularBufferConfig(3 * buffer_page_size, {{src0_cb_index, df}})
                             .set_page_size(src0_cb_index, buffer_page_size);
    }
    tt::tt_metal::CBHandle cb_src0_workers = CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
    auto reserved_packet_header_CB_handle =
        CreateCircularBuffer(program, sender_worker_core_range, cb_reserved_packet_header_config);

    // Tensor Info
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        static_cast<uint32_t>(input_tensor_buffer_type),  // buffer0_type
        src0_cb_index,                                    // cb0_id
        num_pages_per_packet,                             // packet_size_in_pages
        op_config.get_page_size(),                        // tensor0_page_size
    };

    if (!tilized) {
        reader_compile_args = {
            static_cast<uint32_t>(input_tensor_buffer_type),      // buffer0_type
            src0_cb_index,                                        // cb0_id
            num_width_shards > 1 ? buffer_page_size : page_size,  // page_size
            row_size,
            num_packets_per_row,  // num_packets_per_row
            max_packet_size       // max_packet_size
        };
    }

    std::vector<uint32_t> writer_compile_args = {
        reserved_packet_header_CB_index,                   // reserved_packet_header_cb_id
        num_packet_headers_storable,                       // num_packet_headers_storable
        static_cast<uint32_t>(output_tensor_buffer_type),  // buffer0_type
        src0_cb_index,                                     // cb0_id
        num_pages_per_packet,                              // packet_size_in_pages
        op_config.get_page_size(),                         // tensor0_page_size
        num_targets_forward,                               // num_targets_forward_direction
        num_targets_backward,                              // num_targets_backward_direction
        dynamic_alternate                                  // alternate

    };

    if (!tilized) {
        writer_compile_args = {
            reserved_packet_header_CB_index,                   // reserved_packet_header_cb_id
            num_packet_headers_storable,                       // num_packet_headers_storable
            static_cast<uint32_t>(output_tensor_buffer_type),  // buffer0_type
            src0_cb_index,                                     // cb0_id
            num_width_shards > 1 ? buffer_page_size : page_size,
            row_size,
            max_packet_size,
            num_packets_per_row,   // num_packets_per_row
            num_targets_forward,   // num_targets_forward_direction
            num_targets_backward,  // num_targets_backward_direction
            dynamic_alternate,     // alternate
        };
    }
    std::map<std::string, std::string> kernel_defines;
    if (sharded) {
        kernel_defines["SHARDED"] = "1";
        shard_builder::extend_sharding_compile_time_args(input_tensor, reader_compile_args);
        shard_builder::extend_sharding_compile_time_args(input_tensor, writer_compile_args);
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
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        // Set reader runtime args
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        if (!tilized) {
            base_pages_per_worker = num_rows / num_links;
        }
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);
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
        uint32_t out_ready_sem_wait_value = (dynamic_alternate ? (ring_size + 1) : ring_size) * num_links;
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
        };
        if (sharded) {
            shard_builder::extend_sharding_run_time_args(input_tensor, writer_rt_args);
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
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id, worker_sender_writer_kernel_id, semaphore, sender_worker_cores, ring_index](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            auto semaphore = static_cast<const ttnn::AllBroadcastAsync*>(operation)->semaphore;

            log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());

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
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
