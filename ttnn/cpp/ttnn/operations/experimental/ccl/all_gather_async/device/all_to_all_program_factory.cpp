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

using namespace tt::constants;

namespace ttnn {

using namespace ccl;

tt::tt_metal::operation::ProgramWithCallbacks all_to_all_async_minimal(
    const Tensor& input_tensor,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
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
    std::vector<Tensor> output_tensors = {output_tensor};
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
    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_workers_per_link, enable_persistent_fabric_mode, device, sub_device_id);

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

    // Tensor Info
    const auto input_tensor_layout = input_tensor.buffer()->buffer_layout();
    const auto input_tensor_buffer_type = input_tensor.buffer()->buffer_type();
    const auto input_tensor_page_layout = input_tensor.layout();
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto output_tensor_layout = output_tensor.buffer()->buffer_layout();
    const auto output_tensor_buffer_type = output_tensor.buffer()->buffer_type();
    const auto output_tensor_page_layout = output_tensor.layout();

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
        dynamic_alternate                                  // alternate
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

    // Determine output shape and fracturing
    const auto input_shape = input_tensor.get_padded_shape();
    const auto in_row_tiles = input_shape[2] / tt::constants::TILE_HEIGHT;
    const auto in_col_tiles = input_shape[3] / tt::constants::TILE_WIDTH;
    auto output_shape = output_tensor.get_padded_shape();
    const auto out_row_tiles = output_shape[2] / tt::constants::TILE_HEIGHT;
    const auto out_col_tiles = output_shape[3] / tt::constants::TILE_WIDTH;
    /*
        When iterating over devices to unicast to, start with the next and iterate forwards.
        Finally, do a local copy (independent core could do this).
    */
    uint32_t receiver_ring_id_start = (ring_index + 1) % ring_size;
    uint32_t receiver_ring_id_end = (ring_index - 1) % ring_size;

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

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore

    // log_info(tt::LogOp, "num_links: {}", num_links); // 1
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = device->worker_core_from_logical_core(core);
        }

        // Set reader runtime args
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);
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
            receiver_ring_id_start,
            receiver_ring_id_end};
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for (const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = (dynamic_alternate ? (ring_size + 1) : ring_size) * num_links;
        uint32_t output_tile_id_start = ring_index * input_tensor_num_pages + input_tile_id_start;
        uint32_t output_tile_id_end = ring_index * input_tensor_num_pages + input_tile_id_end;
        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),  // tensor_address0
            semaphore.address(),                // out_ready_sem_bank_addr (absolute address)
            // output_tile_id_start,               // tile_id_start
            // output_tile_id_end,                 // tile_id_end
            out_row_tiles,
            out_col_tiles,
            out_row_start,
            out_col_start,
            input_shard_row_tiles,
            input_shard_col_tiles,
            receiver_ring_id_start,
            receiver_ring_id_end,
            wait_output_semaphore,
            reset_global_semaphore,
            drain_sync_core.x,
            drain_sync_core.y,
            out_ready_sem_wait_value,
        };
        log_trace(tt::LogOp, "Writer Runtime Args:");
        for (const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        writer_rt_args.push_back(forward_device.has_value());
        if (forward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                input_tensor.device()->id(), forward_device.value()->id(), link, program, {core}, writer_rt_args);
        }
        writer_rt_args.push_back(backward_device.has_value());
        if (backward_device.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                input_tensor.device()->id(), backward_device.value()->id(), link, program, {core}, writer_rt_args);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);
    }

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id, worker_sender_writer_kernel_id, semaphore, sender_worker_cores](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& output = output_tensors[0];

            auto semaphore = static_cast<const ttnn::AllGatherAsync*>(operation)->semaphore;

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
                worker_writer_sender_runtime_args[0] = output.buffer()->address();
                worker_writer_sender_runtime_args[1] = semaphore.address();
            }
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
