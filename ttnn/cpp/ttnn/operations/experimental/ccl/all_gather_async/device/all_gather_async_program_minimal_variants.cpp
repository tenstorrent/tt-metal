// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
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

void append_fabric_connection_rt_args(
    const std::optional<tt::tt_fabric::SenderWorkerAdapterSpec>& connection,
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

tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_interleaved_dim3_1_1_32_any(
    const Tensor& input_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};
    auto mesh_device = input_tensor.mesh_device();
    const bool enable_async_output_tensor = false;
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_workers_per_link, mesh_device, sub_device_id);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // tripple buffering
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
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
        "interleaved_dim3_1_1_32_any_reader.cpp",
        sender_worker_core_range,
        reader_kernel_config);

    // Writer
    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    writer_kernel_config.compile_args = {
        ring_index,                                        // my_chip_id
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
        "interleaved_dim3_1_1_32_any_writer.cpp",
        sender_worker_core_range,
        writer_kernel_config);

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
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),  // tensor_address0
            input_tile_id_start,               // tile_id_start
            input_tile_id_end,                 // tile_id_end
        };
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
            output_tile_id_start,               // tile_id_start
            output_tile_id_end,                 // tile_id_end
            wait_output_semaphore,              // wait_output_semaphore
            reset_global_semaphore,             // reset_global_semaphore
            drain_sync_core.x,                  // out_ready_sem_noc0_x
            drain_sync_core.y,                  // out_ready_sem_noc0_y
            out_ready_sem_wait_value,           // out_ready_sem_wait_value
        };
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

            auto semaphore = static_cast<const ttnn::AllGatherAsync*>(operation)->semaphore.at(0);

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

tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_interleaved_dim3_1_1_any_any(
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> empty_fused_op_signaler;
    return all_gather_async_minimal_interleaved_dim3_1_1_any_any_helper(
        program,
        input_tensor,
        intermediate_tensor,
        sender_device,
        forward_device,
        backward_device,
        output_tensor,
        dim,
        num_links,
        ring_size,
        ring_index,
        topology,
        semaphore,
        sub_device_id,
        empty_fused_op_signaler);
}

tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_minimal_interleaved_dim3_1_1_any_any_helper(
    tt::tt_metal::Program& program,
    const Tensor& input_tensor,
    Tensor& intermediate_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    const CoreCoord core_grid_offset) {
    auto mesh_device = input_tensor.mesh_device();
    const bool enable_async_output_tensor = false;
    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.device()->id(),
        is_first_chip,
        is_last_chip);

    /* All gather fusion */
    bool fuse_op = fused_op_signaler.has_value();

    // Need a seperate signaler for the sender workers, to handle the first tensor slice that is locally available
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward;
    if (fuse_op) {
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
        fused_op_signaler_backward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {intermediate_tensor, output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    TT_FATAL(
        !((topology == ccl::Topology::Linear) && fuse_op), "Fusing operations is not supported for linear topology");
    if (topology == ccl::Topology::Ring && ring_index % 2 == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    // Get worker cores
    // 1 sender (reader + writer), and 2 receivers (forward/backward, each with a reader/writer)
    uint32_t num_senders_per_link = 1;
    uint32_t num_receivers_per_link = 2;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_senders_per_link, mesh_device, sub_device_id, core_grid_offset);
    const auto [total_worker_core_range, total_worker_cores] = choose_worker_cores(
        num_links, (num_senders_per_link + num_receivers_per_link), mesh_device, sub_device_id, core_grid_offset);
    const auto receiver_worker_core_range = total_worker_core_range.subtract(sender_worker_core_range);
    const auto receiver_worker_cores = corerange_to_cores(receiver_worker_core_range, std::nullopt, true);
    std::set<CoreRange> receiver_forward_core_ranges;
    receiver_forward_core_ranges.insert(CoreRange(receiver_worker_cores[1]));
    CoreRangeSet receiver_forward_core_range_set = CoreRangeSet(receiver_forward_core_ranges);
    std::set<CoreRange> receiver_backward_core_ranges;
    receiver_backward_core_ranges.insert(CoreRange(receiver_worker_cores[0]));
    CoreRangeSet receiver_backward_core_range_set = CoreRangeSet(receiver_backward_core_ranges);
    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages = 3 * num_pages_per_packet;  // triple buffering
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_forward_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig cb_sender_forward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_forward_cb_index, df}})
            .set_page_size(sender_forward_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_sender_forward_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_sender_forward_config);
    uint32_t sender_backward_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_sender_backward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{sender_backward_cb_index, df}})
            .set_page_size(sender_backward_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_sender_backward_workers =
        CreateCircularBuffer(program, sender_worker_core_range, cb_sender_backward_config);

    // CBs for transferring data between receiver_reader and receiver_writer
    uint32_t receiver_forward_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig cb_receiver_forward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{receiver_forward_cb_index, df}})
            .set_page_size(receiver_forward_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_receiver_forward_workers =
        CreateCircularBuffer(program, receiver_forward_core_ranges, cb_receiver_forward_config);
    uint32_t receiver_backward_cb_index = tt::CB::c_in4;
    tt::tt_metal::CircularBufferConfig cb_receiver_backward_config =
        tt::tt_metal::CircularBufferConfig(
            cb_num_pages * l1_scratch_cb_page_size_bytes, {{receiver_backward_cb_index, df}})
            .set_page_size(receiver_backward_cb_index, l1_scratch_cb_page_size_bytes);
    tt::tt_metal::CBHandle cb_receiver_backward_workers =
        CreateCircularBuffer(program, receiver_backward_core_ranges, cb_receiver_backward_config);

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
    const auto input_tensor_shape = input_tensor.get_padded_shape();
    const auto output_tensor_shape = output_tensor.get_padded_shape();
    const auto intermediate_tensor_buffer_type = intermediate_tensor.buffer()->buffer_type();

    uint32_t tiles_to_write_per_packet = 2;

    // KERNEL CREATION
    // Reader
    auto sender_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    sender_reader_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(input_tensor_buffer_type),         // input_buffer_type
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        sender_forward_cb_index,                                 // cb_forward_id
        sender_backward_cb_index,                                // cb_backward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // tensor0_page_size
        num_targets_forward,                                     // num_slices_forward_direction
        num_targets_backward,                                    // num_slices_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        tiles_to_write_per_packet,                               // contig_pages_advanced
    };
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_dim3_1_1_any_any_reader.cpp",
        sender_worker_core_range,
        sender_reader_kernel_config);

    // Writer
    auto sender_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    sender_writer_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        reserved_packet_header_CB_index,                         // reserved_packet_header_cb_id
        num_packet_headers_storable,                             // num_packet_headers_storable
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        static_cast<uint32_t>(output_tensor_buffer_type),        // output_buffer_type
        sender_forward_cb_index,                                 // cb_forward_id
        sender_backward_cb_index,                                // cb_backward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // tensor0_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        dynamic_alternate,                                       // alternate
        fuse_op,                                                 // fused op
        static_cast<uint32_t>(topology),                         // topology
        tiles_to_write_per_packet,                               // contig_pages_advanced
    };
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_dim3_1_1_any_any_writer.cpp",
        sender_worker_core_range,
        sender_writer_kernel_config);

    // Forward Receiver Kernels
    // Writer
    auto forward_receiver_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    forward_receiver_writer_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        receiver_forward_cb_index,                               // cb_forward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // output_tensor_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        1,                                                       // direction
        fuse_op,                                                 // fused op
        tiles_to_write_per_packet,                               // contig_pages_advanced
    };
    auto worker_forward_receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_dim3_1_1_any_any_receiver_writer.cpp",
        receiver_forward_core_range_set,
        forward_receiver_writer_kernel_config);
    // Reader
    auto forward_receiver_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    forward_receiver_reader_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        receiver_forward_cb_index,                               // cb_forward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // output_tensor_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        1,                                                       // direction
        tiles_to_write_per_packet,                               // contig_pages_advanced
    };
    auto worker_forward_receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_dim3_1_1_any_any_receiver_reader.cpp",
        receiver_forward_core_range_set,
        forward_receiver_reader_kernel_config);

    // Backward Receiver Kernels
    // Writer
    auto backward_receiver_writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    backward_receiver_writer_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        receiver_backward_cb_index,                              // cb_backward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // output_tensor_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        0,                                                       // direction
        fuse_op,                                                 // fused op
        tiles_to_write_per_packet,                               // contig_pages_advanced
    };
    auto worker_backward_receiver_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_dim3_1_1_any_any_receiver_writer.cpp",
        receiver_backward_core_range_set,
        backward_receiver_writer_kernel_config);
    // Reader
    auto backward_receiver_reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    backward_receiver_reader_kernel_config.compile_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(intermediate_tensor_buffer_type),  // intermediate_buffer_type
        receiver_backward_cb_index,                              // cb_forward_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // output_tensor_page_size
        num_targets_forward,                                     // num_targets_forward_direction
        num_targets_backward,                                    // num_targets_backward_direction
        static_cast<uint32_t>(topology),                         // topology
        0,                                                       // direction
        tiles_to_write_per_packet,                               // contig_pages_advanced
    };
    auto worker_backward_receiver_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "interleaved_dim3_1_1_any_any_receiver_reader.cpp",
        receiver_backward_core_range_set,
        backward_receiver_reader_kernel_config);

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }

        /* All gather fusion */
        if (fuse_op) {
            auto sender_workers = corerange_to_cores(sender_worker_core_range, std::nullopt, true);
            fused_op_signaler_forward->init_all_gather(program, mesh_device, sender_worker_core_range, sender_workers);
            fused_op_signaler_backward->init_all_gather(program, mesh_device, sender_worker_core_range, sender_workers);
            fused_op_signaler_sender_workers->init_all_gather(
                program, mesh_device, sender_worker_core_range, sender_workers);
        }

        // Set Sender Reader runtime args
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);

        TT_FATAL(!(input_tensor_shape[3] % TILE_WIDTH), "Input tensor width must be a multiple of TILE_WIDTH");
        TT_FATAL(!(output_tensor_shape[3] % TILE_WIDTH), "Output tensor width must be a multiple of TILE_WIDTH");
        uint32_t TILE_WIDTH = 32;
        uint32_t input_tensor_Wt = input_tensor_shape[3] / TILE_WIDTH;
        uint32_t output_tensor_Wt = output_tensor_shape[3] / TILE_WIDTH;

        std::set<CoreRange> receiver_forward_semaphore_core_ranges;
        receiver_forward_semaphore_core_ranges.insert(CoreRange(receiver_worker_cores[1]));
        receiver_forward_semaphore_core_ranges.insert(CoreRange(sender_worker_cores[0]));
        CoreRangeSet receiver_forward_semaphore_core_range_set = CoreRangeSet(receiver_forward_semaphore_core_ranges);
        auto sender_to_forward_receiver_semaphore_id =
            CreateSemaphore(program, receiver_forward_semaphore_core_range_set, 0);
        std::set<CoreRange> receiver_backward_semaphore_core_ranges;
        receiver_backward_semaphore_core_ranges.insert(CoreRange(receiver_worker_cores[0]));
        receiver_backward_semaphore_core_ranges.insert(CoreRange(sender_worker_cores[0]));
        CoreRangeSet receiver_backward_semaphore_core_range_set = CoreRangeSet(receiver_backward_semaphore_core_ranges);
        auto sender_to_backward_receiver_semaphore_id =
            CreateSemaphore(program, receiver_backward_semaphore_core_range_set, 0);
        std::vector<CoreCoord> receiver_worker_cores_noc;
        for (const auto& core : receiver_worker_cores) {
            receiver_worker_cores_noc.push_back(mesh_device->worker_core_from_logical_core(core));
        }

        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),          // input_tensor_address
            intermediate_tensor.buffer()->address(),   // output_tensor_address
            input_tensor_Wt,                           // width in tiles of the output shard
            input_tile_id_end,                         // slice_num_pages
            ring_size,                                 // ring_size
            semaphore.at(0).address(),                 // out_ready_semaphore_forward
            semaphore.at(1).address(),                 // out_ready_semaphore_backward
            sender_to_forward_receiver_semaphore_id,   // signal_receiver_sem_forward
            sender_to_backward_receiver_semaphore_id,  // signal_receiver_sem_forward
            receiver_worker_cores_noc.at(1).x,         // forward receiver core x
            receiver_worker_cores_noc.at(1).y,         // forward receiver core y
            receiver_worker_cores_noc.at(0).x,         // backward receiver core x
            receiver_worker_cores_noc.at(0).y,         // backward receiver core y
        };
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set Sender Writer runtime args
        uint32_t out_ready_sem_wait_value = (dynamic_alternate ? (ring_size + 1) : ring_size) * num_links;

        std::vector<uint32_t> writer_rt_args = {
            intermediate_tensor.buffer()->address(),  // intermediate_tensor_address
            output_tensor.buffer()->address(),        // output_tensor_address
            input_tensor_Wt,                          // width in tiles of the output shard
            output_tensor_Wt,                         // width in tiles of entire output
            input_tensor_num_pages,                   // slice_num_pages
            drain_sync_core.x,                        // out_ready_sem_noc0_x
            drain_sync_core.y,                        // out_ready_sem_noc0_y
            ring_size,                                // ring_size
            semaphore.at(0).address(),                // out_ready_semaphore_forward
            semaphore.at(1).address()                 // out_ready_semaphore_backward
        };
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
        if (fuse_op) {
            fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(writer_rt_args, 1, 0, 1);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_writer_kernel_id, {core}, writer_rt_args);

        // Set Receiver runtime args
        // Reader
        std::vector<uint32_t> forward_receiver_reader_rt_args = {
            intermediate_tensor.buffer()->address(),  // input_tensor_address
            input_tile_id_end,                        // slice_num_pages
            ring_size,                                // ring_size
            sender_to_forward_receiver_semaphore_id,  // signal_receiver_sem_forward
        };
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_forward_receiver_reader_kernel_id,
            {receiver_worker_cores[1]},
            forward_receiver_reader_rt_args);
        std::vector<uint32_t> backward_receiver_reader_rt_args = {
            intermediate_tensor.buffer()->address(),   // input_tensor_address
            input_tile_id_end,                         // slice_num_pages
            ring_size,                                 // ring_size
            sender_to_backward_receiver_semaphore_id,  // signal_receiver_sem_backward
        };
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_backward_receiver_reader_kernel_id,
            {receiver_worker_cores[0]},
            backward_receiver_reader_rt_args);

        // Writer
        std::vector<uint32_t> forward_receiver_writer_rt_args = {
            output_tensor.buffer()->address(),  // output_tensor_address
            input_tensor_Wt,                    // width in tiles of the output shard
            output_tensor_Wt,                   // width in tiles of entire output
            input_tile_id_end,                  // slice_num_pages
            ring_size,                          // ring_size
        };
        if (fuse_op) {
            fused_op_signaler_forward->push_all_gather_fused_op_rt_args(forward_receiver_writer_rt_args, 1, 0, 1);
        }
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_forward_receiver_writer_kernel_id,
            {receiver_worker_cores[1]},
            forward_receiver_writer_rt_args);
        std::vector<uint32_t> backward_receiver_writer_rt_args = {
            output_tensor.buffer()->address(),  // output_tensor_address
            input_tensor_Wt,                    // width in tiles of the output shard
            output_tensor_Wt,                   // width in tiles of entire output
            input_tile_id_end,                  // slice_num_pages
            ring_size,                          // ring_size
        };
        if (fuse_op) {
            fused_op_signaler_backward->push_all_gather_fused_op_rt_args(backward_receiver_writer_rt_args, 1, 0, 0);
        }
        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_backward_receiver_writer_kernel_id,
            {receiver_worker_cores[0]},
            backward_receiver_writer_rt_args);
    }

    auto override_runtime_arguments_callback = [worker_sender_reader_kernel_id,
                                                worker_sender_writer_kernel_id,
                                                worker_forward_receiver_reader_kernel_id,
                                                worker_forward_receiver_writer_kernel_id,
                                                worker_backward_receiver_reader_kernel_id,
                                                worker_backward_receiver_writer_kernel_id,
                                                sender_worker_cores,
                                                receiver_worker_cores](
                                                   const void* operation,
                                                   Program& program,
                                                   const std::vector<Tensor>& input_tensors,
                                                   const std::vector<std::optional<const Tensor>>&
                                                       optional_input_tensors,
                                                   const std::vector<Tensor>& output_tensors) {
        const auto& input = input_tensors[0];
        const auto& output = output_tensors[1];
        const auto& intermed = output_tensors[0];

        // update senders
        auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
        // update receivers
        auto& worker_forward_receiver_reader_runtime_args_by_core =
            GetRuntimeArgs(program, worker_forward_receiver_reader_kernel_id);
        auto& worker_forward_receiver_writer_runtime_args_by_core =
            GetRuntimeArgs(program, worker_forward_receiver_writer_kernel_id);
        auto& worker_backward_receiver_reader_runtime_args_by_core =
            GetRuntimeArgs(program, worker_backward_receiver_reader_kernel_id);
        auto& worker_backward_receiver_writer_runtime_args_by_core =
            GetRuntimeArgs(program, worker_backward_receiver_writer_kernel_id);
        for (const auto& core : sender_worker_cores) {
            // sender reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input.buffer()->address();
            worker_reader_sender_runtime_args[1] = intermed.buffer()->address();
            // sender writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = intermed.buffer()->address();
            worker_writer_sender_runtime_args[1] = output.buffer()->address();
        }

        const auto& forward_receiver_core = receiver_worker_cores[1];
        // forward receiver reader
        auto& worker_forward_receiver_reader_runtime_args =
            worker_forward_receiver_reader_runtime_args_by_core[forward_receiver_core.x][forward_receiver_core.y];
        worker_forward_receiver_reader_runtime_args[0] = intermed.buffer()->address();
        // forward receiver writer
        auto& worker_forward_receiver_writer_runtime_args =
            worker_forward_receiver_writer_runtime_args_by_core[forward_receiver_core.x][forward_receiver_core.y];
        worker_forward_receiver_writer_runtime_args[0] = output.buffer()->address();
        const auto& backward_receiver_core = receiver_worker_cores[0];
        // backward receiver reader
        auto& worker_backward_receiver_reader_runtime_args =
            worker_backward_receiver_reader_runtime_args_by_core[backward_receiver_core.x][backward_receiver_core.y];
        worker_forward_receiver_reader_runtime_args[0] = intermed.buffer()->address();
        // backward receiver writer
        auto& worker_backward_receiver_writer_runtime_args =
            worker_backward_receiver_writer_runtime_args_by_core[backward_receiver_core.x][backward_receiver_core.y];
        worker_backward_receiver_writer_runtime_args[0] = output.buffer()->address();
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

tt::tt_metal::operation::ProgramWithCallbacks all_gather_async_llama_sharded(
    const Tensor& input_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    Tensor& output_tensor,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id) {
    tt::tt_metal::Program program{};

    IDevice* mesh_device = input_tensor.mesh_device();
    if (!mesh_device) {
        mesh_device = input_tensor.device();
    }

    const bool enable_async_output_tensor = false;

    bool is_first_chip = ring_index == 0;
    bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        choose_worker_cores(num_links, num_workers_per_link, mesh_device, sub_device_id);

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto output_tensor_cores = output_tensor.memory_config().shard_spec()->grid;
    const auto output_tensor_shard_shape = output_tensor.memory_config().shard_spec()->shape;
    const auto output_tensor_shard_num_pages = output_tensor_shard_shape[0] * output_tensor_shard_shape[1] / TILE_HW;

    log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    log_debug(tt::LogOp, "output_tensor_cores: {}", output_tensor_cores);
    log_debug(tt::LogOp, "output_tensor_shard_shape: {}", output_tensor_shard_shape);
    log_debug(tt::LogOp, "output_tensor_shard_num_pages: {}", output_tensor_shard_num_pages);

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages =
        input_tensor_num_pages / num_links +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
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

    // KERNEL CREATION
    // Reader
    auto reader_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    reader_kernel_config.compile_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    for (const auto& arg : reader_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "llama_shapes_sharded_reader.cpp",
        sender_worker_core_range,
        reader_kernel_config);

    // Writer
    auto writer_kernel_config = tt::tt_metal::WriterDataMovementConfig{};
    writer_kernel_config.compile_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        dynamic_alternate                 // dynamic_alternate
    };
    log_trace(tt::LogOp, "Writer Compile Args:");
    for (const auto& arg : writer_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "llama_shapes_sharded_writer.cpp",
        sender_worker_core_range,
        writer_kernel_config);

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);
    auto cores_per_device = output_cores_vec.size() + ring_size - 1 / ring_size;
    uint32_t start_core_index_for_device = output_cores_vec.size() / ring_size * ring_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;
    TT_FATAL(
        output_cores_vec.size() % ring_size == 0 || output_cores_vec.size() == 1,
        "output sharded cores ( {} ) must be divisible by num_links ( {} ) or 1 for this work distribution scheme",
        output_cores_vec.size(),
        ring_size);
    auto output_cores_this_device = std::vector<CoreCoord>(
        output_cores_vec.begin() + start_core_index_for_device, output_cores_vec.begin() + end_core_index_for_device);
    log_trace(tt::LogOp, "output_cores_this_device: {}", output_cores_this_device);
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];

        // construct input and output core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);

        uint32_t worker_num_tiles_to_read = input_tile_id_end - input_tile_id_start;
        uint32_t input_first_core_tile_start_offset = input_tile_id_start % input_tensor_shard_num_pages;
        uint32_t output_first_core_tile_start_offset =
            (input_tensor_num_pages * ring_index + input_tile_id_start) % output_tensor_shard_num_pages;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> output_tensor_cores_x;
        std::vector<uint32_t> output_tensor_cores_y;
        for (uint32_t i = input_tile_id_start / input_tensor_shard_num_pages;
             i < (input_tile_id_end + input_tensor_shard_num_pages - 1) / input_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = input_tile_id_start / output_tensor_shard_num_pages;
             i < (input_tile_id_end + output_tensor_shard_num_pages - 1) / output_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(output_cores_this_device[i]);
            output_tensor_cores_x.push_back(this_core.x);
            output_tensor_cores_y.push_back(this_core.y);
        }

        log_debug(tt::LogOp, "input_tile_id_start: {}", input_tile_id_start);
        log_debug(tt::LogOp, "input_tile_id_end: {}", input_tile_id_end);
        log_debug(tt::LogOp, "worker_num_tiles_to_read: {}", worker_num_tiles_to_read);
        log_debug(tt::LogOp, "input_first_core_tile_start_offset: {}", input_first_core_tile_start_offset);
        log_debug(tt::LogOp, "output_first_core_tile_start_offset: {}", output_first_core_tile_start_offset);
        log_debug(tt::LogOp, "input_tensor_cores_x: {}", input_tensor_cores_x);
        log_debug(tt::LogOp, "input_tensor_cores_y: {}", input_tensor_cores_y);
        log_debug(tt::LogOp, "output_tensor_cores_x: {}", output_tensor_cores_x);
        log_debug(tt::LogOp, "output_tensor_cores_y: {}", output_tensor_cores_y);

        if (link == 0) {
            // drain sync core is the first worker core
            drain_sync_core = mesh_device->worker_core_from_logical_core(core);
        }
        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),    // tensor_address0
            input_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for (const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = (dynamic_alternate ? (ring_size + 1) : ring_size) * num_links;
        std::vector<uint32_t> writer_rt_args = {
            output_tensor.buffer()->address(),    // tensor_address0
            semaphore.address(),                  // out_ready_sem_bank_addr (absolute address)
            output_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            wait_output_semaphore,                // wait_output_semaphore
            reset_global_semaphore,               // reset_global_semaphore
            drain_sync_core.x,                    // out_ready_sem_noc0_x
            drain_sync_core.y,                    // out_ready_sem_noc0_y
            out_ready_sem_wait_value,             // out_ready_sem_wait_value
        };
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());
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

            auto semaphore = static_cast<const ttnn::AllGatherAsync*>(operation)->semaphore.at(0);

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
