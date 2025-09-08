// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
///
#include <algorithm>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/fabric.hpp>
#include "ttnn/tensor/tensor_impl.hpp"
#include "ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_op.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_op.hpp"
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

tt::tt_metal::operation::ProgramWithCallbacks llama_all_gather_matmul_async_sharded(
    const Tensor& input_tensor,
    const Tensor& input1,
    Tensor& output_tensor,
    const Tensor& intermediate_tensor,
    const Tensor& aggregated_tensor,
    IDevice* sender_device,
    std::optional<IDevice*> forward_device,
    std::optional<IDevice*> backward_device,
    const uint32_t dim,
    const uint32_t num_links,
    const uint32_t ring_size,
    const uint32_t ring_index,
    ccl::Topology topology,
    const GlobalSemaphore& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    DeviceComputeKernelConfig compute_kernel_config,
    const operations::matmul::MatmulProgramConfig& program_config,
    const std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer>& global_cb) {
    tt::tt_metal::Program program{};

    IDevice* mesh_device = input_tensor.device();
    if (!mesh_device) {
        mesh_device = input_tensor.device();
    }

    // Section for fusion signaler initialization
    auto tensor_slicer =
        ttnn::ccl::InterleavedRingAllGatherTensorSlicer(input_tensor, intermediate_tensor, dim, ring_index);
    const uint32_t num_transfers = ring_size;
    const uint32_t weight_tensor_width = input1.padded_shape()[3] / 32;

    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_ALL_GATHER);
    matmul_fused_op_signaler->init_llama_all_gather(
        num_transfers,
        ring_size,
        ring_index,
        tensor_slicer.num_cols,
        tensor_slicer.output_page_offset,
        tensor_slicer.num_cols *
            weight_tensor_width /* weight_output_page_offset: stride across a tensor slice in the weight_tensor */,
        tt::CB::c_in3 /* start_cb_index */
    );
    matmul_fused_op_signaler->init_fused_op(
        program,
        sender_device,
        aggregated_tensor.memory_config().shard_spec()->grid.bounding_box(),
        ttnn::experimental::ccl::FusedOpSignalerMode::SINGLE);
    // Section end for fusion signaler initialization

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> intermediate_tensors = {intermediate_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, intermediate_tensors, topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    if (topology == ccl::Topology::Ring) {
        num_targets_forward = ring_size - 1;
        num_targets_backward = 0;
    }

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;

    // Cannot have CCL workers on the same cores as the worker_receiver (for now!)
    auto sub_device_core_range_set = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0)));
    // auto bbox = sub_device_core_range_set.bounding_box();
    // CoreRangeSet bbox_crs(bbox);

    auto aggregated_tensor_cores = aggregated_tensor.memory_config().shard_spec()->grid;
    auto bbox = aggregated_tensor_cores.bounding_box();
    auto bbox_physical_start_core = mesh_device->worker_core_from_logical_core(bbox.start_coord);
    auto bbox_physical_end_core = mesh_device->worker_core_from_logical_core(bbox.end_coord);

    auto output_tensor_cores = output_tensor.memory_config().shard_spec()->grid;
    auto intermediate_tensor_cores = intermediate_tensor.memory_config().shard_spec()->grid;
    auto available_cores = sub_device_core_range_set.subtract(intermediate_tensor_cores);
    available_cores = available_cores.subtract(output_tensor_cores);

    const auto [sender_worker_core_range, sender_worker_cores] =
        ar_choose_worker_cores(num_links, num_workers_per_link, available_cores);

    // Tensor Info
    const auto input_tensor_num_pages = input_tensor.buffer()->num_pages();
    const auto input_tensor_cores = input_tensor.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input_tensor.memory_config().shard_spec()->shape;
    const auto input_tensor_shard_num_pages = input_tensor_shard_shape[0] * input_tensor_shard_shape[1] / TILE_HW;
    const auto intermediate_tensor_shard_shape = intermediate_tensor.memory_config().shard_spec()->shape;
    const auto intermediate_tensor_shard_num_pages =
        intermediate_tensor_shard_shape[0] * intermediate_tensor_shard_shape[1] / TILE_HW;
    const auto intermediate_tensor_page_size = intermediate_tensor.buffer()->page_size();

    log_debug(tt::LogOp, "input_tensor_num_pages: {}", input_tensor_num_pages);
    log_debug(tt::LogOp, "input_tensor_cores: {}", input_tensor_cores);
    log_debug(tt::LogOp, "input_tensor_shard_shape: {}", input_tensor_shard_shape);
    log_debug(tt::LogOp, "input_tensor_shard_num_pages: {}", input_tensor_shard_num_pages);
    log_debug(tt::LogOp, "intermediate_tensor_cores: {}", intermediate_tensor_cores);
    log_debug(tt::LogOp, "intermediate_tensor_shard_shape: {}", intermediate_tensor_shard_shape);
    log_debug(tt::LogOp, "intermediate_tensor_shard_num_pages: {}", intermediate_tensor_shard_num_pages);

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
    CreateCircularBuffer(program, sender_worker_core_range, cb_src0_config);

    uint32_t inter_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig cb_inter_config =
        tt::tt_metal::CircularBufferConfig(
            intermediate_tensor_shard_num_pages * intermediate_tensor_page_size, {{inter_cb_index, df}})
            .set_page_size(inter_cb_index, intermediate_tensor_page_size)
            .set_globally_allocated_address(*intermediate_tensor.buffer());
    CreateCircularBuffer(program, intermediate_tensor_cores, cb_inter_config);

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    tt::tt_metal::CircularBufferConfig cb_reserved_packet_header_config =
        tt::tt_metal::CircularBufferConfig(
            num_packet_headers_storable * packet_header_size_bytes * 2,
            {{reserved_packet_header_CB_index, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_CB_index, packet_header_size_bytes);
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
    for ([[maybe_unused]] const auto& arg : reader_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/"
        "worker_reader.cpp",
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
    for ([[maybe_unused]] const auto& arg : writer_kernel_config.compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    auto worker_sender_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/"
        "worker_writer.cpp",
        sender_worker_core_range,
        writer_kernel_config);

    // Receiver

    // uint32_t semaphore_id_to_notify_to_start_mcast = CreateSemaphore(program, intermediate_tensor_cores, 0);
    auto receiver_kernel_config = tt::tt_metal::ReaderDataMovementConfig{};
    receiver_kernel_config.compile_args = {
        num_links,                                                         // sem_wait_val
        inter_cb_index,                                                    // intermediate cb index
        op_config.get_page_size(),                                         // tensor0_page_size
        ring_size,                                                         // ring_size
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores[0],  // semaphore id to notify to start mcast
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores[1],  // semaphore id to notify to start mcast
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores[2],  // semaphore id to notify to start mcast
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores[3],  // semaphore id to notify to start mcast
    };
    auto worker_receiver_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/"
        "worker_receiver.cpp",
        intermediate_tensor_cores,
        receiver_kernel_config);
    tt::tt_metal::SetRuntimeArgs(
        program,
        worker_receiver_kernel_id,
        intermediate_tensor_cores,
        {semaphore.address(),  // sem_address
         0,           // core id, corresponds to the id of which device it expect data from, will be reset later
         ring_index,  // device id
         aggregated_tensor.buffer()->address(),
         static_cast<uint32_t>(bbox_physical_start_core.x),
         static_cast<uint32_t>(bbox_physical_start_core.y),
         static_cast<uint32_t>(bbox_physical_end_core.x),
         static_cast<uint32_t>(bbox_physical_end_core.y),
         static_cast<uint32_t>(bbox.size()),
         intermediate_tensor_shard_num_pages,
         0,    // mm_core_offset
         0,    // next_core_to_left to be notified to start mcast
         0});  // next_core_to_right to be notified to start mcast

    // Kernel Runtime Args

    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto intermediate_cores_vec = corerange_to_cores(intermediate_tensor_cores, std::nullopt, true);
    auto cores_per_device = (intermediate_cores_vec.size() + ring_size - 1) / ring_size;

    // Set runtime args for each core
    for (uint32_t i = 0; i < intermediate_cores_vec.size(); i++) {
        uint32_t mm_core_offset = (ring_index + ring_size - i) % ring_size;
        uint32_t next_core_to_left = (i - 1 + ring_size) % ring_size;
        uint32_t next_core_to_right = (i + 1 + ring_size) % ring_size;

        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_receiver_kernel_id,
            {intermediate_cores_vec[i]},
            {semaphore.address(),
             i,
             ring_index,
             aggregated_tensor.buffer()->address(),
             static_cast<uint32_t>(bbox_physical_start_core.x),
             static_cast<uint32_t>(bbox_physical_start_core.y),
             static_cast<uint32_t>(bbox_physical_end_core.x),
             static_cast<uint32_t>(bbox_physical_end_core.y),
             static_cast<uint32_t>(bbox.size()),
             intermediate_tensor_shard_num_pages,
             mm_core_offset,
             next_core_to_left,
             next_core_to_right});
    }
    uint32_t start_core_index_for_device = intermediate_cores_vec.size() / ring_size * ring_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;

    // Since each intermediate tensor core maps to a device in the ring,
    // each device only sem incs the intermediate tensor cores that are assigned to it.
    CoreCoord drain_sync_core = mesh_device->worker_core_from_logical_core(intermediate_cores_vec[ring_index]);

    TT_FATAL(
        intermediate_cores_vec.size() % ring_size == 0 || intermediate_cores_vec.size() == 1,
        "intermediate sharded cores ( {} ) must be divisible by num_links ( {} ) or 1 for this work distribution "
        "scheme",
        intermediate_cores_vec.size(),
        ring_size);
    auto intermediate_cores_this_device = std::vector<CoreCoord>(
        intermediate_cores_vec.begin() + start_core_index_for_device,
        intermediate_cores_vec.begin() + end_core_index_for_device);
    log_trace(tt::LogOp, "intermediate_cores_this_device: {}", intermediate_cores_this_device);
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];

        // construct input and intermediate core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = link * base_pages_per_worker + std::min(link, remainder);
        uint32_t input_tile_id_end = (link + 1) * base_pages_per_worker + std::min(link + 1, remainder);

        uint32_t worker_num_tiles_to_read = input_tile_id_end - input_tile_id_start;
        uint32_t input_first_core_tile_start_offset = input_tile_id_start % input_tensor_shard_num_pages;
        uint32_t intermediate_first_core_tile_start_offset =
            (input_tensor_num_pages * ring_index + input_tile_id_start) % intermediate_tensor_shard_num_pages;

        std::vector<uint32_t> input_tensor_cores_x;
        std::vector<uint32_t> input_tensor_cores_y;
        std::vector<uint32_t> intermediate_tensor_cores_x;
        std::vector<uint32_t> intermediate_tensor_cores_y;
        for (uint32_t i = input_tile_id_start / input_tensor_shard_num_pages;
             i < (input_tile_id_end + input_tensor_shard_num_pages - 1) / input_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(input_cores_vec[i]);
            input_tensor_cores_x.push_back(this_core.x);
            input_tensor_cores_y.push_back(this_core.y);
        }
        for (uint32_t i = input_tile_id_start / intermediate_tensor_shard_num_pages;
             i < (input_tile_id_end + intermediate_tensor_shard_num_pages - 1) / intermediate_tensor_shard_num_pages;
             i++) {
            auto this_core = mesh_device->worker_core_from_logical_core(intermediate_cores_this_device[i]);
            intermediate_tensor_cores_x.push_back(this_core.x);
            intermediate_tensor_cores_y.push_back(this_core.y);
        }

        log_debug(tt::LogOp, "input_tile_id_start: {}", input_tile_id_start);
        log_debug(tt::LogOp, "input_tile_id_end: {}", input_tile_id_end);
        log_debug(tt::LogOp, "worker_num_tiles_to_read: {}", worker_num_tiles_to_read);
        log_debug(tt::LogOp, "input_first_core_tile_start_offset: {}", input_first_core_tile_start_offset);
        log_debug(
            tt::LogOp, "intermediate_first_core_tile_start_offset: {}", intermediate_first_core_tile_start_offset);
        log_debug(tt::LogOp, "input_tensor_cores_x: {}", input_tensor_cores_x);
        log_debug(tt::LogOp, "input_tensor_cores_y: {}", input_tensor_cores_y);
        log_debug(tt::LogOp, "intermediate_tensor_cores_x: {}", intermediate_tensor_cores_x);
        log_debug(tt::LogOp, "intermediate_tensor_cores_y: {}", intermediate_tensor_cores_y);

        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor.buffer()->address(),           // input tensor_address0
            intermediate_tensor.buffer()->address(),    // output tensor_address0
            input_tensor_shard_num_pages,               // num_tiles_per_core
            worker_num_tiles_to_read,                   // num_tiles_to_read
            input_first_core_tile_start_offset,         // first_core_tile_start_offset
            intermediate_first_core_tile_start_offset,  // intermediate_first_core_tile_start_offset
            input_tensor_cores_x.size(),                // num_cores it reads from
            ring_index,                                 // ring_index
            semaphore.address(),                        // out_ready_sem_bank_addr (absolute address)
            drain_sync_core.x,                          // out_ready_sem_noc0_x
            drain_sync_core.y,                          // out_ready_sem_noc0_y
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        reader_rt_args.push_back(intermediate_tensor_cores_x.size());
        reader_rt_args.insert(
            reader_rt_args.end(), intermediate_tensor_cores_x.begin(), intermediate_tensor_cores_x.end());
        reader_rt_args.insert(
            reader_rt_args.end(), intermediate_tensor_cores_y.begin(), intermediate_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for ([[maybe_unused]] const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        tt::tt_metal::SetRuntimeArgs(program, worker_sender_reader_kernel_id, {core}, reader_rt_args);

        // Set writer runtime args
        std::vector<uint32_t> writer_rt_args = {
            intermediate_tensor.buffer()->address(),    // tensor_address0
            semaphore.address(),                        // out_ready_sem_bank_addr (absolute address)
            intermediate_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,                   // num_tiles_to_read
            intermediate_first_core_tile_start_offset,  // first_core_tile_start_offset
            intermediate_tensor_cores_x.size(),         // num_cores it writes to
            drain_sync_core.x,                          // out_ready_sem_noc0_x
            drain_sync_core.y,                          // out_ready_sem_noc0_y
        };
        writer_rt_args.insert(
            writer_rt_args.end(), intermediate_tensor_cores_x.begin(), intermediate_tensor_cores_x.end());
        writer_rt_args.insert(
            writer_rt_args.end(), intermediate_tensor_cores_y.begin(), intermediate_tensor_cores_y.end());
        log_trace(tt::LogOp, "Writer Runtime Args:");
        for ([[maybe_unused]] const auto& arg : writer_rt_args) {
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
    }

    // Call MM program factory with matmul_fused_op_signaler
    std::optional<tt::tt_metal::operation::ProgramWithCallbacks> matmul_program_with_callbacks =
        ttnn::operations::llama_matmul::matmul_multi_core_agmm_fusion_helper(
            program,
            aggregated_tensor,         // in0
            {input1},                  // in1
            std::nullopt,              // bias
            {output_tensor},           // out0
            false,                     // broadcast_batch
            compute_kernel_config,     // compute_kernel_config
            program_config,            // program_config
            false,                     // untilize_out
            matmul_fused_op_signaler,  // fused_op_signaler
            global_cb,                 // global_cb
            sub_device_id);            // sub_device_id

    std::optional<tt::tt_metal::operation::OverrideRuntimeArgumentsCallback<std::vector<Tensor>>>
        matmul_override_runtime_arguments_callback = matmul_program_with_callbacks->override_runtime_arguments_callback;

    auto override_runtime_arguments_callback =
        [worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         worker_receiver_kernel_id,
         sender_worker_cores,
         intermediate_cores_vec,
         ring_index,
         matmul_override_runtime_arguments_callback](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            const auto& input = input_tensors[0];
            const auto& input1 = input_tensors[1];

            const auto& mm_output = output_tensors[0];
            const auto& intermediate = input_tensors[2];
            const auto& aggregated = output_tensors[1];

            auto semaphore =
                static_cast<const ttnn::LlamaAllGatherMatmulAsync*>(operation)->all_gather_params.semaphore;

            log_trace(tt::LogOp, "DEBUG: semaphore: {}", semaphore.address());

            // update senders
            auto& worker_reader_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_reader_kernel_id);
            auto& worker_writer_sender_runtime_args_by_core = GetRuntimeArgs(program, worker_sender_writer_kernel_id);
            for (const auto& core : sender_worker_cores) {
                // reader
                auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
                worker_reader_sender_runtime_args[0] = input.buffer()->address();
                worker_reader_sender_runtime_args[1] = intermediate.buffer()->address();
                worker_reader_sender_runtime_args[8] = semaphore.address();
                // writer
                auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
                worker_writer_sender_runtime_args[0] = intermediate.buffer()->address();
                worker_writer_sender_runtime_args[1] = semaphore.address();
            }

            // update worker receiver
            auto& worker_receiver_runtime_args_by_core = GetRuntimeArgs(program, worker_receiver_kernel_id);
            for (const auto& core : intermediate_cores_vec) {
                auto& worker_receiver_runtime_args = worker_receiver_runtime_args_by_core[core.x][core.y];
                worker_receiver_runtime_args[0] = semaphore.address();
                worker_receiver_runtime_args[3] = aggregated.buffer()->address();
            }

            if (matmul_override_runtime_arguments_callback.has_value()) {
                matmul_override_runtime_arguments_callback.value()(
                    operation,
                    program,
                    {aggregated, input1}, /* all gather output tensor, weight tensor */
                    optional_input_tensors,
                    {mm_output} /* matmul output tensor */
                );
            }
        };

    return {
        .program = std::move(matmul_program_with_callbacks->program),
        .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn
