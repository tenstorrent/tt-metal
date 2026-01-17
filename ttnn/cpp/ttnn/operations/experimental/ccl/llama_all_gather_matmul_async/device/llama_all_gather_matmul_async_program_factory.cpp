// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.hpp"

#include <algorithm>
#include <ranges>
#include <optional>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/operations/experimental/ccl/all_reduce_async/device/all_reduce_async_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_1d_mm_fusion.hpp"

using namespace tt::constants;

namespace ttnn::operations::experimental::ccl::llama_all_gather_matmul_async::program {

LlamaAllGatherMatmulAsyncProgramFactory::cached_mesh_workload_t
LlamaAllGatherMatmulAsyncProgramFactory::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_vars;

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        mesh_workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_vars.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached_program.shared_variables));
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_vars)};
}

LlamaAllGatherMatmulAsyncProgramFactory::cached_program_t LlamaAllGatherMatmulAsyncProgramFactory::create_at(
    const operation_attributes_t& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input0 = tensor_args.input0;
    const auto& input1 = tensor_args.input1;
    const auto& intermediate_tensor = tensor_args.intermediate;
    auto& output_tensor = tensor_return_value.mm;
    const auto& aggregated_tensor = tensor_return_value.aggregated;

    const auto& compute_kernel_config = args.matmul_struct.compute_kernel_config.value();
    const auto& program_config = args.matmul_struct.program_config.value();
    const auto& global_cb = args.matmul_struct.global_cb;

    auto* mesh_device = input0.device();
    IDevice* sender_device = mesh_device->get_device(mesh_coordinate);

    std::vector<IDevice*> devices_to_use = {};
    if (args.cluster_axis.has_value()) {
        // User specified the cluster-axis. Derive devices based on the current coordinate
        // and the cluster-axis.
        const auto& mesh_view = mesh_device->get_view();
        devices_to_use = (args.cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                                                          : mesh_view.get_devices_on_row(mesh_coordinate[0]);
    } else {
        devices_to_use = args.devices;
    }

    std::optional<IDevice*> forward_device = std::nullopt;
    std::optional<IDevice*> backward_device = std::nullopt;

    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < args.ring_size; ++i) {
        if (devices_to_use.at(i) == sender_device) {
            device_index = i;
            if (i != 0) {
                backward_device = devices_to_use.at(i - 1);
            } else if (args.topology == ttnn::ccl::Topology::Ring) {
                backward_device = devices_to_use.at(args.ring_size - 1);
            }
            if (i != args.ring_size - 1) {
                forward_device = devices_to_use.at(i + 1);
            } else if (args.topology == ttnn::ccl::Topology::Ring) {
                forward_device = devices_to_use.at(0);
            }
        }
    }

    uint32_t ring_index = device_index;
    tt::tt_metal::Program program{};

    // Section for fusion signaler initialization
    auto tensor_slicer =
        ttnn::ccl::InterleavedRingAllGatherTensorSlicer(input0, intermediate_tensor, args.dim, ring_index);
    const uint32_t num_transfers = args.ring_size;
    const uint32_t weight_tensor_width = input1.padded_shape()[3] / 32;

    std::optional<ttnn::experimental::ccl::MatmulFusedOpSignaler> matmul_fused_op_signaler =
        ttnn::experimental::ccl::MatmulFusedOpSignaler(
            ttnn::experimental::ccl::MatmulFusedOpSignalerType::LLAMA_ALL_GATHER);

    matmul_fused_op_signaler->init_llama_all_gather(
        num_transfers,
        args.ring_size,
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
    [[maybe_unused]] bool is_last_chip = ring_index == args.ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device->id(),
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input0};
    std::vector<Tensor> intermediate_tensors = {intermediate_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, intermediate_tensors, args.topology);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(args.ring_size, ring_index, args.topology);
    if (args.topology == ttnn::ccl::Topology::Ring) {
        num_targets_forward = args.ring_size - 1;
        num_targets_backward = 0;
    }

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;

    // Cannot have CCL workers on the same cores as the worker_receiver (for now!)
    auto sub_device_core_range_set = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX,
        args.sub_device_id.value_or(mesh_device->get_sub_device_ids().at(0)));
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
        ar_choose_worker_cores(args.num_links, num_workers_per_link, available_cores);

    // Tensor Info
    const auto input_tensor_num_pages = input0.buffer()->num_pages();
    const auto input_tensor_cores = input0.memory_config().shard_spec()->grid;
    const auto input_tensor_shard_shape = input0.memory_config().shard_spec()->shape;
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
        (input_tensor_num_pages / args.num_links) +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input0.dtype());
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
        args.num_links,                                                    // sem_wait_val
        inter_cb_index,                                                    // intermediate cb index
        op_config.get_page_size(),                                         // tensor0_page_size
        args.ring_size,                                                    // ring_size
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
        {args.semaphore.address(),  // sem_address
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
    auto cores_per_device = (intermediate_cores_vec.size() + args.ring_size - 1) / args.ring_size;

    // Set runtime args for each core
    for (uint32_t i = 0; i < intermediate_cores_vec.size(); i++) {
        uint32_t mm_core_offset = (ring_index + args.ring_size - i) % args.ring_size;
        uint32_t next_core_to_left = (i - 1 + args.ring_size) % args.ring_size;
        uint32_t next_core_to_right = (i + 1 + args.ring_size) % args.ring_size;

        tt::tt_metal::SetRuntimeArgs(
            program,
            worker_receiver_kernel_id,
            {intermediate_cores_vec[i]},
            {args.semaphore.address(),
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
    uint32_t start_core_index_for_device = intermediate_cores_vec.size() / args.ring_size * ring_index;
    uint32_t end_core_index_for_device = start_core_index_for_device + cores_per_device;

    // Since each intermediate tensor core maps to a device in the ring,
    // each device only sem incs the intermediate tensor cores that are assigned to it.
    CoreCoord drain_sync_core = mesh_device->worker_core_from_logical_core(intermediate_cores_vec[ring_index]);

    TT_FATAL(
        intermediate_cores_vec.size() % args.ring_size == 0 || intermediate_cores_vec.size() == 1,
        "intermediate sharded cores ( {} ) must be divisible by num_links ( {} ) or 1 for this work distribution "
        "scheme",
        intermediate_cores_vec.size(),
        args.ring_size);
    auto intermediate_cores_this_device = std::vector<CoreCoord>(
        intermediate_cores_vec.begin() + start_core_index_for_device,
        intermediate_cores_vec.begin() + end_core_index_for_device);
    log_trace(tt::LogOp, "intermediate_cores_this_device: {}", intermediate_cores_this_device);
    for (uint32_t link = 0; link < args.num_links; link++) {
        CoreCoord core = sender_worker_cores[link];

        // construct input and intermediate core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / args.num_links;
        uint32_t remainder = input_tensor_num_pages % args.num_links;
        uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

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
            input0.buffer()->address(),                 // input tensor_address0
            intermediate_tensor.buffer()->address(),    // output tensor_address0
            input_tensor_shard_num_pages,               // num_tiles_per_core
            worker_num_tiles_to_read,                   // num_tiles_to_read
            input_first_core_tile_start_offset,         // first_core_tile_start_offset
            intermediate_first_core_tile_start_offset,  // intermediate_first_core_tile_start_offset
            input_tensor_cores_x.size(),                // num_cores it reads from
            ring_index,                                 // ring_index
            args.semaphore.address(),                   // out_ready_sem_bank_addr (absolute address)
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
            args.semaphore.address(),                   // out_ready_sem_bank_addr (absolute address)
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
    auto matmul_shared_variables = ttnn::operations::llama_matmul::matmul_multi_core_agmm_fusion_helper(
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
        args.sub_device_id,        // sub_device_id
        matmul_fused_op_signaler->start_cb_index,
        std::nullopt);

    return cached_program_t{
        std::move(program),
        {worker_sender_reader_kernel_id,
         worker_sender_writer_kernel_id,
         worker_receiver_kernel_id,
         sender_worker_cores,
         intermediate_cores_vec,
         ring_index,
         matmul_shared_variables}};
}

void LlamaAllGatherMatmulAsyncProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& args,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const auto& input0 = tensor_args.input0;
    const auto& input1 = tensor_args.input1;
    const auto& intermediate_tensor = tensor_args.intermediate;
    auto& output_tensor = tensor_return_value.mm;
    const auto& aggregated_tensor = tensor_return_value.aggregated;

    log_trace(tt::LogOp, "DEBUG: semaphore: {}", args.semaphore.address());

    for (auto& [coordinate_range, program] : cached_workload.workload.get_programs()) {
        auto& shared_vars = cached_workload.shared_variables.at(coordinate_range);

        // update senders
        auto& worker_reader_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_reader_kernel_id);
        auto& worker_writer_sender_runtime_args_by_core =
            GetRuntimeArgs(program, shared_vars.worker_sender_writer_kernel_id);
        for (const auto& core : shared_vars.sender_worker_cores) {
            // reader
            auto& worker_reader_sender_runtime_args = worker_reader_sender_runtime_args_by_core[core.x][core.y];
            worker_reader_sender_runtime_args[0] = input0.buffer()->address();
            worker_reader_sender_runtime_args[1] = intermediate_tensor.buffer()->address();
            worker_reader_sender_runtime_args[8] = args.semaphore.address();
            // writer
            auto& worker_writer_sender_runtime_args = worker_writer_sender_runtime_args_by_core[core.x][core.y];
            worker_writer_sender_runtime_args[0] = intermediate_tensor.buffer()->address();
            worker_writer_sender_runtime_args[1] = args.semaphore.address();
        }

        // update worker receiver
        auto& worker_receiver_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.worker_receiver_kernel_id);
        for (const auto& core : shared_vars.intermediate_cores_vec) {
            auto& worker_receiver_runtime_args = worker_receiver_runtime_args_by_core[core.x][core.y];
            worker_receiver_runtime_args[0] = args.semaphore.address();
            worker_receiver_runtime_args[3] = aggregated_tensor.buffer()->address();
        }

        llama_matmul::override_agmm_fusion_program_parameters(
            shared_vars.matmul_shared_variables,
            args.matmul_struct,
            program,
            {aggregated_tensor, input1},
            {},
            {output_tensor});
    }
}

}  // namespace ttnn::operations::experimental::ccl::llama_all_gather_matmul_async::program
