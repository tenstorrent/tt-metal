// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_llama_sharded_program_factory.hpp"

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/experimental/ccl/llama_common.hpp"

namespace ttnn {

using namespace ccl;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace experimental::prim {

namespace {

// Builds the ProgramDescriptor for one coord.  ring_index, forward/backward
// neighbours, and per-link work distribution vary with the coord; the rest
// mirrors the legacy create_at body verbatim.
ProgramDescriptor build_program_descriptor(
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinate& mesh_coordinate) {
    const auto& input_tensor = tensor_args.input_tensor;

    const auto& sender_device_coord = mesh_coordinate;  // coord
    const auto& forward_coord = get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, 1, operation_attributes.topology, operation_attributes.cluster_axis);
    const auto& backward_coord = get_physical_neighbor_from_physical_coord(
        input_tensor, sender_device_coord, -1, operation_attributes.topology, operation_attributes.cluster_axis);
    TT_FATAL(forward_coord.has_value() || backward_coord.has_value(), "DEBUG: forward_coord or backward_coord is null");

    const auto& num_links = operation_attributes.num_links;
    const auto& ring_size = operation_attributes.ring_size;
    const auto& ring_index = get_linearized_index_from_physical_coord(
        input_tensor, sender_device_coord, operation_attributes.cluster_axis);  // device_index
    const auto& topology = operation_attributes.topology;
    const auto& semaphore = operation_attributes.semaphore.at(0);
    const auto& barrier_semaphore = operation_attributes.barrier_semaphore;
    bool using_persistent_buffers = operation_attributes.using_persistent_buffers;
    const auto& sub_device_id = operation_attributes.sub_device_id;
    bool use_optimal_ccl_for_llama = operation_attributes.use_optimal_ccl_for_llama;

    log_trace(tt::LogOp, "Detected all gather specialized shape. all_gather_async_llama_sharded is called");

    auto* mesh_device = input_tensor.device();
    if (!mesh_device) {
        mesh_device = input_tensor.device();
    }

    const bool enable_async_output_tensor = false;

    [[maybe_unused]] bool is_first_chip = ring_index == 0;
    [[maybe_unused]] bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device coord: {}, is_first_chip: {}, is_last_chip: {}",
        sender_device_coord,
        is_first_chip,
        is_last_chip);

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = {input_tensor};
    std::vector<Tensor> output_tensors = {output_tensor};
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [num_targets_forward, num_targets_backward] =
        get_forward_backward_line_mcast_distance(ring_size, ring_index, topology, true);
    auto [forward_args, backward_args] = get_forward_backward_line_mcast_configuration(
        sender_device_coord, forward_coord, backward_coord, num_targets_forward, num_targets_backward, mesh_device);

    // Get worker cores, assuming 1 worker per link
    uint32_t num_workers_per_link = 1;
    const auto [sender_worker_core_range, sender_worker_cores] =
        use_optimal_ccl_for_llama
            ? llama_specific::get_custom_worker_core_placement(num_links * num_workers_per_link)
            : ttnn::ccl::choose_worker_cores(num_links, num_workers_per_link, mesh_device, sub_device_id);

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

    ProgramDescriptor desc;

    // L1 Scratch CB Creation
    const size_t packet_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    uint32_t num_pages_per_packet = packet_size_bytes / l1_scratch_cb_page_size_bytes;
    uint32_t cb_num_pages =
        (input_tensor_num_pages / num_links) +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes}},
    });
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_packet_headers_storable * packet_header_size_bytes * 2,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes}},
    });

    // KERNEL CREATION
    // Reader
    std::vector<uint32_t> reader_compile_args = {
        ring_index,                 // my_chip_id
        src0_cb_index,              // cb0_id
        op_config.get_page_size(),  // tensor0_page_size
    };
    log_trace(tt::LogOp, "Reader Compile Args:");
    for ([[maybe_unused]] const auto& arg : reader_compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "llama_shapes_sharded_reader.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = sender_worker_core_range;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(reader_kernel_desc));
    const KernelHandle worker_sender_reader_kernel_id = desc.kernels.size() - 1;

    // Writer
    std::vector<uint32_t> writer_compile_args = {
        ring_index,                       // my_chip_id
        reserved_packet_header_CB_index,  // reserved_packet_header_cb_id
        num_packet_headers_storable,      // num_packet_headers_storable
        src0_cb_index,                    // cb0_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_targets_forward_direction
        num_targets_backward,             // num_targets_backward_direction
        ring_size,                        // ring_size
        barrier_semaphore.has_value() &&  // use_barrier_sem
            !using_persistent_buffers,
    };
    writer_compile_args.insert(writer_compile_args.end(), forward_args.begin(), forward_args.end());
    writer_compile_args.insert(writer_compile_args.end(), backward_args.begin(), backward_args.end());
    log_trace(tt::LogOp, "Writer Compile Args:");
    for ([[maybe_unused]] const auto& arg : writer_compile_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }
    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "llama_shapes_sharded_writer.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = sender_worker_core_range;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(writer_kernel_desc));
    const KernelHandle worker_sender_writer_kernel_id = desc.kernels.size() - 1;

    // Kernel Runtime Args
    CoreCoord drain_sync_core;  // the first worker of each chip is the drain sync core, which contains the output ready
                                // semaphore
    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto output_cores_vec = corerange_to_cores(output_tensor_cores, std::nullopt, true);
    auto cores_per_device = output_cores_vec.size() + ring_size - (1 / ring_size);
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
    CoreCoord barrier_core;
    for (uint32_t link = 0; link < num_links; link++) {
        CoreCoord core = sender_worker_cores[link];
        barrier_core = mesh_device->worker_core_from_logical_core(core);

        // construct input and output core x and y
        uint32_t base_pages_per_worker = input_tensor_num_pages / num_links;
        uint32_t remainder = input_tensor_num_pages % num_links;
        uint32_t input_tile_id_start = (link * base_pages_per_worker) + std::min(link, remainder);
        uint32_t input_tile_id_end = ((link + 1) * base_pages_per_worker) + std::min(link + 1, remainder);

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
        // Reader: input_tensor is a tensor buffer → BufferBinding (patched on cache hit).
        KernelDescriptor::RTArgList reader_rt_args;
        reader_rt_args.push_back(input_tensor.buffer());  // binding
        reader_rt_args.push_back(input_tensor_shard_num_pages);
        reader_rt_args.push_back(worker_num_tiles_to_read);
        reader_rt_args.push_back(input_first_core_tile_start_offset);
        reader_rt_args.push_back(static_cast<uint32_t>(input_tensor_cores_x.size()));
        for (uint32_t v : input_tensor_cores_x) {
            reader_rt_args.push_back(v);
        }
        for (uint32_t v : input_tensor_cores_y) {
            reader_rt_args.push_back(v);
        }
        desc.kernels[worker_sender_reader_kernel_id].emplace_runtime_args(core, reader_rt_args);

        // Writer: output_tensor binding + workload-scoped semaphores (raw).
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = ring_size * num_links;
        std::vector<uint32_t> writer_tail;
        writer_tail.push_back(forward_coord.has_value());
        if (forward_coord.has_value()) {
            const auto src_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
            const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                src_fabric_node_id, dst_fabric_node_id, link, desc, core, writer_tail);
        }
        writer_tail.push_back(backward_coord.has_value());
        if (backward_coord.has_value()) {
            const auto src_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
            const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<ProgramDescriptor>(
                src_fabric_node_id, dst_fabric_node_id, link, desc, core, writer_tail);
        }

        KernelDescriptor::RTArgList writer_rt_args;
        writer_rt_args.push_back(output_tensor.buffer());  // binding
        writer_rt_args.push_back(semaphore.address());     // workload semaphore
        writer_rt_args.push_back(output_tensor_shard_num_pages);
        writer_rt_args.push_back(worker_num_tiles_to_read);
        writer_rt_args.push_back(output_first_core_tile_start_offset);
        writer_rt_args.push_back(static_cast<uint32_t>(output_tensor_cores_x.size()));
        writer_rt_args.push_back(static_cast<uint32_t>(wait_output_semaphore));
        writer_rt_args.push_back(static_cast<uint32_t>(reset_global_semaphore));
        writer_rt_args.push_back(drain_sync_core.x);
        writer_rt_args.push_back(drain_sync_core.y);
        writer_rt_args.push_back(out_ready_sem_wait_value);
        writer_rt_args.push_back(barrier_semaphore.has_value() ? barrier_semaphore.value().address() : 0u);
        writer_rt_args.push_back(barrier_core.x);
        writer_rt_args.push_back(barrier_core.y);
        for (uint32_t v : output_tensor_cores_x) {
            writer_rt_args.push_back(v);
        }
        for (uint32_t v : output_tensor_cores_y) {
            writer_rt_args.push_back(v);
        }
        writer_rt_args.append(writer_tail);
        desc.kernels[worker_sender_writer_kernel_id].emplace_runtime_args(core, writer_rt_args);
    }

    return desc;
}

}  // namespace

WorkloadDescriptor LlamaShardedMeshWorkloadFactory::create_workload_descriptor(
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
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

}  // namespace experimental::prim

}  // namespace ttnn
