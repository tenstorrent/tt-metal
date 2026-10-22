// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_async_llama_sharded_program_factory.hpp"
#include "ttnn/operations/ccl/shared_with_host/ccl_runtime_args.hpp"

#include "ttnn/operations/experimental/ccl/llama_common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace ttnn {

using namespace ccl;
using namespace tt::constants;

namespace experimental::prim {

namespace {

constexpr uint32_t kLlamaShardedReaderKernelIdx = 0;
constexpr uint32_t kLlamaShardedWriterKernelIdx = 1;

tt::tt_metal::ProgramDescriptor build_llama_sharded_program_descriptor(
    const AllGatherAsyncParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor) {
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

    tt::tt_metal::ProgramDescriptor desc;

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

    auto* input_buffer = input_tensor.buffer();
    TT_FATAL(input_buffer != nullptr, "Llama sharded all-gather input buffer must be allocated on device");
    auto* output_buffer = output_tensor.buffer();
    TT_FATAL(output_buffer != nullptr, "Llama sharded all-gather output buffer must be allocated on device");

    // Tensor Info
    const auto input_tensor_num_pages = input_buffer->num_pages();
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
        (input_tensor_num_pages / num_links) +
        1;  // We are dealing with small shapes, so assuming all pages for a worker can be fit into the CB
    uint32_t src0_cb_index = tt::CB::c_in0;
    tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });
    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = static_cast<uint32_t>(num_packet_headers_storable * packet_header_size_bytes * 2),
        .core_ranges = sender_worker_core_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = static_cast<uint32_t>(packet_header_size_bytes),
        }}},
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
    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "llama_shapes_sharded_reader.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = sender_worker_core_range;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

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
    tt::tt_metal::KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_gather_async/device/kernels/"
        "llama_shapes_sharded_writer.cpp";
    writer_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = sender_worker_core_range;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_args);
    writer_kernel_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    const uint32_t reader_kernel_idx = static_cast<uint32_t>(desc.kernels.size());
    desc.kernels.push_back(std::move(reader_kernel_desc));
    const uint32_t writer_kernel_idx = static_cast<uint32_t>(desc.kernels.size());
    desc.kernels.push_back(std::move(writer_kernel_desc));
    TT_FATAL(
        reader_kernel_idx == kLlamaShardedReaderKernelIdx && writer_kernel_idx == kLlamaShardedWriterKernelIdx,
        "Llama sharded all-gather reader/writer must be kernel indices {}/{}, got {}/{}",
        kLlamaShardedReaderKernelIdx,
        kLlamaShardedWriterKernelIdx,
        reader_kernel_idx,
        writer_kernel_idx);

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
        // Set reader runtime args
        std::vector<uint32_t> reader_rt_args = {
            input_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,            // num_tiles_to_read
            input_first_core_tile_start_offset,  // first_core_tile_start_offset
            input_tensor_cores_x.size(),         // num_cores
        };
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_x.begin(), input_tensor_cores_x.end());
        reader_rt_args.insert(reader_rt_args.end(), input_tensor_cores_y.begin(), input_tensor_cores_y.end());
        log_trace(tt::LogOp, "Reader Runtime Args:");
        for ([[maybe_unused]] const auto& arg : reader_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }
        tt::tt_metal::KernelDescriptor::RTArgList reader_rt_arg_list;
        reader_rt_arg_list.append(reader_rt_args);
        desc.kernels[kLlamaShardedReaderKernelIdx].emplace_runtime_args(core, reader_rt_arg_list);

        // Set writer runtime args
        bool wait_output_semaphore = (link == 0) && !enable_async_output_tensor;
        bool reset_global_semaphore = (link == 0) && !enable_async_output_tensor;
        uint32_t out_ready_sem_wait_value = ring_size * num_links;
        std::vector<uint32_t> writer_rt_args = {
            output_tensor_shard_num_pages,        // num_tiles_per_core
            worker_num_tiles_to_read,             // num_tiles_to_read
            output_first_core_tile_start_offset,  // first_core_tile_start_offset
            output_tensor_cores_x.size(),         // num_cores
            wait_output_semaphore,                // wait_output_semaphore
            reset_global_semaphore,               // reset_global_semaphore
            drain_sync_core.x,                    // out_ready_sem_noc0_x
            drain_sync_core.y,                    // out_ready_sem_noc0_y
            out_ready_sem_wait_value,             // out_ready_sem_wait_value
            barrier_core.x,                       // barrier_sem_noc0_x
            barrier_core.y                        // barrier_sem_noc0_y
        };
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_x.begin(), output_tensor_cores_x.end());
        writer_rt_args.insert(writer_rt_args.end(), output_tensor_cores_y.begin(), output_tensor_cores_y.end());
        log_trace(tt::LogOp, "Writer Runtime Args:");
        for ([[maybe_unused]] const auto& arg : writer_rt_args) {
            log_trace(tt::LogOp, "\t{}", arg);
        }

        writer_rt_args.push_back(forward_coord.has_value());
        if (forward_coord.has_value()) {
            const auto src_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
            const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                src_fabric_node_id, dst_fabric_node_id, link, desc, core, writer_rt_args);
        }
        writer_rt_args.push_back(backward_coord.has_value());
        if (backward_coord.has_value()) {
            const auto src_fabric_node_id = mesh_device->get_fabric_node_id(sender_device_coord);
            const auto dst_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
            tt::tt_fabric::append_fabric_connection_rt_args<tt::tt_metal::ProgramDescriptor>(
                src_fabric_node_id, dst_fabric_node_id, link, desc, core, writer_rt_args);
        }

        tt::tt_metal::KernelDescriptor::RTArgList writer_rt_arg_list;
        writer_rt_arg_list.append(writer_rt_args);
        desc.kernels[kLlamaShardedWriterKernelIdx].emplace_runtime_args(core, writer_rt_arg_list);
    }

    desc.kernels[kLlamaShardedReaderKernelIdx].emplace_common_runtime_args({input_buffer});
    // Caller-owned semaphores stay out of the program hash. override_runtime_arguments rewrites these slots.
    const uint32_t semaphore_addr = static_cast<uint32_t>(semaphore.address());
    const uint32_t barrier_addr = barrier_semaphore ? static_cast<uint32_t>(barrier_semaphore->address()) : 0u;
    desc.kernels[kLlamaShardedWriterKernelIdx].emplace_common_runtime_args(
        {output_buffer,
         semaphore_addr,  // smuggled-rta-ok: semaphore, override
         barrier_addr});  // smuggled-rta-ok: semaphore, override
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor LlamaShardedMeshWorkloadFactory::create_workload_descriptor(
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& tensor_args,
    Tensor& output_tensor,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;
    const auto coords = tensor_coords.coords();
    workload_descriptor.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = build_llama_sharded_program_descriptor(operation_attributes, coord, tensor_args, output_tensor);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload_descriptor;
}

void LlamaShardedMeshWorkloadFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const AllGatherAsyncParams& operation_attributes,
    const AllGatherAsyncInputs& /*tensor_args*/,
    Tensor& /*output_tensor*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_coordinate*/) {
    auto& writer = tt::tt_metal::GetCommonRuntimeArgs(program, kLlamaShardedWriterKernelIdx);
    const auto& barrier = operation_attributes.barrier_semaphore;
    writer[ttnn::ccl::LlamaGatherWriterCommonArgs::semaphore] =
        static_cast<uint32_t>(operation_attributes.semaphore.at(0).address());
    writer[ttnn::ccl::LlamaGatherWriterCommonArgs::barrier] =
        barrier.has_value() ? static_cast<uint32_t>(barrier->address()) : 0u;
}

}  // namespace experimental::prim

}  // namespace ttnn
