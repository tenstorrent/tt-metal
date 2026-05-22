// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/llama_all_gather_matmul_async_program_factory.hpp"

#include <algorithm>
#include <ranges>
#include <optional>
#include <variant>

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

namespace ttnn::experimental::prim {

namespace {

// Build the per-coord ProgramDescriptor for the fused llama AllGather + Matmul op.
// Mirrors the legacy create_at() body 1:1 but appends kernels/CBs/semaphores onto
// the supplied ProgramDescriptor instead of constructing a Program in place.
// The MatmulFusedOpSignaler bridge between the AllGather half (this builder) and
// the Matmul half (matmul_multi_core_agmm_fusion_helper_descriptor) is the same
// as the legacy path: the matmul builder reads back the four receiver
// semaphores that init_fused_op() populated here.
tt::tt_metal::ProgramDescriptor build_descriptor_at(
    const LlamaAllGatherMatmulAsyncParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const LlamaAllGatherMatmulAsyncInputs& tensor_args,
    LlamaAllGatherMatmulAsyncResult& tensor_return_value) {
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ProgramDescriptor;

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

    const auto& mesh_view = mesh_device->get_view();
    std::vector<IDevice*> devices_to_use = {};
    std::vector<tt::tt_fabric::FabricNodeId> fabric_node_ids;
    if (args.cluster_axis.has_value()) {
        devices_to_use = (args.cluster_axis.value() == 0) ? mesh_view.get_devices_on_column(mesh_coordinate[1])
                                                          : mesh_view.get_devices_on_row(mesh_coordinate[0]);
        fabric_node_ids = (args.cluster_axis.value() == 0) ? mesh_view.get_fabric_node_ids_on_column(mesh_coordinate[1])
                                                           : mesh_view.get_fabric_node_ids_on_row(mesh_coordinate[0]);
    } else {
        devices_to_use = args.devices;
        fabric_node_ids.reserve(devices_to_use.size());
        for (auto* device : devices_to_use) {
            auto coord = mesh_view.find_device(device->id());
            fabric_node_ids.push_back(mesh_device->get_fabric_node_id(coord));
        }
    }

    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;

    uint32_t device_index = 0;  // Initialize device index
    for (uint32_t i = 0; i < args.ring_size; ++i) {
        if (devices_to_use.at(i) == sender_device) {
            device_index = i;
            if (i != 0) {
                backward_fabric_node_id = fabric_node_ids.at(i - 1);
            } else if (args.topology == ttnn::ccl::Topology::Ring) {
                backward_fabric_node_id = fabric_node_ids.at(args.ring_size - 1);
            }
            if (i != args.ring_size - 1) {
                forward_fabric_node_id = fabric_node_ids.at(i + 1);
            } else if (args.topology == ttnn::ccl::Topology::Ring) {
                forward_fabric_node_id = fabric_node_ids.at(0);
            }
        }
    }

    uint32_t ring_index = device_index;
    ProgramDescriptor desc;

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

    // ProgramDescriptor overload of init_fused_op: appends SemaphoreDescriptors onto
    // desc.semaphores instead of calling CreateSemaphore(program, ...). The four
    // fused_op_receiver_signal_semaphores it allocates are read back below in the
    // matmul builder's compile-time args.
    matmul_fused_op_signaler->init_fused_op(
        desc,
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
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = cb_num_pages * l1_scratch_cb_page_size_bytes;
        cb_desc.core_ranges = sender_worker_core_range;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes});
        desc.cbs.push_back(std::move(cb_desc));
    }

    uint32_t inter_cb_index = tt::CB::c_in2;
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = intermediate_tensor_shard_num_pages * intermediate_tensor_page_size;
        cb_desc.core_ranges = intermediate_tensor_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(inter_cb_index),
            .data_format = df,
            .page_size = intermediate_tensor_page_size});
        cb_desc.buffer = intermediate_tensor.buffer();
        desc.cbs.push_back(std::move(cb_desc));
    }

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_CB_index = tt::CB::c_in1;
    static constexpr auto num_packet_headers_storable = 8;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = num_packet_headers_storable * packet_header_size_bytes * 2;
        cb_desc.core_ranges = sender_worker_core_range;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes});
        desc.cbs.push_back(std::move(cb_desc));
    }

    // KERNEL CREATION
    // Reader
    KernelDescriptor worker_sender_reader_desc;
    worker_sender_reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/"
        "worker_reader.cpp";
    worker_sender_reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    worker_sender_reader_desc.core_ranges = sender_worker_core_range;
    worker_sender_reader_desc.compile_time_args = {
        ring_index,                            // my_chip_id
        static_cast<uint32_t>(src0_cb_index),  // cb0_id
        op_config.get_page_size(),             // tensor0_page_size
    };
    worker_sender_reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};
    log_trace(tt::LogOp, "Reader Compile Args:");
    for ([[maybe_unused]] const auto& arg : worker_sender_reader_desc.compile_time_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }

    // Writer
    KernelDescriptor worker_sender_writer_desc;
    worker_sender_writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/"
        "worker_writer.cpp";
    worker_sender_writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    worker_sender_writer_desc.core_ranges = sender_worker_core_range;
    worker_sender_writer_desc.compile_time_args = {
        ring_index,                                              // my_chip_id
        static_cast<uint32_t>(reserved_packet_header_CB_index),  // reserved_packet_header_cb_id
        static_cast<uint32_t>(num_packet_headers_storable),      // num_packet_headers_storable
        static_cast<uint32_t>(src0_cb_index),                    // cb0_id
        num_pages_per_packet,                                    // packet_size_in_pages
        op_config.get_page_size(),                               // tensor0_page_size
        static_cast<uint32_t>(num_targets_forward),              // num_targets_forward_direction
        static_cast<uint32_t>(num_targets_backward),             // num_targets_backward_direction
        static_cast<uint32_t>(dynamic_alternate)                 // dynamic_alternate
    };
    worker_sender_writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};
    log_trace(tt::LogOp, "Writer Compile Args:");
    for ([[maybe_unused]] const auto& arg : worker_sender_writer_desc.compile_time_args) {
        log_trace(tt::LogOp, "\t{}", arg);
    }

    // Receiver
    KernelDescriptor worker_receiver_desc;
    worker_receiver_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/llama_all_gather_matmul_async/device/kernels/"
        "worker_receiver.cpp";
    worker_receiver_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    worker_receiver_desc.core_ranges = intermediate_tensor_cores;
    worker_receiver_desc.compile_time_args = {
        args.num_links,                                                    // sem_wait_val
        static_cast<uint32_t>(inter_cb_index),                             // intermediate cb index
        op_config.get_page_size(),                                         // tensor0_page_size
        args.ring_size,                                                    // ring_size
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores[0],  // semaphore id to notify to start mcast
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores[1],  // semaphore id to notify to start mcast
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores[2],  // semaphore id to notify to start mcast
        matmul_fused_op_signaler->fused_op_receiver_signal_semaphores[3],  // semaphore id to notify to start mcast
    };
    worker_receiver_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    // Kernel Runtime Args

    auto input_cores_vec = corerange_to_cores(input_tensor_cores, std::nullopt, true);
    auto intermediate_cores_vec = corerange_to_cores(intermediate_tensor_cores, std::nullopt, true);
    auto cores_per_device = (intermediate_cores_vec.size() + args.ring_size - 1) / args.ring_size;

    // Set runtime args for each core. The legacy code first SetRuntimeArgs on the whole
    // intermediate_tensor_cores CoreRangeSet with a placeholder row, then immediately overrides
    // it per-core in the loop below. The descriptor model only needs the final per-core values,
    // which the loop below emits for every core in intermediate_cores_vec.
    for (uint32_t i = 0; i < intermediate_cores_vec.size(); i++) {
        uint32_t mm_core_offset = (ring_index + args.ring_size - i) % args.ring_size;
        uint32_t next_core_to_left = (i - 1 + args.ring_size) % args.ring_size;
        uint32_t next_core_to_right = (i + 1 + args.ring_size) % args.ring_size;

        // aggregated_tensor.buffer() bound as Buffer* so the cache-hit fast path patches its
        // base address on every dispatch (legacy override_runtime_arguments[3] = aggregated->address()).
        std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> receiver_rt_args = {
            static_cast<uint32_t>(args.semaphore.address()),
            i,
            ring_index,
            aggregated_tensor.buffer(),
            static_cast<uint32_t>(bbox_physical_start_core.x),
            static_cast<uint32_t>(bbox_physical_start_core.y),
            static_cast<uint32_t>(bbox_physical_end_core.x),
            static_cast<uint32_t>(bbox_physical_end_core.y),
            static_cast<uint32_t>(bbox.size()),
            static_cast<uint32_t>(intermediate_tensor_shard_num_pages),
            mm_core_offset,
            next_core_to_left,
            next_core_to_right};
        worker_receiver_desc.emplace_runtime_args(intermediate_cores_vec[i], receiver_rt_args);
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

        // Set reader runtime args. input0 / intermediate buffers bound as Buffer* so the fast
        // cache-hit path patches their addresses on each dispatch.
        std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> reader_rt_args = {
            input0.buffer(),                                      // input tensor_address0
            intermediate_tensor.buffer(),                         // output tensor_address0
            static_cast<uint32_t>(input_tensor_shard_num_pages),  // num_tiles_per_core
            worker_num_tiles_to_read,                             // num_tiles_to_read
            input_first_core_tile_start_offset,                   // first_core_tile_start_offset
            intermediate_first_core_tile_start_offset,            // intermediate_first_core_tile_start_offset
            static_cast<uint32_t>(input_tensor_cores_x.size()),   // num_cores it reads from
            ring_index,                                           // ring_index
            static_cast<uint32_t>(args.semaphore.address()),      // out_ready_sem_bank_addr (absolute address)
            static_cast<uint32_t>(drain_sync_core.x),             // out_ready_sem_noc0_x
            static_cast<uint32_t>(drain_sync_core.y),             // out_ready_sem_noc0_y
        };
        for (uint32_t v : input_tensor_cores_x) {
            reader_rt_args.emplace_back(v);
        }
        for (uint32_t v : input_tensor_cores_y) {
            reader_rt_args.emplace_back(v);
        }
        reader_rt_args.emplace_back(static_cast<uint32_t>(intermediate_tensor_cores_x.size()));
        for (uint32_t v : intermediate_tensor_cores_x) {
            reader_rt_args.emplace_back(v);
        }
        for (uint32_t v : intermediate_tensor_cores_y) {
            reader_rt_args.emplace_back(v);
        }
        log_trace(tt::LogOp, "Reader Runtime Args: <{} entries>", reader_rt_args.size());
        worker_sender_reader_desc.emplace_runtime_args(core, reader_rt_args);

        // Set writer runtime args. intermediate_tensor.buffer() bound as Buffer*.
        std::vector<uint32_t> writer_rt_args_plain = {
            // placeholder for intermediate_tensor.buffer()->address()  -- filled in via variant below
            0u,
            static_cast<uint32_t>(args.semaphore.address()),             // out_ready_sem_bank_addr (absolute address)
            static_cast<uint32_t>(intermediate_tensor_shard_num_pages),  // num_tiles_per_core
            worker_num_tiles_to_read,                                    // num_tiles_to_read
            intermediate_first_core_tile_start_offset,                   // first_core_tile_start_offset
            static_cast<uint32_t>(intermediate_tensor_cores_x.size()),   // num_cores it writes to
            static_cast<uint32_t>(drain_sync_core.x),                    // out_ready_sem_noc0_x
            static_cast<uint32_t>(drain_sync_core.y),                    // out_ready_sem_noc0_y
        };
        writer_rt_args_plain.insert(
            writer_rt_args_plain.end(), intermediate_tensor_cores_x.begin(), intermediate_tensor_cores_x.end());
        writer_rt_args_plain.insert(
            writer_rt_args_plain.end(), intermediate_tensor_cores_y.begin(), intermediate_tensor_cores_y.end());
        log_trace(tt::LogOp, "Writer Runtime Args: <{} entries>", writer_rt_args_plain.size());

        writer_rt_args_plain.push_back(static_cast<uint32_t>(forward_fabric_node_id.has_value()));
        if (forward_fabric_node_id.has_value()) {
            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            // append_fabric_connection_rt_args is templated; passing the descriptor here causes
            // it to register the fabric connection state on the descriptor instead of a Program.
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_fabric_node_id, forward_fabric_node_id.value(), link, desc, core, writer_rt_args_plain);
        }
        writer_rt_args_plain.push_back(static_cast<uint32_t>(backward_fabric_node_id.has_value()));
        if (backward_fabric_node_id.has_value()) {
            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
            tt::tt_fabric::append_fabric_connection_rt_args(
                sender_fabric_node_id, backward_fabric_node_id.value(), link, desc, core, writer_rt_args_plain);
        }

        // Now wrap the writer args into a variant list with intermediate_tensor.buffer() at slot 0.
        std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> writer_rt_args;
        writer_rt_args.reserve(writer_rt_args_plain.size());
        writer_rt_args.emplace_back(intermediate_tensor.buffer());
        for (size_t i = 1; i < writer_rt_args_plain.size(); ++i) {
            writer_rt_args.emplace_back(writer_rt_args_plain[i]);
        }
        worker_sender_writer_desc.emplace_runtime_args(core, writer_rt_args);
    }

    // Push the AllGather kernels first so they own the low CB indices; the matmul builder
    // uses start_cb_index = matmul_fused_op_signaler->start_cb_index (set to c_in3 in the
    // init_llama_all_gather call above) to avoid colliding with src0/inter/packet_header CBs.
    desc.kernels.push_back(std::move(worker_sender_reader_desc));
    desc.kernels.push_back(std::move(worker_sender_writer_desc));
    desc.kernels.push_back(std::move(worker_receiver_desc));

    // Call MM program factory with matmul_fused_op_signaler — descriptor variant appends its
    // kernels/CBs/semaphores onto the same desc.
    ttnn::operations::llama_matmul::matmul_multi_core_agmm_fusion_helper_descriptor(
        desc,
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

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor LlamaAllGatherMatmulAsyncProgramFactory::create_workload_descriptor(
    const LlamaAllGatherMatmulAsyncParams& operation_attributes,
    const LlamaAllGatherMatmulAsyncInputs& tensor_args,
    LlamaAllGatherMatmulAsyncResult& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    for (const auto& coord : tensor_coords.coords()) {
        auto desc = build_descriptor_at(operation_attributes, coord, tensor_args, tensor_return_value);
        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::experimental::prim
