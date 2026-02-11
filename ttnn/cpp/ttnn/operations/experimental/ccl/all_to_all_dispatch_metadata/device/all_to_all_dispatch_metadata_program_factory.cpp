// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "all_to_all_dispatch_metadata_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <vector>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "cpp/ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include <tt-metalium/core_coord.hpp>
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/mesh_graph.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <limits>

namespace ttnn::operations::experimental::ccl {

namespace detail {

// Launch mux worker kernels for fabric communication
// Returns: tuple of (mux_kernel_id, mux_kernel_config, mux_neighbor_core_maps)
auto launch_mux_workers(
    const MeshDevice& mesh_device,
    const CoreRangeSet& mux_core_range_set,
    const tt::tt_fabric::FabricNodeId src_node_id,
    const std::vector<ttnn::MeshCoordinate>& neighbors,
    const uint32_t num_links,
    const uint32_t num_workers,
    tt::tt_metal::Program& program) {
    auto num_full_size_channels = num_workers;
    constexpr auto num_header_only_channels = 0;
    constexpr auto num_buffers_full_size_channels = 2;
    const size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t l1_unreserved_base_address =
        mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        0,
        buffer_size_bytes_full_size_channel,
        l1_unreserved_base_address);

    // Need num_links × neighbors.size() mux cores (one per link per direction)
    const uint32_t num_mux_cores_needed = num_links * neighbors.size();
    TT_FATAL(
        corerange_to_cores(mux_core_range_set).size() >= num_mux_cores_needed,
        "Not enough mux cores in mux_core_range_set. Need {} cores ({} links × {} neighbors), have {}",
        num_mux_cores_needed,
        num_links,
        neighbors.size(),
        corerange_to_cores(mux_core_range_set).size());
    const auto needed_mux_core_range_set =
        tt::tt_metal::select_from_corerangeset(mux_core_range_set, 0, num_mux_cores_needed - 1);
    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        needed_mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
            .compile_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
            .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});

    std::vector<std::map<ttnn::MeshCoordinate, CoreCoord>> mux_neigbor_core_maps;
    mux_neigbor_core_maps.reserve(num_links);

    const auto mux_cores = corerange_to_cores(needed_mux_core_range_set);
    auto mux_core_iter = mux_cores.begin();
    for (uint32_t link = 0; link < num_links; ++link) {
        std::map<ttnn::MeshCoordinate, CoreCoord> mux_neigbor_core_map;
        for (const auto& neighbor_coord : neighbors) {
            auto mux_logical_core = *(mux_core_iter++);
            const auto mux_virtual_core = mesh_device.worker_core_from_logical_core(mux_logical_core);

            std::vector<uint32_t> mux_rt_args = {};
            const auto dst_node_id = mesh_device.get_fabric_node_id(neighbor_coord);
            mux_rt_args = mux_kernel_config.get_fabric_mux_run_time_args(
                src_node_id, dst_node_id, link, program, {mux_logical_core});

            tt::tt_metal::SetRuntimeArgs(program, mux_kernel_id, {mux_logical_core}, mux_rt_args);
            mux_neigbor_core_map[neighbor_coord] = mux_virtual_core;
        }
        mux_neigbor_core_maps.push_back(mux_neigbor_core_map);
    }

    return std::make_tuple(mux_kernel_id, mux_kernel_config, mux_neigbor_core_maps);
}

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }

uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }

uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

uint32_t get_num_rows(const ttnn::Tensor& tensor) {
    auto logical_volume = tensor.logical_shape().volume();
    auto hidden_size = tensor.logical_shape()[-1];
    TT_FATAL(logical_volume % hidden_size == 0, "Logical volume must be divisible by hidden size");
    return logical_volume / hidden_size;
}

std::pair<std::array<uint32_t, 7>, std::array<uint32_t, 7>> get_cb_sizes(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& indices_tensor,
    const ttnn::Tensor& scores_tensor,
    const ttnn::Tensor& mapping_tensor,
    uint32_t num_links,
    [[maybe_unused]] std::optional<uint32_t> axis) {
    auto aligned_input_page_size = get_aligned_page_size(input_tensor);
    auto aligned_indices_page_size = get_aligned_page_size(indices_tensor);
    auto aligned_scores_page_size = get_aligned_page_size(scores_tensor);
    auto aligned_mapping_page_size = get_aligned_page_size(mapping_tensor);
    uint32_t tokens_per_device = get_num_rows(input_tensor);
    uint32_t tokens_per_core = tt::div_up(tokens_per_device, num_links);

    // New mapping format: [devices, experts]
    // Each page is one device's row, we only need to store the source device's page (1 page)
    constexpr uint32_t mapping_pages_in_cb = 1;

    auto mesh_view = input_tensor.device()->get_view();
    uint32_t num_devices = mesh_view.num_devices();

    constexpr uint32_t buffering_factor = 2;
    constexpr uint32_t num_packet_headers = 8;

    auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    std::array<uint32_t, 7> cb_sizes = {
        buffering_factor * aligned_input_page_size,
        tokens_per_core * aligned_indices_page_size,
        mapping_pages_in_cb * aligned_mapping_page_size,  // Only 1 page for source device's mapping
        num_devices * tokens_per_core * sizeof(uint8_t),
        tokens_per_device * (aligned_indices_page_size + aligned_scores_page_size),
        num_packet_headers * packet_header_size_bytes,
        tokens_per_core * aligned_scores_page_size,  // scores tensor CB
    };

    std::array<uint32_t, 7> cb_page_sizes = {
        aligned_input_page_size,
        aligned_indices_page_size,
        aligned_mapping_page_size,
        tokens_per_core * sizeof(uint8_t),
        aligned_indices_page_size + aligned_scores_page_size,
        packet_header_size_bytes,
        aligned_scores_page_size,  // scores tensor page size
    };

    return {cb_sizes, cb_page_sizes};
}

}  // namespace detail

AllToAllDispatchMetadataDeviceOperation::AllToAllDispatchMetadataSparse::cached_mesh_workload_t
AllToAllDispatchMetadataDeviceOperation::AllToAllDispatchMetadataSparse::create_mesh_workload(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    auto* mesh_device = tensor_args.input_tensor.device();

    // Determine if we're in persistent mode:
    // - cross_device_semaphore is provided externally
    // - all 3 output tensors are provided (persistent buffers)
    // In this mode, we skip the init_semaphore to avoid the barrier overhead
    bool skip_init_semaphore =
        operation_attributes.cross_device_semaphore.has_value() && tensor_args.optional_output_tensors.has_value();

    log_debug(
        tt::LogOp,
        "Persistent mode: {} (cross_device_semaphore={}, optional_output_tensors={})",
        skip_init_semaphore,
        operation_attributes.cross_device_semaphore.has_value(),
        tensor_args.optional_output_tensors.has_value());

    std::optional<GlobalSemaphore> init_barrier_semaphore = std::nullopt;
    GlobalSemaphore final_barrier_semaphore = skip_init_semaphore
                                                  ? operation_attributes.cross_device_semaphore.value()
                                                  : ttnn::global_semaphore::create_global_semaphore(
                                                        mesh_device, operation_attributes.worker_core_range_set, 0);

    if (!skip_init_semaphore) {
        init_barrier_semaphore =
            ttnn::global_semaphore::create_global_semaphore(mesh_device, operation_attributes.worker_core_range_set, 0);
    }

    tt::tt_metal::distributed::Synchronize(
        mesh_device, std::nullopt, {});  // interaction with subdevice needs to be investigated

    for (const auto& coord : tensor_coords.coords()) {
        auto cached_program = create_at(
            operation_attributes,
            coord,
            tensor_args,
            tensor_return_value,
            tensor_coords,
            init_barrier_semaphore,
            final_barrier_semaphore,
            skip_init_semaphore);
        workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached_program.program));
        shared_variables.emplace(coord, std::move(cached_program.shared_variables));
    }
    return cached_mesh_workload_t(std::move(workload), std::move(shared_variables));
}

ttnn::device_operation::CachedProgram<
    AllToAllDispatchMetadataDeviceOperation::AllToAllDispatchMetadataSparse::shared_variables_t>
AllToAllDispatchMetadataDeviceOperation::AllToAllDispatchMetadataSparse::create_at(
    const operation_attributes_t& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const std::optional<GlobalSemaphore>& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    bool skip_init_semaphore) {
    tt::tt_metal::Program program{};

    auto input_tensor = tensor_args.input_tensor;
    auto indices_tensor = tensor_args.expert_indices_tensor;
    auto mapping_tensor = tensor_args.expert_mapping_tensor;
    auto scores_tensor = tensor_args.expert_scores_tensor;
    const auto& output_tensor = tensor_return_value.at(0);
    const auto& metadata_tensor = tensor_return_value.at(1);    // output indices tensor
    const auto& scores_out_tensor = tensor_return_value.at(2);  // output scores tensor
    auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    // Debug: print metadata tensor allocation details
    log_debug(
        tt::LogOp,
        "Metadata tensor buffer address: {} size: {}",
        metadata_tensor.buffer()->address(),
        metadata_tensor.buffer()->size());

    // drain_sync_tilizer_core is resolved in the invoke function (all_to_all_dispatch_metadata.cpp)
    // - Either explicitly provided by the caller, OR
    // - Extracted from persistent output tensor's shard spec
    // It's guaranteed to have a value by the time we reach here (invoke function errors otherwise)
    CoreCoord drain_sync_tilizer_core = operation_attributes.drain_sync_tilizer_core.value();
    log_debug(tt::LogOp, "drain_sync_tilizer_core: ({}, {})", drain_sync_tilizer_core.x, drain_sync_tilizer_core.y);

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    uint32_t src_mesh_id = *src_fabric_node_id.mesh_id;
    uint32_t src_chip_id = (uint32_t)src_fabric_node_id.chip_id;
    uint32_t linearized_mesh_coord = ttnn::operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    log_debug(
        tt::LogOp,
        "\nCreating all to all dispatch metadata program for mesh coordinate: ({}, {}) with mesh id: {} "
        "chip id: {} linearized mesh coord: {}",
        mesh_coordinate[0],
        mesh_coordinate[1],
        src_mesh_id,
        src_chip_id,
        linearized_mesh_coord);

    const auto [neighbors, directions] =
        ttnn::operations::ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, operation_attributes.axis);

    auto input_shape = input_tensor.tensor_spec().logical_shape();
    auto indices_shape = indices_tensor.tensor_spec().logical_shape();
    auto mapping_shape = mapping_tensor.tensor_spec().logical_shape();

    uint32_t num_devices = mesh_view.num_devices();
    uint32_t dispatch_devices =
        operation_attributes.axis.has_value()
            ? operation_attributes.axis.value() == 0 ? mesh_view.num_rows() : mesh_view.num_cols()
            : mesh_view.num_devices();

    uint32_t hidden_size = input_shape[-1];
    uint32_t batch_size = input_shape[0] * dispatch_devices;

    uint32_t tokens_per_device = detail::get_num_rows(input_tensor);
    uint32_t selected_experts_k = indices_shape[-1];
    // New expert mapping format: [devices, experts]
    // mapping_shape[0] = num_devices, mapping_shape[-1] = experts
    uint32_t experts = mapping_shape[-1];

    auto input_page_size = detail::get_page_size(input_tensor);
    auto indices_page_size = detail::get_page_size(indices_tensor);
    auto scores_page_size = detail::get_page_size(scores_tensor);
    auto mapping_page_size = detail::get_page_size(mapping_tensor);
    auto output_page_size = detail::get_page_size(output_tensor);
    auto metadata_page_size = detail::get_page_size(metadata_tensor);

    auto input_pages = detail::get_num_pages(input_tensor);
    auto indices_pages = detail::get_num_pages(indices_tensor);
    auto scores_pages = detail::get_num_pages(scores_tensor);
    auto mapping_pages = detail::get_num_pages(mapping_tensor);
    auto output_pages = detail::get_num_pages(output_tensor);
    auto metadata_pages = detail::get_num_pages(metadata_tensor);

    auto input_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices_tensor.dtype());
    auto scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(scores_tensor.dtype());
    auto mapping_data_format = tt::tt_metal::datatype_to_dataformat_converter(mapping_tensor.dtype());

    // input sharded buffer
    uint32_t input_tensor_cb_id = tt::CBIndex::c_0;
    // full indices buffer
    uint32_t indices_tensor_cb_id = tt::CBIndex::c_1;
    // full mapping buffer
    uint32_t mapping_tensor_cb_id = tt::CBIndex::c_2;
    // client interface
    uint32_t packet_header_cb_id = tt::CBIndex::c_3;
    // book-keeping buffer to avoid sending the same token multiple times
    uint32_t send_preparation_buffer_id = tt::CBIndex::c_4;
    // intermediate buffer for holding metadata before writing out to the device (for FullPacket impl)
    uint32_t metadata_buffer_id = tt::CBIndex::c_5;
    // scores tensor buffer (same shape as indices)
    uint32_t scores_tensor_cb_id = tt::CBIndex::c_6;

    uint32_t aligned_input_page_size = detail::get_aligned_page_size(input_tensor);
    log_debug(
        tt::LogOp,
        "input shape: {}, input_pages: {}, input_page_size: {}, aligned_input_page_size: {}",
        input_tensor.logical_shape(),
        input_pages,
        input_page_size,
        aligned_input_page_size);

    uint32_t aligned_indices_page_size = detail::get_aligned_page_size(indices_tensor);
    log_debug(
        tt::LogOp,
        "indices shape: {}, indices_pages: {}, indices_page_size: {}, aligned_indices_page_size: {}",
        indices_tensor.logical_shape(),
        indices_pages,
        indices_page_size,
        aligned_indices_page_size);

    uint32_t aligned_scores_page_size = detail::get_aligned_page_size(scores_tensor);
    log_debug(
        tt::LogOp,
        "scores shape: {}, scores_pages: {}, scores_page_size: {}, aligned_scores_page_size: {}",
        scores_tensor.logical_shape(),
        scores_pages,
        scores_page_size,
        aligned_scores_page_size);

    uint32_t aligned_mapping_page_size = detail::get_aligned_page_size(mapping_tensor);
    log_debug(
        tt::LogOp,
        "mapping shape: {}, mapping_pages: {}, mapping_page_size: {}, aligned_mapping_page_size: {}",
        mapping_tensor.logical_shape(),
        mapping_pages,
        mapping_page_size,
        aligned_mapping_page_size);

    uint32_t aligned_output_page_size = detail::get_aligned_page_size(output_tensor);
    log_debug(
        tt::LogOp,
        "output shape: {}, output_pages: {}, output_page_size: {}, aligned_output_page_size: {}",
        output_tensor.logical_shape(),
        output_pages,
        output_page_size,
        aligned_output_page_size);

    uint32_t aligned_metadata_page_size = detail::get_aligned_page_size(metadata_tensor);
    log_debug(
        tt::LogOp,
        "metadata shape: {}, metadata_pages: {}, metadata_page_size: {}, aligned_metadata_page_size: {}",
        metadata_tensor.logical_shape(),
        metadata_pages,
        metadata_page_size,
        aligned_metadata_page_size);

    auto [cb_sizes, cb_page_sizes] = detail::get_cb_sizes(
        input_tensor, indices_tensor, scores_tensor, mapping_tensor, num_links, operation_attributes.axis);

    tt::tt_metal::CircularBufferConfig cb_input_tensor_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[0], {{input_tensor_cb_id, input_data_format}})
            .set_page_size(input_tensor_cb_id, cb_page_sizes[0]);

    tt::tt_metal::CircularBufferConfig cb_indices_tensor_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[1], {{indices_tensor_cb_id, indices_data_format}})
            .set_page_size(indices_tensor_cb_id, cb_page_sizes[1]);

    tt::tt_metal::CircularBufferConfig cb_mapping_tensor_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[2], {{mapping_tensor_cb_id, mapping_data_format}})
            .set_page_size(mapping_tensor_cb_id, cb_page_sizes[2]);

    tt::tt_metal::CircularBufferConfig cb_send_preparation_buffer_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[3], {{send_preparation_buffer_id, tt::DataFormat::UInt8}})
            .set_page_size(send_preparation_buffer_id, cb_page_sizes[3]);

    tt::tt_metal::CircularBufferConfig cb_metadata_buffer_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[4], {{metadata_buffer_id, mapping_data_format}})
            .set_page_size(metadata_buffer_id, cb_page_sizes[4]);

    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[5], {{packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(packet_header_cb_id, cb_page_sizes[5]);

    tt::tt_metal::CircularBufferConfig cb_scores_tensor_config =
        tt::tt_metal::CircularBufferConfig(cb_sizes[6], {{scores_tensor_cb_id, scores_data_format}})
            .set_page_size(scores_tensor_cb_id, cb_page_sizes[6]);

    auto worker_core_range_set = operation_attributes.worker_core_range_set;

    auto subdevice_cores = corerange_to_cores(worker_core_range_set);
    TT_FATAL(
        subdevice_cores.size() >= num_links,
        "Not enough cores {} to send all links {}",
        subdevice_cores.size(),
        num_links);

    // Determine number of workers based on worker mode:
    // - DIRECT: 1 worker per link (can't share direct fabric connections)
    // - MUX_TOKEN_SPLIT: Multiple workers per link, tokens distributed across workers
    // - MUX_PAYLOAD_SPLIT: Multiple workers per link, same tokens but payload split across workers
    const WorkerMode worker_mode = operation_attributes.worker_mode;
    const bool use_mux = (worker_mode != WorkerMode::DIRECT);
    const bool payload_split_mode = (worker_mode == WorkerMode::MUX_PAYLOAD_SPLIT);
    uint32_t num_cores;
    uint32_t workers_per_link;

    if (use_mux) {
        // Use all available worker cores, distributing them across links
        // For example: 8 cores / 4 links = 2 workers per link
        num_cores = subdevice_cores.size();
        workers_per_link = num_cores / num_links;
        TT_FATAL(
            workers_per_link >= 1,
            "Not enough cores {} for {} links (need at least 1 worker per link)",
            num_cores,
            num_links);
        // Ensure num_cores is a multiple of num_links for even distribution
        num_cores = workers_per_link * num_links;
    } else {
        // Without mux: 1 worker per link (can't share direct fabric connections)
        num_cores = num_links;
        workers_per_link = 1;
    }

    // In payload split mode: tokens distributed across links, workers on same link share tokens
    // In token split mode: tokens distributed across all workers
    // In direct mode: tokens distributed across workers (1 worker per link)
    //
    // Note: tt::div_up may distribute tokens unevenly. When tokens_per_device is not evenly divisible
    // by num_links/num_cores, the last link/worker handles fewer tokens. This is an intentional
    // tradeoff for simpler work partitioning.
    uint32_t tokens_per_core = payload_split_mode ? tt::div_up(tokens_per_device, num_links)  // tokens per link
                                                  : tt::div_up(tokens_per_device, num_cores);

    log_debug(
        tt::LogOp,
        "Worker distribution: {} total workers, {} links, {} workers per link, {} tokens per {}, payload_split={}",
        num_cores,
        num_links,
        workers_per_link,
        tokens_per_core,
        payload_split_mode ? "link" : "core",
        payload_split_mode);

    auto sender_core_grid = tt::tt_metal::num_cores_to_corerangeset_in_subcoregrids(
        subdevice_cores.at(0), num_cores, worker_core_range_set, true);
    std::vector<CoreCoord> sender_cores = corerange_to_cores(sender_core_grid);

    // create circular buffers
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_input_tensor_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_indices_tensor_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_scores_tensor_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_mapping_tensor_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, packet_header_cb_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_send_preparation_buffer_config);
    tt::tt_metal::CreateCircularBuffer(program, sender_core_grid, cb_metadata_buffer_config);

    std::vector<uint32_t> dest_mesh_id, dest_chip_id;
    for (const auto& coord : tensor_coords.coords()) {
        auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    log_debug(tt::LogOp, "dest_chip_id: {}", ttnn::operations::ccl::common::stringify(dest_chip_id));
    log_debug(tt::LogOp, "dest_mesh_id: {}", ttnn::operations::ccl::common::stringify(dest_mesh_id));
    log_debug(tt::LogOp, "directions: {}", ttnn::operations::ccl::common::stringify(directions));

    auto fabric_max_packet_size = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const auto l1_alignment = tt::tt_metal::hal::get_l1_alignment();

    // New mapping format: [devices, experts]
    // Each page is one device's view. Kernels only need to read 1 page (source device's mapping row).
    constexpr uint32_t mapping_pages_for_kernel = 1;

    std::vector<uint32_t> reader_compile_time_args = {
        input_tensor_cb_id,
        indices_tensor_cb_id,
        mapping_tensor_cb_id,
        packet_header_cb_id,
        send_preparation_buffer_id,

        input_pages,
        indices_pages,
        mapping_pages_for_kernel,  // Only 1 page - source device's mapping row
        output_pages,
        metadata_pages,

        input_page_size,
        indices_page_size,
        mapping_page_size,
        output_page_size,
        metadata_page_size,

        num_devices,
        hidden_size,
        batch_size,
        selected_experts_k,
        experts,
        tokens_per_device,

        num_links,
        (uint32_t)topology,

        src_mesh_id,
        (uint32_t)src_chip_id,
        mesh_view.num_rows(),
        mesh_view.num_cols(),

        aligned_input_page_size,
        aligned_indices_page_size,
        aligned_mapping_page_size,
        aligned_output_page_size,
        aligned_metadata_page_size,

        (uint32_t)fabric_max_packet_size,

        l1_alignment,
        metadata_buffer_id,
        0,
        linearized_mesh_coord,

        // scores tensor args
        scores_tensor_cb_id,
        scores_pages,
        scores_page_size,
        aligned_scores_page_size,
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(scores_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(mapping_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(metadata_tensor.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(scores_out_tensor.buffer()).append_to(reader_compile_time_args);

    const auto& writer_compile_time_args = reader_compile_time_args;

    std::map<std::string, std::string> reader_defines = {
        {"AXIS", std::to_string(operation_attributes.axis.has_value() ? operation_attributes.axis.value() : -1)},
    };

    tt::tt_metal::KernelHandle ternary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/device/kernels/dataflow/"
        "reader_all_to_all_dispatch_metadata.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = reader_compile_time_args,
            .defines = reader_defines});

    // Code-gen a mesh-position to fabric chip ID array for the writer kernel
    // Code-gen a mesh-position to mesh-id array for the writer kernel
    // Code-gen a direction array that is set to true when a direction has a valid connection (when a neighbor exists or
    // if it's along a valid cluster axis)
    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", ttnn::operations::ccl::common::stringify(dest_chip_id)},
        {"DEST_MESH_ID", ttnn::operations::ccl::common::stringify(dest_mesh_id)},
        {"DIRECTIONS", ttnn::operations::ccl::common::stringify(directions)},
        {"DISPATCH_ALGORITHM", std::to_string(static_cast<uint8_t>(operation_attributes.dispatch_algorithm))}};

    if (operation_attributes.axis.has_value()) {
        writer_defines["AXIS"] = std::to_string(operation_attributes.axis.value());
    }

    // Conditionally set up mux infrastructure (use_mux already defined above)
    std::optional<tt::tt_fabric::FabricMuxConfig> mux_kernel_config;
    std::vector<std::map<ttnn::MeshCoordinate, CoreCoord>> mux_neigbor_core_maps;
    [[maybe_unused]] tt::tt_metal::KernelHandle mux_kernel_id = 0;

    if (use_mux) {
        writer_defines["USE_MUX"] = "1";
        log_debug(tt::LogOp, "Using fabric mux for dispatch");

        // Validate mux core range has enough cores (need 1 mux per link)
        auto mux_cores = corerange_to_cores(operation_attributes.mux_core_range_set);
        TT_FATAL(
            mux_cores.size() >= num_links,
            "Not enough mux cores {} to support {} links (need at least 1 mux per link)",
            mux_cores.size(),
            num_links);

        // Launch mux workers
        // Note: num_workers passed to launch_mux_workers should be workers_per_link (clients per mux channel)
        auto [launched_mux_kernel_id, launched_mux_config, launched_mux_neigbor_core_maps] = detail::launch_mux_workers(
            *mesh_device,
            operation_attributes.mux_core_range_set,
            src_fabric_node_id,
            neighbors,
            num_links,
            workers_per_link,  // workers per link = clients per mux channel
            program);
        mux_kernel_id = launched_mux_kernel_id;
        mux_kernel_config = launched_mux_config;
        mux_neigbor_core_maps = std::move(launched_mux_neigbor_core_maps);
    }

    if (payload_split_mode) {
        writer_defines["PAYLOAD_SPLIT_MODE"] = "1";
        writer_defines["WORKERS_PER_LINK"] = std::to_string(workers_per_link);
        log_debug(tt::LogOp, "Using payload split mode with {} workers per link", workers_per_link);
    }

    if (skip_init_semaphore) {
        writer_defines["SKIP_INIT_SEMAPHORE"] = "1";
        log_debug(tt::LogOp, "Skipping init semaphore (persistent mode enabled)");
    }

    // Build writer compile-time args - add mux args if enabled
    std::vector<uint32_t> writer_compile_time_args_final = writer_compile_time_args;
    if (use_mux) {
        ttnn::ccl::fabric_mux_connection_ct_args(
            workers_per_link,  // num_mux_clients = workers per link sharing each mux channel
            tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
            mux_kernel_config.value(),
            writer_compile_time_args_final);
    }

    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/all_to_all_dispatch_metadata/device/kernels/dataflow/"
        "writer_all_to_all_dispatch_metadata.cpp",
        sender_core_grid,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = writer_compile_time_args_final,
            .defines = writer_defines});

    std::vector<uint32_t> reader_runtime_args = {
        input_tensor.buffer()->address(),
        indices_tensor.buffer()->address(),
        scores_tensor.buffer()->address(),
        mapping_tensor.buffer()->address(),
        output_tensor.buffer()->address(),
        metadata_tensor.buffer()->address(),
        scores_out_tensor.buffer()->address(),
        (uint32_t)cross_device_semaphore.address(),
    };

    uint32_t reader_token_start_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);
    uint32_t reader_token_end_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);

    // Pre-allocate slots for payload split args (only used in payload_split_mode)
    uint32_t reader_payload_offset_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);
    uint32_t reader_payload_size_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);
    uint32_t reader_is_primary_idx = reader_runtime_args.size();
    reader_runtime_args.push_back(0);

    // Get drain sync tilizer core NOC coordinates for direct metadata output
    // drain_sync_tilizer_core was already resolved at the start of create_at
    auto drain_sync_tilizer_noc_core = mesh_device->worker_core_from_logical_core(drain_sync_tilizer_core);

    // Debug: print address comparison for overlap detection
    log_debug(
        tt::LogOp,
        "Address comparison - metadata_tensor: {} (size: {}), cross_device_semaphore: {}, init_semaphore: {}",
        metadata_tensor.buffer()->address(),
        metadata_tensor.buffer()->size(),
        cross_device_semaphore.address(),
        init_semaphore.has_value() ? init_semaphore->address() : 0);
    log_debug(
        tt::LogOp,
        "drain_sync_tilizer_core logical: ({}, {}), NOC: ({}, {})",
        drain_sync_tilizer_core.x,
        drain_sync_tilizer_core.y,
        drain_sync_tilizer_noc_core.x,
        drain_sync_tilizer_noc_core.y);

    // For mux: set up termination masters (one per link)
    // Each link has workers_per_link workers sharing the same mux cores
    // The first worker on each link (worker_idx % workers_per_link == 0) is the termination master
    std::vector<CoreCoord> termination_master_cores;
    std::vector<CoreCoord> termination_master_virtual_cores;
    std::vector<uint32_t> termination_master_semaphore_ids;
    if (use_mux) {
        for (uint32_t link = 0; link < num_links; link++) {
            uint32_t master_worker_idx = link * workers_per_link;  // First worker on this link
            const auto& master_core = sender_cores[master_worker_idx];
            termination_master_cores.push_back(master_core);
            termination_master_virtual_cores.push_back(mesh_device->worker_core_from_logical_core(master_core));
            termination_master_semaphore_ids.push_back(tt::tt_metal::CreateSemaphore(program, {master_core}, 0));
            log_debug(
                tt::LogOp,
                "Link {} termination master: worker {} at core ({}, {}), semaphore_id {}",
                link,
                master_worker_idx,
                master_core.x,
                master_core.y,
                termination_master_semaphore_ids.back());
        }
    }

    uint32_t tokens_per_core_start = 0;
    auto mux_core_map_iter = mux_neigbor_core_maps.cbegin();
    uint32_t current_link_id = 0;

    for (uint32_t worker_idx = 0; worker_idx < num_cores; worker_idx++) {
        const auto& sender_core = sender_cores[worker_idx];
        uint32_t link_id = worker_idx / workers_per_link;
        uint32_t worker_idx_within_link = worker_idx % workers_per_link;
        bool is_termination_master = (worker_idx_within_link == 0);

        std::vector<uint32_t> writer_runtime_args = {
            input_tensor.buffer()->address(),
            indices_tensor.buffer()->address(),
            scores_tensor.buffer()->address(),
            mapping_tensor.buffer()->address(),
            output_tensor.buffer()->address(),
            metadata_tensor.buffer()->address(),
            scores_out_tensor.buffer()->address(),
            (uint32_t)cross_device_semaphore.address(),
            init_semaphore.has_value() ? (uint32_t)init_semaphore->address() : 0,  // 0 when skipping init semaphore
        };
        uint32_t writer_token_start_idx = writer_runtime_args.size();
        writer_runtime_args.push_back(0);
        uint32_t writer_token_end_idx = writer_runtime_args.size();
        writer_runtime_args.push_back(0);
        // Add drain sync tilizer core NOC coordinates for direct metadata write
        writer_runtime_args.push_back(drain_sync_tilizer_noc_core.x);
        writer_runtime_args.push_back(drain_sync_tilizer_noc_core.y);

        if (payload_split_mode) {
            // In payload split mode:
            // - Tokens are distributed across links (tokens_per_link = tokens_per_device / num_links)
            // - Workers on the same link handle the SAME tokens but split the payload
            // - Worker 0 sends first half of payload, worker 1 sends second half, etc.
            uint32_t tokens_per_link = tokens_per_device / num_links;
            uint32_t link_token_start = link_id * tokens_per_link;
            uint32_t link_token_end = std::min(link_token_start + tokens_per_link, tokens_per_device);

            // Both workers on this link process the same token range
            reader_runtime_args[reader_token_start_idx] = link_token_start;
            reader_runtime_args[reader_token_end_idx] = link_token_end;
            writer_runtime_args[writer_token_start_idx] = link_token_start;
            writer_runtime_args[writer_token_end_idx] = link_token_end;

            // Calculate payload split parameters
            // Worker 0 sends first portion, worker 1 sends second portion, etc.
            uint32_t full_payload_size = input_page_size;
            uint32_t payload_size = full_payload_size / workers_per_link;
            uint32_t payload_offset = worker_idx_within_link * payload_size;
            // Primary worker (worker 0) sends metadata + atomic_inc and is also termination master
            bool is_primary_payload_worker = (worker_idx_within_link == 0);

            // Set payload split RT args using pre-allocated indices
            // Reader needs them to read only the relevant portion of input tokens
            reader_runtime_args[reader_payload_offset_idx] = payload_offset;
            reader_runtime_args[reader_payload_size_idx] = payload_size;
            reader_runtime_args[reader_is_primary_idx] = is_primary_payload_worker ? 1 : 0;

            writer_runtime_args.push_back(payload_offset);
            writer_runtime_args.push_back(payload_size);
            writer_runtime_args.push_back(is_primary_payload_worker ? 1 : 0);

            log_debug(
                tt::LogOp,
                "Worker {} payload split: link={}, tokens=[{},{}), offset={}, size={}, is_primary={}",
                worker_idx,
                link_id,
                link_token_start,
                link_token_end,
                payload_offset,
                payload_size,
                is_primary_payload_worker);
        } else {
            // Token split mode or direct mode: distribute tokens across workers
            reader_runtime_args[reader_token_start_idx] = tokens_per_core_start;
            reader_runtime_args[reader_token_end_idx] =
                std::min(tokens_per_core_start + tokens_per_core, tokens_per_device);
            writer_runtime_args[writer_token_start_idx] = tokens_per_core_start;
            writer_runtime_args[writer_token_end_idx] = reader_runtime_args[reader_token_end_idx];
            tokens_per_core_start = reader_runtime_args[reader_token_end_idx];

            // Set defaults for payload split args (full page, all workers are "primary")
            reader_runtime_args[reader_payload_offset_idx] = 0;
            reader_runtime_args[reader_payload_size_idx] = input_page_size;
            reader_runtime_args[reader_is_primary_idx] = 1;  // All workers are primary in non-split mode

            // Writer also needs defaults
            writer_runtime_args.push_back(0);                // payload_offset
            writer_runtime_args.push_back(input_page_size);  // payload_size
            writer_runtime_args.push_back(1);                // is_primary (all workers in non-split mode)
        }

        if (use_mux) {
            // Use mux connection runtime args
            // Get termination master info for this link
            const auto& link_termination_master_virtual_core = termination_master_virtual_cores[link_id];
            uint32_t link_termination_master_semaphore_id = termination_master_semaphore_ids[link_id];

            // Move to the correct mux core map for this link (if link changed)
            while (current_link_id < link_id) {
                ++mux_core_map_iter;
                ++current_link_id;
            }

            for (const auto& neighbor_coordinate : neighbors) {
                const auto& mux_virtual_core = mux_core_map_iter->at(neighbor_coordinate);
                ttnn::ccl::fabric_mux_connection_rt_args(
                    true,  // mux_connection_valid
                    is_termination_master,
                    tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                    mux_virtual_core,
                    worker_idx_within_link,  // worker_index within the mux channel (0 to workers_per_link-1)
                    sender_core,
                    mux_kernel_config.value(),
                    program,
                    link_termination_master_virtual_core,
                    writer_runtime_args,
                    link_termination_master_semaphore_id);
            }

            log_debug(
                tt::LogOp,
                "Worker {} at core ({}, {}) on link {} (worker {} within link), is_termination_master={}",
                worker_idx,
                sender_core.x,
                sender_core.y,
                link_id,
                worker_idx_within_link,
                is_termination_master);
        } else {
            // Use direct fabric connection runtime args
            for (const auto& neighbor_coordinate : neighbors) {
                log_debug(
                    tt::LogOp,
                    "Connection between mesh coord ({}, {}) and ({}, {}) at core {} will choose link_id: {} and "
                    "handles "
                    "token indices from {} to {}",
                    mesh_coordinate[0],
                    mesh_coordinate[1],
                    neighbor_coordinate[0],
                    neighbor_coordinate[1],
                    sender_core,
                    link_id,
                    reader_runtime_args[8],
                    reader_runtime_args[9]);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    src_fabric_node_id,
                    mesh_device->get_fabric_node_id(neighbor_coordinate),
                    link_id,
                    program,
                    sender_core,
                    writer_runtime_args);
            }
        }

        tt::tt_metal::SetRuntimeArgs(program, ternary_reader_kernel_id, sender_core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, binary_writer_kernel_id, sender_core, writer_runtime_args);
    }

    return {
        std::move(program),
        {.ternary_reader_kernel_id = ternary_reader_kernel_id,
         .binary_writer_kernel_id = binary_writer_kernel_id,
         .cores = sender_cores,
         .init_semaphore = init_semaphore,
         .cross_device_semaphore = cross_device_semaphore,
         .skip_init_semaphore = skip_init_semaphore}};
}

void AllToAllDispatchMetadataDeviceOperation::AllToAllDispatchMetadataSparse::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared_variables = cached_workload.shared_variables.at(range);
        const auto& ternary_reader_kernel_id = shared_variables.ternary_reader_kernel_id;
        const auto& binary_writer_kernel_id = shared_variables.binary_writer_kernel_id;
        const auto& cores = shared_variables.cores;

        const auto& output_tensor = tensor_return_value.at(0);
        const auto& metadata_tensor = tensor_return_value.at(1);
        const auto& scores_out_tensor = tensor_return_value.at(2);

        // Use the cross_device_semaphore from operation_attributes if provided (for double-buffering),
        // otherwise fall back to the cached one from shared_variables
        const auto& cross_device_semaphore = operation_attributes.cross_device_semaphore.has_value()
                                                 ? operation_attributes.cross_device_semaphore.value()
                                                 : shared_variables.cross_device_semaphore;

        for (const auto& core : cores) {
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, ternary_reader_kernel_id, core);
            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, binary_writer_kernel_id, core);
            reader_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            reader_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
            reader_runtime_args.at(2) = tensor_args.expert_scores_tensor.buffer()->address();
            reader_runtime_args.at(3) = tensor_args.expert_mapping_tensor.buffer()->address();
            reader_runtime_args.at(4) = output_tensor.buffer()->address();
            reader_runtime_args.at(5) = metadata_tensor.buffer()->address();
            reader_runtime_args.at(6) = scores_out_tensor.buffer()->address();
            reader_runtime_args.at(7) = (uint32_t)cross_device_semaphore.address();

            writer_runtime_args.at(0) = tensor_args.input_tensor.buffer()->address();
            writer_runtime_args.at(1) = tensor_args.expert_indices_tensor.buffer()->address();
            writer_runtime_args.at(2) = tensor_args.expert_scores_tensor.buffer()->address();
            writer_runtime_args.at(3) = tensor_args.expert_mapping_tensor.buffer()->address();
            writer_runtime_args.at(4) = output_tensor.buffer()->address();
            writer_runtime_args.at(5) = metadata_tensor.buffer()->address();
            writer_runtime_args.at(6) = scores_out_tensor.buffer()->address();
            writer_runtime_args.at(7) = (uint32_t)cross_device_semaphore.address();
            // init_semaphore is optional - only update if it exists (not in persistent mode)
            if (shared_variables.init_semaphore.has_value()) {
                writer_runtime_args.at(8) = (uint32_t)shared_variables.init_semaphore->address();
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::ccl
