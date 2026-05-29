// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ranges>
#include <variant>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {
namespace detail {

std::vector<uint32_t> data_parallel_split(
    uint32_t token_size_bytes, const uint32_t max_packet_size_bytes, const uint32_t num_data_parallel_cores) {
    std::vector<uint32_t> data_parallel_sizes_bytes;

    const uint32_t need_data_parallel_cores =
        std::max(num_data_parallel_cores, token_size_bytes / max_packet_size_bytes);
    data_parallel_sizes_bytes.reserve(need_data_parallel_cores);

    const uint32_t max_segment_size_bytes = token_size_bytes / need_data_parallel_cores;

    for (uint32_t c = 0; c < num_data_parallel_cores; ++c) {
        const uint32_t token_increment = std::min(token_size_bytes, max_segment_size_bytes);
        data_parallel_sizes_bytes.push_back(token_increment);
        token_size_bytes -= token_increment;

        if (token_size_bytes == 0) {
            break;
        }
    }

    return data_parallel_sizes_bytes;
}

auto launch_mux_workers(
    const MeshDevice& mesh_device,
    const CoreRangeSet& mux_core_range_set,
    const tt::tt_fabric::FabricNodeId src_node_id,
    const std::vector<ttnn::MeshCoordinate>& neighbors,
    const uint32_t num_links,
    const uint32_t num_workers,
    Program& program) {
    const auto num_header_only_channels = tt::div_up(num_workers, num_links);
    const auto num_full_size_channels = tt::div_up(num_workers, num_links);
    constexpr auto num_buffers_full_size_channels = 15;
    constexpr auto num_buffers_header_only_channels = 15;

    const size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const auto l1_unreserved_base_address =
        mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        num_buffers_header_only_channels,
        buffer_size_bytes_full_size_channel,
        l1_unreserved_base_address);

    const auto occupied_l1_tensor_addr = mesh_device.lowest_occupied_compute_l1_address();
    if (occupied_l1_tensor_addr.has_value()) {
        TT_FATAL(
            mux_kernel_config.get_memory_map_end_address() <= *occupied_l1_tensor_addr,
            "Mux L1 memory [base={:#x}, end={:#x}] overlaps with L1 tensor {:#x} and is in danger of being clobbered.",
            l1_unreserved_base_address,
            mux_kernel_config.get_memory_map_end_address(),
            *occupied_l1_tensor_addr);
    }

    // Calculate required vs available mux cores for fabric communication (one core per link per neighbor)
    const uint32_t needed_cores = num_links * neighbors.size();
    const uint32_t available_cores = mux_core_range_set.num_cores();

    // Validate sufficient cores exist before selection to prevent segfault in select_from_corerangeset
    TT_FATAL(
        needed_cores <= available_cores,
        "Not enough mux cores! Needed: {} (num_links={} * neighbors.size()={}), Available: {}. "
        "mux_core_range_set={}",
        needed_cores,
        num_links,
        neighbors.size(),
        available_cores,
        mux_core_range_set.str());

    const auto needed_mux_core_range_set = select_from_corerangeset(mux_core_range_set, 0, needed_cores - 1);

    auto mux_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        needed_mux_core_range_set,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
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

void add_termination_master_rt_args(
    const std::map<ttnn::MeshCoordinate, CoreCoord>& mux_neigbor_core_map, std::vector<uint32_t>& writer_runtime_args) {
    for (const auto& c : mux_neigbor_core_map) {
        const auto& mux_virtual_core = c.second;
        writer_runtime_args.push_back(mux_virtual_core.x);
        writer_runtime_args.push_back(mux_virtual_core.y);
    }
}

// ProgramDescriptor variant of launch_mux_workers above.
// Appends the mux kernel onto the caller's ProgramDescriptor (with its runtime
// args baked in per logical core) instead of issuing imperative CreateKernel /
// SetRuntimeArgs calls.  The kernel index is the position of the appended
// KernelDescriptor in desc.kernels.
auto launch_mux_workers_descriptor(
    const MeshDevice& mesh_device,
    const CoreRangeSet& mux_core_range_set,
    const tt::tt_fabric::FabricNodeId src_node_id,
    const std::vector<ttnn::MeshCoordinate>& neighbors,
    const uint32_t num_links,
    const uint32_t num_workers,
    tt::tt_metal::ProgramDescriptor& desc) {
    const auto num_header_only_channels = tt::div_up(num_workers, num_links);
    const auto num_full_size_channels = tt::div_up(num_workers, num_links);
    constexpr auto num_buffers_full_size_channels = 15;
    constexpr auto num_buffers_header_only_channels = 15;

    const size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const auto l1_unreserved_base_address =
        mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    auto mux_kernel_config = tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        num_buffers_header_only_channels,
        buffer_size_bytes_full_size_channel,
        l1_unreserved_base_address);

    const auto occupied_l1_tensor_addr = mesh_device.lowest_occupied_compute_l1_address();
    if (occupied_l1_tensor_addr.has_value()) {
        TT_FATAL(
            mux_kernel_config.get_memory_map_end_address() <= *occupied_l1_tensor_addr,
            "Mux L1 memory [base={:#x}, end={:#x}] overlaps with L1 tensor {:#x} and is in danger of being clobbered.",
            l1_unreserved_base_address,
            mux_kernel_config.get_memory_map_end_address(),
            *occupied_l1_tensor_addr);
    }

    // Calculate required vs available mux cores for fabric communication (one core per link per neighbor)
    const uint32_t needed_cores = num_links * neighbors.size();
    const uint32_t available_cores = mux_core_range_set.num_cores();

    // Validate sufficient cores exist before selection to prevent segfault in select_from_corerangeset
    TT_FATAL(
        needed_cores <= available_cores,
        "Not enough mux cores! Needed: {} (num_links={} * neighbors.size()={}), Available: {}. "
        "mux_core_range_set={}",
        needed_cores,
        num_links,
        neighbors.size(),
        available_cores,
        mux_core_range_set.str());

    const auto needed_mux_core_range_set = select_from_corerangeset(mux_core_range_set, 0, needed_cores - 1);

    tt::tt_metal::KernelDescriptor mux_kernel_desc;
    mux_kernel_desc.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
    mux_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    mux_kernel_desc.core_ranges = needed_mux_core_range_set;
    mux_kernel_desc.compile_time_args = mux_kernel_config.get_fabric_mux_compile_time_args();
    mux_kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
        .noc = tt::tt_metal::NOC::NOC_1,
    };
    mux_kernel_desc.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
    const auto mux_kernel_index = static_cast<tt::tt_metal::KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(mux_kernel_desc));

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
            // Templated FabricMuxConfig helper supports both Program and ProgramDescriptor;
            // here it appends fabric routing args onto the descriptor side and returns the
            // resulting per-core rt-arg vector.
            mux_rt_args =
                mux_kernel_config.get_fabric_mux_run_time_args(src_node_id, dst_node_id, link, desc, mux_logical_core);

            std::vector<std::variant<uint32_t, tt::tt_metal::Buffer*>> mux_rt_args_variant;
            mux_rt_args_variant.reserve(mux_rt_args.size());
            for (uint32_t a : mux_rt_args) {
                mux_rt_args_variant.emplace_back(a);
            }
            desc.kernels[mux_kernel_index].emplace_runtime_args(mux_logical_core, mux_rt_args_variant);
            mux_neigbor_core_map[neighbor_coord] = mux_virtual_core;
        }
        mux_neigbor_core_maps.push_back(mux_neigbor_core_map);
    }

    return std::make_tuple(mux_kernel_index, mux_kernel_config, mux_neigbor_core_maps);
}

}  // namespace detail

tt::tt_metal::WorkloadDescriptor UnifiedSelectReduce::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    auto* mesh_device = tensor_args.dense_input_tensor.device();
    const ttnn::CoreRangeSet worker_core_range_set(operation_attributes.worker_cores);

    // Workload-scoped GlobalSemaphores: allocated once on cache miss and parked
    // on workload_descriptor.semaphores so they outlive the cached workload.
    // The init/final barrier semaphores synchronize handoffs between devices,
    // hence the Synchronize call before kicking off any per-coord program.
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, worker_core_range_set, 0);

    GlobalSemaphore final_barrier_semaphore =
        operation_attributes.optional_cross_device_semaphore.has_value()
            ? operation_attributes.optional_cross_device_semaphore.value()
            : ttnn::global_semaphore::create_global_semaphore(mesh_device, worker_core_range_set, 0);

    tt::tt_metal::distributed::Synchronize(
        mesh_device, std::nullopt, {});  // interaction with subdevice needs to be investigated

    workload_descriptor.semaphores.push_back(init_barrier_semaphore);
    if (!operation_attributes.optional_cross_device_semaphore.has_value()) {
        // Only park the final barrier when we allocated it ourselves; if the
        // caller passed one in, lifetime is the caller's responsibility.
        workload_descriptor.semaphores.push_back(final_barrier_semaphore);
    }

    const auto all_mesh_coordinates = tensor_coords.coords();

    // Build a ProgramDescriptor per mesh coord — the descriptor builder bakes
    // in per-coord fabric routing (src_chip_id, neighbors) so each program is
    // unique.
    for (const auto& coord : all_mesh_coordinates) {
        tt::tt_metal::ProgramDescriptor desc;

        // Allocate the two worker-scoped sync semaphores as SemaphoreDescriptors
        // on `desc`.  The legacy factory created these via
        // CreateSemaphore(program, ...) inside create_at(); the descriptor
        // builder takes their IDs by value so it does not need to know how the
        // caller allocated them.
        //
        // metadata_sync covers the bounding box of the worker cores and is
        // initialized to 1; compute_sync covers all worker cores and starts at
        // 0.  These initial values match the legacy CreateSemaphore calls.
        const auto metadata_sync_core_ranges = CoreRangeSet(worker_core_range_set.bounding_box());
        const auto probe_core = worker_core_range_set.bounding_box().start_coord;
        auto metadata_sync_id_opt = desc.find_available_semaphore_id(probe_core, tt::CoreType::WORKER);
        TT_FATAL(metadata_sync_id_opt.has_value(), "No available semaphore ID for metadata sync");
        const uint32_t metadata_sync_semaphore_id = metadata_sync_id_opt.value();
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = metadata_sync_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = metadata_sync_core_ranges,
            .initial_value = 1});

        // Compute sync ID must be distinct from metadata_sync on shared cores;
        // pushing metadata_sync above ensures find_available_semaphore_id will
        // skip the freshly used ID here.
        auto compute_sync_id_opt = desc.find_available_semaphore_id(probe_core, tt::CoreType::WORKER);
        TT_FATAL(compute_sync_id_opt.has_value(), "No available semaphore ID for compute sync");
        const uint32_t compute_sync_semaphore_id = compute_sync_id_opt.value();
        desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
            .id = compute_sync_semaphore_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = worker_core_range_set,
            .initial_value = 0});

        build_selective_reduce_combine_program_artifacts_descriptor(
            desc,
            operation_attributes,
            coord,
            all_mesh_coordinates,
            tensor_args,
            tensor_return_value,
            init_barrier_semaphore,
            final_barrier_semaphore,
            metadata_sync_semaphore_id,
            compute_sync_semaphore_id);

        workload_descriptor.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }

    return workload_descriptor;
}

SelectiveReduceCombineProgramArtifacts build_selective_reduce_combine_program_artifacts(
    tt::tt_metal::Program& program,
    const experimental::prim::SelectiveReduceCombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const std::vector<MeshCoordinate>& all_mesh_coordinates,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    const uint32_t metadata_sync_semaphore_id,
    const uint32_t compute_sync_semaphore_id,
    const uint32_t compute_cores_per_combine_core,
    const std::optional<std::vector<CoreCoord>>& compute_cores_by_ring_id) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    const auto& input_tensor = tensor_args.dense_input_tensor;
    const auto& dense_token_maps_tensor = tensor_args.dense_token_maps_tensor;
    const auto& dense_token_counts_tensor = tensor_args.dense_token_counts_tensor;
    const auto& token_activations_tensor = tensor_args.dense_activations_tensor;

    const auto& output_tensor = tensor_return_value;
    const auto batch_size = operation_attributes.batch_size;
    const auto seq_size = operation_attributes.seq_size;
    const auto select_experts_k = operation_attributes.select_experts_k;
    const auto hidden_size = operation_attributes.hidden_size;

    const auto total_tokens = batch_size * seq_size;
    // Eventually map number of experts to device
    const auto experts = operation_attributes.experts;

    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto axis = operation_attributes.axis;

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const uint32_t num_devices_total = mesh_view.num_devices();
    const bool double_buffer_source = compute_cores_by_ring_id.has_value();

    // NOTE: shared experts are slightly delicate since they show up as an additional entry in the mapping tensor the
    // result is fractional experts per device so div_up is required to get the right value here.
    const uint32_t experts_per_device = tt::div_up(experts, num_devices_total);

    const auto input_dtype = input_tensor.dtype();
    const auto& dense_token_maps_tensor_spec = dense_token_maps_tensor.tensor_spec();

    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes =
        input_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;

    const uint32_t token_size_bytes = hidden_size * input_tensor.element_size();
    const uint32_t dense_token_maps_page_size_bytes = dense_token_maps_tensor_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto aligned_dense_token_maps_page_size_bytes = tt::align(dense_token_maps_page_size_bytes, l1_alignment);

    // in validate, assert that worker_core_range_set.size() == num_token_parallel_cores*num_data_parallel_cores;
    const auto num_token_parallel_cores = operation_attributes.num_token_parallel_cores;
    auto num_data_parallel_cores = operation_attributes.num_data_parallel_cores;
    const auto& worker_cores = operation_attributes.worker_cores;

    // in validate mux_core_range_set.size() == 2(directions) * num_links
    const auto& mux_core_range_set = operation_attributes.mux_core_range_set;

    const auto data_parallel_sizes_bytes =
        detail::data_parallel_split(token_size_bytes, max_packet_size_bytes, num_data_parallel_cores);

    num_data_parallel_cores = data_parallel_sizes_bytes.size();
    const auto num_worker_cores = num_token_parallel_cores * num_data_parallel_cores;
    const std::vector<CoreCoord> sender_cores(worker_cores.begin(), worker_cores.begin() + num_worker_cores);
    const ttnn::CoreRangeSet needed_worker_core_range_set(sender_cores);

    // buffer padding NOT supported because we don't rely on tensor shapes to represent the data layout
    const auto token_segment_buffer_size_bytes =
        *std::max_element(data_parallel_sizes_bytes.begin(), data_parallel_sizes_bytes.end());

    constexpr auto double_buffer = 2;
    const auto num_buffers = (double_buffer_source) ? double_buffer : experts_per_device;

    // TODO (AFM) this is an ugly kludge until we can get GPT-OSS on the mainline op #43645
    uint32_t expert_token_segment_buffer_block_size_bytes;
    if (double_buffer_source) {
        // slightly awkward. we want the token dimension but the underlying shape might not represent the data layout.
        //  This is in line with the assumption that tokens are split across the entirety of the shard, regardless of
        //  number of tokens
        const auto input_shards = input_tensor.memory_config().shard_spec()->grid.num_cores();
        const auto token_expert_row_offset = input_tensor.logical_shape().volume() / input_shards /
                                             (hidden_size / num_data_parallel_cores / double_buffer) /
                                             num_token_parallel_cores;

        expert_token_segment_buffer_block_size_bytes = token_segment_buffer_size_bytes * token_expert_row_offset;
    } else {
        expert_token_segment_buffer_block_size_bytes =
            token_segment_buffer_size_bytes * total_tokens / num_token_parallel_cores;
    }

    const auto buffer_size_bytes = expert_token_segment_buffer_block_size_bytes * num_buffers;

    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    // input sharded buffer
    // start at this cb index so we don't clash with compute when fused
    constexpr auto data_cb_id = tt::CBIndex::c_3;
    CircularBufferConfig cb_data_config = CircularBufferConfig(buffer_size_bytes, {{data_cb_id, input_data_format}})
                                              .set_page_size(data_cb_id, buffer_size_bytes)
                                              .set_globally_allocated_address(*input_tensor.buffer());

    // dense_token_maps_tensor page buffer
    // tensor pages are padded for alignment
    const uint32_t dense_token_maps_stride_elm = dense_token_maps_tensor.logical_shape()[-1] / total_tokens;
    constexpr auto dense_token_maps_cb_id = tt::CBIndex::c_4;
    const uint32_t aligned_dense_token_maps_buffer_size_bytes =
        tt::align(experts_per_device * aligned_dense_token_maps_page_size_bytes, l1_alignment);
    const auto dense_token_maps_data_format = datatype_to_dataformat_converter(dense_token_maps_tensor.dtype());
    CircularBufferConfig cb_dense_token_maps_config =
        CircularBufferConfig(
            aligned_dense_token_maps_buffer_size_bytes, {{dense_token_maps_cb_id, dense_token_maps_data_format}})
            .set_page_size(dense_token_maps_cb_id, aligned_dense_token_maps_page_size_bytes);

    // active token counts page buffer
    const auto token_counts_data_format = datatype_to_dataformat_converter(dense_token_counts_tensor.dtype());
    // offset into token maps, number of tokens, offset into activations
    const auto token_offset_count_bytes_per_expert = 3 * tt::datum_size(token_counts_data_format);
    constexpr auto token_counts_cb_id = tt::CBIndex::c_5;
    const auto token_counts_tensor_page_size_bytes = dense_token_counts_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t aligned_token_counts_buffer_size = tt::align(
        token_counts_tensor_page_size_bytes + token_offset_count_bytes_per_expert * experts_per_device, l1_alignment);
    CircularBufferConfig cb_token_counts_config =
        CircularBufferConfig(aligned_token_counts_buffer_size, {{token_counts_cb_id, token_counts_data_format}})
            .set_page_size(token_counts_cb_id, aligned_token_counts_buffer_size);

    // token activations metadata
    // page size: total tokens * (2 * experts_per_device + 1 + 3) * sizeof(uint32_t)
    const uint32_t activations_stride_elm = token_activations_tensor.logical_shape()[-1] / total_tokens;

    const auto token_activations_page_size_bytes = token_activations_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_token_activations_page_size_bytes = tt::align(token_activations_page_size_bytes, l1_alignment);
    constexpr auto token_activations_cb_id = tt::CBIndex::c_6;
    CircularBufferConfig cb_token_activations_config =
        CircularBufferConfig(
            aligned_token_activations_page_size_bytes, {{token_activations_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(token_activations_cb_id, token_activations_page_size_bytes);

    // client interface
    constexpr auto num_headers = 3;  // data unicast headers and atomic inc multicast headers
    constexpr auto client_interface_cb_id = tt::CBIndex::c_7;
    CircularBufferConfig client_interface_cb_config =
        CircularBufferConfig(num_headers * CLIENT_INTERFACE_SIZE, {{client_interface_cb_id, tt::DataFormat::UInt32}})
            .set_page_size(client_interface_cb_id, CLIENT_INTERFACE_SIZE);

    // create circular buffers
    const auto data_cb_handle = CreateCircularBuffer(program, needed_worker_core_range_set, cb_data_config);
    CreateCircularBuffer(program, needed_worker_core_range_set, cb_dense_token_maps_config);
    CreateCircularBuffer(program, needed_worker_core_range_set, cb_token_counts_config);
    CreateCircularBuffer(program, needed_worker_core_range_set, cb_token_activations_config);
    CreateCircularBuffer(program, needed_worker_core_range_set, client_interface_cb_config);

    // fabric routing info
    std::vector<uint32_t> dest_mesh_id, dest_chip_id, route;
    for (const auto& coord : all_mesh_coordinates) {
        const auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    const auto [neighbors, directions] =
        operations::ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, axis);

    // launch mux
    const auto [mux_kernel_id, mux_kernel_config, mux_neigbor_core_maps] = detail::launch_mux_workers(
        *mesh_device, mux_core_range_set, fabric_node_id, neighbors, num_links, num_worker_cores, program);

    const auto needed_worker_core_bounding_box = needed_worker_core_range_set.bounding_box();
    const auto start_coord = mesh_device->worker_core_from_logical_core(needed_worker_core_bounding_box.start_coord);
    const auto end_coord = mesh_device->worker_core_from_logical_core(needed_worker_core_bounding_box.end_coord);

    // launch reader kernel
    std::unordered_map<std::string, uint32_t> reader_named_ct_args = {
        {"dense_token_maps_cb_id", dense_token_maps_cb_id},
        {"token_counts_cb_id", token_counts_cb_id},
        {"token_activations_cb_id", token_activations_cb_id},
        {"token_activations_page_size_bytes", token_activations_page_size_bytes},
        {"aligned_token_activations_page_size_bytes", aligned_token_activations_page_size_bytes},
        {"activations_stride_elm", activations_stride_elm},
        {"dense_token_maps_page_size_bytes", aligned_dense_token_maps_page_size_bytes},
        {"token_counts_page_size_bytes", token_counts_tensor_page_size_bytes},
        {"dense_token_maps_stride_elm", dense_token_maps_stride_elm},
        {"num_local_experts", experts_per_device},
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        {"global_num_tokens", total_tokens},
        {"select_experts_k", select_experts_k},
        {"sync_semaphore_id", metadata_sync_semaphore_id},
        {"noc_x_start", start_coord.x},
        {"noc_y_start", start_coord.y},
        {"noc_x_end", end_coord.x},
        {"noc_y_end", end_coord.y},
        {"worker_bounding_box_size", needed_worker_core_bounding_box.size()},
    };

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(dense_token_maps_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(dense_token_counts_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(token_activations_tensor.buffer()).append_to(reader_compile_time_args);

    const DataMovementConfig reader_config{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
        .compile_args = reader_compile_time_args,
        .named_compile_args = reader_named_ct_args};

    KernelHandle ternary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/reader.cpp",
        needed_worker_core_range_set,
        reader_config);

    // launch writer kernel
    const uint32_t flat_mesh_idx = operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);
    const bool use_init_semaphore = !tensor_args.optional_output_tensor.has_value() ||
                                    !operation_attributes.optional_cross_device_semaphore.has_value();

    // Writer compute sync: when used from MoE, use matmul's data-ready semaphore; else create local (standalone).
    const uint32_t writer_compute_sync_semaphore_id = compute_sync_semaphore_id;

    std::unordered_map<std::string, uint32_t> writer_named_ct_args = {
        {"dense_token_maps_cb_id", dense_token_maps_cb_id},
        {"data_cb_id", data_cb_id},
        {"token_activations_cb_id", token_activations_cb_id},
        {"token_counts_cb_id", token_counts_cb_id},
        {"activations_stride_elm", activations_stride_elm},
        {"packet_header_cb_id", client_interface_cb_id},
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        {"use_init_semaphore", use_init_semaphore},
        {"noc_x_start", start_coord.x},
        {"noc_y_start", start_coord.y},
        {"noc_x_end", end_coord.x},
        {"noc_y_end", end_coord.y},
        {"num_local_experts", experts_per_device},
        {"global_num_tokens", total_tokens},
        {"token_activations_page_size_bytes", aligned_token_activations_page_size_bytes},
        {"source_token_segment_buffer_size_bytes", token_segment_buffer_size_bytes},
        {"source_expert_block_size_bytes", expert_token_segment_buffer_block_size_bytes},
        {"token_size_bytes", token_size_bytes},
        {"dense_token_maps_stride_elm", dense_token_maps_stride_elm},
        {"alignment", l1_alignment},
        {"num_devices", num_devices_total},
        {"src_chip_id", src_chip_id},
        {"mesh_rows", mesh_view.num_rows()},
        {"mesh_cols", mesh_view.num_cols()},
        {"fabric_max_packet_size_bytes", max_packet_size_bytes},
        {"linearized_mesh_coord", flat_mesh_idx},
        {"topology", static_cast<uint32_t>(topology)},
        {"num_mux_workers_per_link", neighbors.size()},
        {"compute_sync_semaphore_id", writer_compute_sync_semaphore_id},
        {"compute_cores_per_combine_core", compute_cores_per_combine_core},
        {"double_buffer_source", compute_cores_by_ring_id.has_value()}};

    std::vector<uint32_t> writer_compile_time_args;
    ttnn::ccl::fabric_mux_connection_ct_args(
        num_data_parallel_cores * num_token_parallel_cores,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        writer_compile_time_args);
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    using operations::ccl::common::stringify;
    std::map<std::string, std::string> writer_defines = {
        {"DEST_CHIP_ID", stringify(dest_chip_id)},
        {"DEST_MESH_ID", stringify(dest_mesh_id)},
        {"DIRECTIONS", stringify(directions)}};

    writer_defines["REPLICATE_GROUP_AXIS"] = std::to_string(axis);

    const DataMovementConfig writer_config{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
        .compile_args = writer_compile_time_args,
        .defines = writer_defines,
        .named_compile_args = writer_named_ct_args};

    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp",
        needed_worker_core_range_set,
        writer_config);

    const auto termination_master_semaphore_id = CreateSemaphore(program, {needed_worker_core_range_set}, 0);
    const uint32_t num_workers_per_link = num_worker_cores / num_links;

    const auto idx = std::views::iota(std::size_t{0}, sender_cores.size());
    auto termination_master_cores = idx |
                                    std::views::filter([=](std::size_t i) { return i % num_workers_per_link == 0; }) |
                                    std::views::transform([&](std::size_t i) { return sender_cores[i]; });

    auto termination_master_core_iter = termination_master_cores.begin();

    uint32_t link_worker_idx = 0, token_parallel_idx = 0, dest_token_segment_offset_bytes = 0;
    auto core_map_iter = mux_neigbor_core_maps.cbegin();
    auto data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
    auto compute_cores_by_ring_iter =
        (compute_cores_by_ring_id.has_value()) ? std::make_optional(compute_cores_by_ring_id->cbegin()) : std::nullopt;
    for (const auto& sender_core : sender_cores) {
        const bool is_init_sync_core = sender_core == sender_cores.at(0);
        std::vector<uint32_t> reader_runtime_args = {
            dense_token_maps_tensor.buffer()->address(),    // dense_token_maps_addr
            dense_token_counts_tensor.buffer()->address(),  // dense_token_counts_addr
            token_activations_tensor.buffer()->address(),   // token_activations_addr
            token_parallel_idx,                             // token_parallel_core_id
            is_init_sync_core                               // sync_core
        };

        const auto source_token_segment_size_bytes = *(data_parallel_size_iter++);
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),  // output_base_addr
            source_token_segment_size_bytes,    // source_token_segment_size_bytes
            dest_token_segment_offset_bytes,    // dest_token_segment_size_bytes
            init_semaphore.address(),           // init_semaphore_addr
            cross_device_semaphore.address(),   // global_semaphore_addr
            is_init_sync_core                   // is_init_sync_core
        };

        // if the input is double buffered, coming from fused moe_compute, add the core coordinates of the compute cores
        // which get semaphore increments upon release of buffer segment.
        if (compute_cores_by_ring_iter.has_value()) {
            auto coords =
                std::ranges::subrange(
                    *compute_cores_by_ring_iter, (*compute_cores_by_ring_iter) + compute_cores_per_combine_core) |
                std::views::transform([&](const auto& c) { return mesh_device->worker_core_from_logical_core(c); }) |
                std::ranges::views::transform([](const auto& c) { return std::array{c.x, c.y}; }) |
                std::ranges::views::join;

            std::ranges::copy(coords, std::back_inserter(writer_runtime_args));
        }

        const bool is_termination_master = (sender_core == *termination_master_core_iter);
        for (const auto& neighbor_coordinate : neighbors) {
            const auto& mux_virtual_core = core_map_iter->at(neighbor_coordinate);

            ttnn::ccl::fabric_mux_connection_rt_args(
                true,
                is_termination_master,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_virtual_core,
                link_worker_idx,
                sender_core,
                mux_kernel_config,
                program,
                mesh_device->worker_core_from_logical_core(*termination_master_core_iter),
                writer_runtime_args,
                termination_master_semaphore_id);
        }

        // termination master is responsible for tearing down mux workers for given link, needs their coordinates
        if (is_termination_master) {
            detail::add_termination_master_rt_args(*core_map_iter, writer_runtime_args);
        }

        SetRuntimeArgs(program, ternary_reader_kernel_id, sender_core, reader_runtime_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, sender_core, writer_runtime_args);

        if (data_parallel_size_iter == data_parallel_sizes_bytes.cend()) {
            data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
            dest_token_segment_offset_bytes = 0;
            ++token_parallel_idx;
            if (compute_cores_by_ring_iter.has_value()) {
                compute_cores_by_ring_iter = std::make_optional(compute_cores_by_ring_id->cbegin());
            }

        } else {
            dest_token_segment_offset_bytes += source_token_segment_size_bytes;
            if (compute_cores_by_ring_iter.has_value()) {
                (*compute_cores_by_ring_iter) += compute_cores_per_combine_core;
            }
        }

        if (++link_worker_idx == num_workers_per_link) {
            link_worker_idx = 0;
            ++core_map_iter;
            ++termination_master_core_iter;
        }
    }

    return {
        .reader_kernel_id = ternary_reader_kernel_id,
        .writer_kernel_id = unary_writer_kernel_id,
        .data_cb_handle = data_cb_handle,
        .cores = sender_cores,
        .init_semaphore = init_semaphore,
        .cross_device_semaphore = cross_device_semaphore};
}

SelectiveReduceCombineProgramArtifactsDescriptor build_selective_reduce_combine_program_artifacts_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const experimental::prim::SelectiveReduceCombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const std::vector<MeshCoordinate>& all_mesh_coordinates,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    const uint32_t metadata_sync_semaphore_id,
    const uint32_t compute_sync_semaphore_id,
    const uint32_t compute_cores_per_combine_core,
    const std::optional<std::vector<CoreCoord>>& compute_cores_by_ring_id) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    const auto& input_tensor = tensor_args.dense_input_tensor;
    const auto& dense_token_maps_tensor = tensor_args.dense_token_maps_tensor;
    const auto& dense_token_counts_tensor = tensor_args.dense_token_counts_tensor;
    const auto& token_activations_tensor = tensor_args.dense_activations_tensor;

    const auto& output_tensor = tensor_return_value;
    const auto batch_size = operation_attributes.batch_size;
    const auto seq_size = operation_attributes.seq_size;
    const auto select_experts_k = operation_attributes.select_experts_k;
    const auto hidden_size = operation_attributes.hidden_size;

    const auto total_tokens = batch_size * seq_size;
    // Eventually map number of experts to device
    const auto experts = operation_attributes.experts;

    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto axis = operation_attributes.axis;

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const uint32_t num_devices_total = mesh_view.num_devices();
    const bool double_buffer_source = compute_cores_by_ring_id.has_value();

    // NOTE: shared experts are slightly delicate since they show up as an additional entry in the mapping tensor the
    // result is fractional experts per device so div_up is required to get the right value here.
    const uint32_t experts_per_device = tt::div_up(experts, num_devices_total);

    const auto input_dtype = input_tensor.dtype();
    const auto& dense_token_maps_tensor_spec = dense_token_maps_tensor.tensor_spec();

    const auto fabric_max_packet_size_bytes = get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t max_packet_size_bytes =
        input_dtype == DataType::BFLOAT16 ? std::bit_floor(fabric_max_packet_size_bytes) : fabric_max_packet_size_bytes;

    const uint32_t token_size_bytes = hidden_size * input_tensor.element_size();
    const uint32_t dense_token_maps_page_size_bytes = dense_token_maps_tensor_spec.compute_page_size_bytes();

    const auto l1_alignment = hal::get_l1_alignment();
    const auto aligned_dense_token_maps_page_size_bytes = tt::align(dense_token_maps_page_size_bytes, l1_alignment);

    // in validate, assert that worker_core_range_set.size() == num_token_parallel_cores*num_data_parallel_cores;
    const auto num_token_parallel_cores = operation_attributes.num_token_parallel_cores;
    auto num_data_parallel_cores = operation_attributes.num_data_parallel_cores;
    const auto& worker_cores = operation_attributes.worker_cores;

    // in validate mux_core_range_set.size() == 2(directions) * num_links
    const auto& mux_core_range_set = operation_attributes.mux_core_range_set;

    const auto data_parallel_sizes_bytes =
        detail::data_parallel_split(token_size_bytes, max_packet_size_bytes, num_data_parallel_cores);

    num_data_parallel_cores = data_parallel_sizes_bytes.size();
    const auto num_worker_cores = num_token_parallel_cores * num_data_parallel_cores;
    const std::vector<CoreCoord> sender_cores(worker_cores.begin(), worker_cores.begin() + num_worker_cores);
    const ttnn::CoreRangeSet needed_worker_core_range_set(sender_cores);

    // buffer padding NOT supported because we don't rely on tensor shapes to represent the data layout
    const auto token_segment_buffer_size_bytes =
        *std::max_element(data_parallel_sizes_bytes.begin(), data_parallel_sizes_bytes.end());

    constexpr auto double_buffer = 2;
    const auto num_buffers = (double_buffer_source) ? double_buffer : experts_per_device;

    // TODO (AFM) this is an ugly kludge until we can get GPT-OSS on the mainline op #43645
    uint32_t expert_token_segment_buffer_block_size_bytes;
    if (double_buffer_source) {
        // slightly awkward. we want the token dimension but the underlying shape might not represent the data layout.
        //  This is in line with the assumption that tokens are split across the entirety of the shard, regardless of
        //  number of tokens
        const auto input_shards = input_tensor.memory_config().shard_spec()->grid.num_cores();
        const auto token_expert_row_offset = input_tensor.logical_shape().volume() / input_shards /
                                             (hidden_size / num_data_parallel_cores / double_buffer) /
                                             num_token_parallel_cores;

        expert_token_segment_buffer_block_size_bytes = token_segment_buffer_size_bytes * token_expert_row_offset;
    } else {
        expert_token_segment_buffer_block_size_bytes =
            token_segment_buffer_size_bytes * total_tokens / num_token_parallel_cores;
    }

    const auto buffer_size_bytes = expert_token_segment_buffer_block_size_bytes * num_buffers;

    const auto input_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    // input sharded buffer
    // start at this cb index so we don't clash with compute when fused.
    // Equivalent of set_globally_allocated_address(*input_tensor.buffer()): CBDescriptor::buffer
    // pins the CB to the input tensor's L1 buffer and the framework's fast cache-hit path patches
    // its address when the tensor moves.
    constexpr auto data_cb_id = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = buffer_size_bytes,
        .core_ranges = needed_worker_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(data_cb_id),
            .data_format = input_data_format,
            .page_size = buffer_size_bytes,
        }}},
        .buffer = input_tensor.buffer(),
    });

    // dense_token_maps_tensor page buffer
    // tensor pages are padded for alignment
    const uint32_t dense_token_maps_stride_elm = dense_token_maps_tensor.logical_shape()[-1] / total_tokens;
    constexpr auto dense_token_maps_cb_id = tt::CBIndex::c_4;
    const uint32_t aligned_dense_token_maps_buffer_size_bytes =
        tt::align(experts_per_device * aligned_dense_token_maps_page_size_bytes, l1_alignment);
    const auto dense_token_maps_data_format = datatype_to_dataformat_converter(dense_token_maps_tensor.dtype());
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_dense_token_maps_buffer_size_bytes,
        .core_ranges = needed_worker_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(dense_token_maps_cb_id),
            .data_format = dense_token_maps_data_format,
            .page_size = aligned_dense_token_maps_page_size_bytes,
        }}},
    });

    // active token counts page buffer
    const auto token_counts_data_format = datatype_to_dataformat_converter(dense_token_counts_tensor.dtype());
    // offset into token maps, number of tokens, offset into activations
    const auto token_offset_count_bytes_per_expert = 3 * tt::datum_size(token_counts_data_format);
    constexpr auto token_counts_cb_id = tt::CBIndex::c_5;
    const auto token_counts_tensor_page_size_bytes = dense_token_counts_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t aligned_token_counts_buffer_size = tt::align(
        token_counts_tensor_page_size_bytes + token_offset_count_bytes_per_expert * experts_per_device, l1_alignment);
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_token_counts_buffer_size,
        .core_ranges = needed_worker_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(token_counts_cb_id),
            .data_format = token_counts_data_format,
            .page_size = aligned_token_counts_buffer_size,
        }}},
    });

    // token activations metadata
    // page size: total tokens * (2 * experts_per_device + 1 + 3) * sizeof(uint32_t)
    const uint32_t activations_stride_elm = token_activations_tensor.logical_shape()[-1] / total_tokens;

    const auto token_activations_page_size_bytes = token_activations_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_token_activations_page_size_bytes = tt::align(token_activations_page_size_bytes, l1_alignment);
    constexpr auto token_activations_cb_id = tt::CBIndex::c_6;
    desc.cbs.push_back(CBDescriptor{
        .total_size = aligned_token_activations_page_size_bytes,
        .core_ranges = needed_worker_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(token_activations_cb_id),
            .data_format = tt::DataFormat::UInt32,
            .page_size = token_activations_page_size_bytes,
        }}},
    });

    // client interface
    constexpr auto num_headers = 3;  // data unicast headers and atomic inc multicast headers
    constexpr auto client_interface_cb_id = tt::CBIndex::c_7;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_headers * CLIENT_INTERFACE_SIZE,
        .core_ranges = needed_worker_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(client_interface_cb_id),
            .data_format = tt::DataFormat::UInt32,
            .page_size = CLIENT_INTERFACE_SIZE,
        }}},
    });

    // fabric routing info
    std::vector<uint32_t> dest_mesh_id, dest_chip_id, route;
    for (const auto& coord : all_mesh_coordinates) {
        const auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    const auto [neighbors, directions] =
        operations::ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, axis);

    // launch mux (descriptor variant: appends mux kernel + its per-core rt args onto desc)
    const auto [mux_kernel_index, mux_kernel_config, mux_neigbor_core_maps] = detail::launch_mux_workers_descriptor(
        *mesh_device, mux_core_range_set, fabric_node_id, neighbors, num_links, num_worker_cores, desc);

    const auto needed_worker_core_bounding_box = needed_worker_core_range_set.bounding_box();
    const auto start_coord = mesh_device->worker_core_from_logical_core(needed_worker_core_bounding_box.start_coord);
    const auto end_coord = mesh_device->worker_core_from_logical_core(needed_worker_core_bounding_box.end_coord);

    // launch reader kernel
    std::unordered_map<std::string, uint32_t> reader_named_ct_args_map = {
        {"dense_token_maps_cb_id", dense_token_maps_cb_id},
        {"token_counts_cb_id", token_counts_cb_id},
        {"token_activations_cb_id", token_activations_cb_id},
        {"token_activations_page_size_bytes", token_activations_page_size_bytes},
        {"aligned_token_activations_page_size_bytes", aligned_token_activations_page_size_bytes},
        {"activations_stride_elm", activations_stride_elm},
        {"dense_token_maps_page_size_bytes", aligned_dense_token_maps_page_size_bytes},
        {"token_counts_page_size_bytes", token_counts_tensor_page_size_bytes},
        {"dense_token_maps_stride_elm", dense_token_maps_stride_elm},
        {"num_local_experts", experts_per_device},
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        {"global_num_tokens", total_tokens},
        {"select_experts_k", select_experts_k},
        {"sync_semaphore_id", metadata_sync_semaphore_id},
        {"noc_x_start", start_coord.x},
        {"noc_y_start", start_coord.y},
        {"noc_x_end", end_coord.x},
        {"noc_y_end", end_coord.y},
        {"worker_bounding_box_size", needed_worker_core_bounding_box.size()},
    };

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(dense_token_maps_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(dense_token_counts_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(token_activations_tensor.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/reader.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = needed_worker_core_range_set;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.named_compile_time_args.assign(reader_named_ct_args_map.begin(), reader_named_ct_args_map.end());
    reader_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_1,
        .noc = NOC::NOC_1,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
    };
    const auto reader_kernel_index = static_cast<KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(reader_kernel_desc));

    // launch writer kernel
    const uint32_t flat_mesh_idx = operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);
    const bool use_init_semaphore = !tensor_args.optional_output_tensor.has_value() ||
                                    !operation_attributes.optional_cross_device_semaphore.has_value();

    // Writer compute sync: when used from MoE, use matmul's data-ready semaphore; else create local (standalone).
    const uint32_t writer_compute_sync_semaphore_id = compute_sync_semaphore_id;

    std::unordered_map<std::string, uint32_t> writer_named_ct_args_map = {
        {"dense_token_maps_cb_id", dense_token_maps_cb_id},
        {"data_cb_id", data_cb_id},
        {"token_activations_cb_id", token_activations_cb_id},
        {"token_counts_cb_id", token_counts_cb_id},
        {"activations_stride_elm", activations_stride_elm},
        {"packet_header_cb_id", client_interface_cb_id},
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        {"use_init_semaphore", use_init_semaphore},
        {"noc_x_start", start_coord.x},
        {"noc_y_start", start_coord.y},
        {"noc_x_end", end_coord.x},
        {"noc_y_end", end_coord.y},
        {"num_local_experts", experts_per_device},
        {"global_num_tokens", total_tokens},
        {"token_activations_page_size_bytes", aligned_token_activations_page_size_bytes},
        {"source_token_segment_buffer_size_bytes", token_segment_buffer_size_bytes},
        {"source_expert_block_size_bytes", expert_token_segment_buffer_block_size_bytes},
        {"token_size_bytes", token_size_bytes},
        {"dense_token_maps_stride_elm", dense_token_maps_stride_elm},
        {"alignment", l1_alignment},
        {"num_devices", num_devices_total},
        {"src_chip_id", src_chip_id},
        {"mesh_rows", mesh_view.num_rows()},
        {"mesh_cols", mesh_view.num_cols()},
        {"fabric_max_packet_size_bytes", max_packet_size_bytes},
        {"linearized_mesh_coord", flat_mesh_idx},
        {"topology", static_cast<uint32_t>(topology)},
        {"num_mux_workers_per_link", neighbors.size()},
        {"compute_sync_semaphore_id", writer_compute_sync_semaphore_id},
        {"compute_cores_per_combine_core", compute_cores_per_combine_core},
        {"double_buffer_source", compute_cores_by_ring_id.has_value()}};

    std::vector<uint32_t> writer_compile_time_args;
    ttnn::ccl::fabric_mux_connection_ct_args(
        num_data_parallel_cores * num_token_parallel_cores,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        writer_compile_time_args);
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    using operations::ccl::common::stringify;
    KernelDescriptor::Defines writer_defines = {
        {"DEST_CHIP_ID", stringify(dest_chip_id)},
        {"DEST_MESH_ID", stringify(dest_mesh_id)},
        {"DIRECTIONS", stringify(directions)},
        {"REPLICATE_GROUP_AXIS", std::to_string(axis)}};

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = needed_worker_core_range_set;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.named_compile_time_args.assign(writer_named_ct_args_map.begin(), writer_named_ct_args_map.end());
    writer_kernel_desc.defines = std::move(writer_defines);
    writer_kernel_desc.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0,
        .noc = NOC::NOC_1,
        .noc_mode = tt::tt_metal::NOC_MODE::DM_DYNAMIC_NOC,
    };
    const auto writer_kernel_index = static_cast<KernelHandle>(desc.kernels.size());
    desc.kernels.push_back(std::move(writer_kernel_desc));

    // Allocate a worker-scoped termination master semaphore via SemaphoreDescriptor.
    // This mirrors CreateSemaphore(program, {needed_worker_core_range_set}, 0) in the
    // legacy builder; the same semaphore id is then handed to every per-worker
    // fabric_mux_connection_rt_args invocation below.  We probe for the next free
    // ID using the first sender core (any worker core inside the range set is fine
    // since the caller has not allocated per-worker mux semaphores yet — those happen
    // below — and the termination master semaphore covers the same range set).
    const auto termination_master_sem_id_opt =
        desc.find_available_semaphore_id(sender_cores.at(0), tt::CoreType::WORKER);
    TT_FATAL(
        termination_master_sem_id_opt.has_value(),
        "No available semaphore ID for termination master in selective_reduce_combine builder");
    const uint32_t termination_master_semaphore_id = termination_master_sem_id_opt.value();
    desc.semaphores.push_back(SemaphoreDescriptor{
        .id = termination_master_semaphore_id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = needed_worker_core_range_set,
        .initial_value = 0,
    });

    const uint32_t num_workers_per_link = num_worker_cores / num_links;

    const auto idx = std::views::iota(std::size_t{0}, sender_cores.size());
    auto termination_master_cores = idx |
                                    std::views::filter([=](std::size_t i) { return i % num_workers_per_link == 0; }) |
                                    std::views::transform([&](std::size_t i) { return sender_cores[i]; });

    auto termination_master_core_iter = termination_master_cores.begin();

    uint32_t link_worker_idx = 0, token_parallel_idx = 0, dest_token_segment_offset_bytes = 0;
    auto core_map_iter = mux_neigbor_core_maps.cbegin();
    auto data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
    auto compute_cores_by_ring_iter =
        (compute_cores_by_ring_id.has_value()) ? std::make_optional(compute_cores_by_ring_id->cbegin()) : std::nullopt;
    for (const auto& sender_core : sender_cores) {
        const bool is_init_sync_core = sender_core == sender_cores.at(0);

        // Reader runtime args.  The first three positions are the input tensor buffer addresses;
        // we substitute Buffer* entries at those positions via emplace_runtime_args so the
        // framework's fast cache-hit path patches them automatically when the tensors move.
        std::vector<std::variant<uint32_t, Buffer*>> reader_runtime_args_variant;
        reader_runtime_args_variant.reserve(5);
        reader_runtime_args_variant.emplace_back(dense_token_maps_tensor.buffer());          // dense_token_maps_addr
        reader_runtime_args_variant.emplace_back(dense_token_counts_tensor.buffer());        // dense_token_counts_addr
        reader_runtime_args_variant.emplace_back(token_activations_tensor.buffer());         // token_activations_addr
        reader_runtime_args_variant.emplace_back(token_parallel_idx);                        // token_parallel_core_id
        reader_runtime_args_variant.emplace_back(static_cast<uint32_t>(is_init_sync_core));  // sync_core
        desc.kernels[reader_kernel_index].emplace_runtime_args(sender_core, reader_runtime_args_variant);

        const auto source_token_segment_size_bytes = *(data_parallel_size_iter++);
        // Build writer rt args in a plain uint32_t vector first so we can pass it to the
        // fabric_mux_connection_rt_args helper (which appends mux-connection trailers and
        // allocates a few worker-scoped semaphores by mutating the descriptor + the vector).
        // After the loop over neighbors and the optional termination-master tail, we swap the
        // first arg (output_tensor.address()) for a Buffer* binding via emplace_runtime_args.
        std::vector<uint32_t> writer_runtime_args = {
            output_tensor.buffer()->address(),  // output_base_addr  (replaced with Buffer* below)
            source_token_segment_size_bytes,    // source_token_segment_size_bytes
            dest_token_segment_offset_bytes,    // dest_token_segment_size_bytes
            init_semaphore.address(),           // init_semaphore_addr
            cross_device_semaphore.address(),   // global_semaphore_addr
            is_init_sync_core                   // is_init_sync_core
        };

        // if the input is double buffered, coming from fused moe_compute, add the core coordinates of the compute cores
        // which get semaphore increments upon release of buffer segment.
        if (compute_cores_by_ring_iter.has_value()) {
            auto coords =
                std::ranges::subrange(
                    *compute_cores_by_ring_iter, (*compute_cores_by_ring_iter) + compute_cores_per_combine_core) |
                std::views::transform([&](const auto& c) { return mesh_device->worker_core_from_logical_core(c); }) |
                std::ranges::views::transform([](const auto& c) { return std::array{c.x, c.y}; }) |
                std::ranges::views::join;

            std::ranges::copy(coords, std::back_inserter(writer_runtime_args));
        }

        const bool is_termination_master = (sender_core == *termination_master_core_iter);
        for (const auto& neighbor_coordinate : neighbors) {
            const auto& mux_virtual_core = core_map_iter->at(neighbor_coordinate);

            // Descriptor variant: allocates the five mux-side semaphores by pushing
            // SemaphoreDescriptors onto desc.semaphores instead of CreateSemaphore(program, ...).
            ttnn::ccl::fabric_mux_connection_rt_args(
                true,
                is_termination_master,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_virtual_core,
                link_worker_idx,
                sender_core,
                mux_kernel_config,
                desc,
                mesh_device->worker_core_from_logical_core(*termination_master_core_iter),
                writer_runtime_args,
                termination_master_semaphore_id);
        }

        // termination master is responsible for tearing down mux workers for given link, needs their coordinates
        if (is_termination_master) {
            detail::add_termination_master_rt_args(*core_map_iter, writer_runtime_args);
        }

        // Convert writer rt args to the variant arg list and substitute the output Buffer*
        // at the first position so the fast cache-hit path patches it.
        std::vector<std::variant<uint32_t, Buffer*>> writer_runtime_args_variant;
        writer_runtime_args_variant.reserve(writer_runtime_args.size());
        writer_runtime_args_variant.emplace_back(output_tensor.buffer());
        for (size_t i = 1; i < writer_runtime_args.size(); ++i) {
            writer_runtime_args_variant.emplace_back(writer_runtime_args[i]);
        }
        desc.kernels[writer_kernel_index].emplace_runtime_args(sender_core, writer_runtime_args_variant);

        if (data_parallel_size_iter == data_parallel_sizes_bytes.cend()) {
            data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
            dest_token_segment_offset_bytes = 0;
            ++token_parallel_idx;
            if (compute_cores_by_ring_iter.has_value()) {
                compute_cores_by_ring_iter = std::make_optional(compute_cores_by_ring_id->cbegin());
            }

        } else {
            dest_token_segment_offset_bytes += source_token_segment_size_bytes;
            if (compute_cores_by_ring_iter.has_value()) {
                (*compute_cores_by_ring_iter) += compute_cores_per_combine_core;
            }
        }

        if (++link_worker_idx == num_workers_per_link) {
            link_worker_idx = 0;
            ++core_map_iter;
            ++termination_master_core_iter;
        }
    }
    // Silence unused-warning for the mux kernel index — the descriptor doesn't expose it via the
    // returned artifacts because the caller doesn't need to address mux-kernel args directly
    // (the rt args were baked in by launch_mux_workers_descriptor).
    (void)mux_kernel_index;

    return {
        .reader_kernel_index = reader_kernel_index,
        .writer_kernel_index = writer_kernel_index,
        .cores = sender_cores,
        .init_semaphore = init_semaphore,
        .cross_device_semaphore = cross_device_semaphore};
}

void selective_reduce_combine_helper_override_runtime_arguments(
    tt::tt_metal::Program& program,
    tt::tt_metal::KernelHandle reader_kernel_id,
    tt::tt_metal::KernelHandle writer_kernel_id,
    tt::tt_metal::CBHandle data_cb_handle,
    const std::vector<CoreCoord>& cores,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const GlobalSemaphore& init_semaphore,
    const GlobalSemaphore& cross_device_semaphore,
    const std::optional<GlobalSemaphore>& optional_cross_device_semaphore) {
    tt::tt_metal::UpdateDynamicCircularBufferAddress(
        program, data_cb_handle, *tensor_args.dense_input_tensor.buffer());

    for (const auto& core : cores) {
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);

        reader_runtime_args.at(0) = tensor_args.dense_token_maps_tensor.buffer()->address();
        reader_runtime_args.at(1) = tensor_args.dense_token_counts_tensor.buffer()->address();
        reader_runtime_args.at(2) = tensor_args.dense_activations_tensor.buffer()->address();

        writer_runtime_args.at(0) = tensor_return_value.buffer()->address();
        writer_runtime_args.at(3) = static_cast<uint32_t>(init_semaphore.address());

        writer_runtime_args.at(4) = (optional_cross_device_semaphore.has_value())
                                        ? optional_cross_device_semaphore->address()
                                        : cross_device_semaphore.address();
    }
}

}  // namespace ttnn::experimental::prim
