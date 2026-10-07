// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <ranges>
#include <vector>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload_descriptor.hpp>
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

SelectiveReduceCombineWorkerLayout compute_worker_layout(
    const Tensor& input_tensor,
    const uint32_t hidden_size,
    const uint32_t num_token_parallel_cores,
    const uint32_t num_data_parallel_cores_attr,
    const bool local_combine) {
    // In local combine mode there is no fabric packet-size constraint; use a large value so
    // the data-parallel split is driven purely by num_data_parallel_cores_attr.
    const auto fabric_max_packet_size_bytes =
        local_combine ? std::numeric_limits<uint32_t>::max() : tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const auto input_dtype = input_tensor.dtype();
    const uint32_t max_packet_size_bytes = input_dtype == tt::tt_metal::DataType::BFLOAT16
                                               ? std::bit_floor(fabric_max_packet_size_bytes)
                                               : fabric_max_packet_size_bytes;
    const uint32_t token_size_bytes = hidden_size * input_tensor.element_size();
    auto data_parallel_sizes_bytes =
        data_parallel_split(token_size_bytes, max_packet_size_bytes, num_data_parallel_cores_attr);
    const uint32_t num_data_parallel_cores = static_cast<uint32_t>(data_parallel_sizes_bytes.size());
    return {
        .data_parallel_sizes_bytes = std::move(data_parallel_sizes_bytes),
        .num_data_parallel_cores = num_data_parallel_cores,
        .num_worker_cores = num_token_parallel_cores * num_data_parallel_cores,
    };
}

tt::tt_fabric::FabricMuxConfig get_fabric_mux_config(
    const uint32_t num_full_size_channels,
    const uint32_t num_header_only_channels,
    uint8_t num_buffers_full_size_channels,
    uint8_t num_buffers_header_only_channels,
    const size_t buffer_size_bytes_full_size_channel,
    const uint32_t l1_unreserved_base_address,
    const size_t usable_l1_end_address) {
    // Shrink the buffer counts until the memory map fits under the ceiling. Each candidate is sized with
    // no ceiling of its own, so this search is what does the shrinking: handing the ceiling to the
    // constructor up front would make *it* fatal on the first oversized candidate instead.
    while (true) {
        TT_FATAL(
            num_buffers_full_size_channels > 0 && num_buffers_header_only_channels > 0,
            "Not enough L1 space for mux core memory requirements given current occupancy. Likely too many experts "
            "per device");

        const auto candidate = tt::tt_fabric::FabricMuxConfig(
            num_full_size_channels,
            num_header_only_channels,
            num_buffers_full_size_channels,
            num_buffers_header_only_channels,
            buffer_size_bytes_full_size_channel,
            l1_unreserved_base_address);
        if (candidate.get_memory_map_end_address() <= usable_l1_end_address) {
            break;
        }

        --num_buffers_full_size_channels;
        --num_buffers_header_only_channels;
    }

    // It fits, so hand back a config that carries the ceiling. Identical memory map, but FabricMuxConfig
    // now asserts the invariant itself -- a backstop if this function's arithmetic ever drifts.
    return tt::tt_fabric::FabricMuxConfig(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        num_buffers_header_only_channels,
        buffer_size_bytes_full_size_channel,
        l1_unreserved_base_address,
        tt::CoreType::WORKER,
        usable_l1_end_address);
}

// Lowest id free on every core in the range; probe one core with find_available_semaphore_id.
uint32_t allocate_worker_semaphore(
    tt::tt_metal::ProgramDescriptor& desc, const CoreRangeSet& core_ranges, uint32_t initial_value) {
    const auto cores = corerange_to_cores(core_ranges);
    TT_FATAL(!cores.empty(), "Expecting a non-empty CoreRangeSet!");
    const auto first_free = desc.find_available_semaphore_id(cores.front(), tt::CoreType::WORKER);
    constexpr uint32_t kMaxSemaphores = 16;
    auto used_on_core = [&](uint32_t sem_id, const CoreCoord& core) {
        for (const auto& sem : desc.semaphores) {
            if (sem.core_type == tt::CoreType::WORKER && sem.id == sem_id && sem.core_ranges.contains(core)) {
                return true;
            }
        }
        return false;
    };
    std::optional<uint32_t> id;
    for (uint32_t candidate = first_free.value_or(0); candidate < kMaxSemaphores; ++candidate) {
        bool used = false;
        for (const auto& core : cores) {
            if (used_on_core(candidate, core)) {
                used = true;
                break;
            }
        }
        if (!used) {
            id = candidate;
            break;
        }
    }
    TT_FATAL(
        id.has_value(),
        "Unable to initialize semaphore on CoreRangeSet {}: all {} IDs are in use",
        core_ranges.str(),
        kMaxSemaphores);
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = *id,
        .core_type = tt::CoreType::WORKER,
        .core_ranges = core_ranges,
        .initial_value = initial_value,
    });
    return *id;
}

auto launch_mux_workers(
    const MeshDevice& mesh_device,
    const CoreRangeSet& mux_core_range_set,
    const tt::tt_fabric::FabricNodeId src_node_id,
    const std::vector<ttnn::MeshCoordinate>& neighbors,
    const uint32_t num_links,
    const uint32_t num_workers,
    tt::tt_metal::ProgramDescriptor& desc) {
    const auto num_header_only_channels = tt::div_up(num_workers, num_links);
    const auto num_full_size_channels = tt::div_up(num_workers, num_links);

    constexpr uint8_t num_buffers_full_size_channels = 15;
    constexpr uint8_t num_buffers_header_only_channels = 15;

    const size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const auto l1_unreserved_base_address =
        mesh_device.allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);

    // The mux carves raw L1 growing up from l1_unreserved_base_address, outside the allocator, so the
    // allocator never learns those bytes are taken and may hand them to a later allocation. That is
    // survivable for a tensor -- its producer rewrites it every iteration -- but not for a GlobalSemaphore,
    // whose value is written once at allocation and thereafter only incremented by the kernels that read
    // it. One clobber of a carried counter is unrecoverable, which is the #56769 hang.
    //
    // The fix is to keep semaphores out of the mux's reach entirely: with l1_small_size > 0 they are
    // allocated from L1_SMALL, which sits at the *top* of L1, and the ceiling below pins the mux beneath
    // it. Without an L1_SMALL region there is nowhere safe to put them, and we cannot detect the collision
    // later -- at this point the semaphores do not exist yet, so an occupancy check cannot see them.
    const auto l1_small_bank_size = mesh_device.allocator()->get_bank_size(tt::tt_metal::BufferType::L1_SMALL);
    TT_FATAL(
        l1_small_bank_size > 0,
        "The fabric mux reserves raw L1 on cores {} outside the allocator, but this device was opened with "
        "l1_small_size = 0. GlobalSemaphores then fall back to BufferType::L1 and can be allocated inside the "
        "mux's region, which silently overwrites them and hangs (#56769). Open the mesh device with "
        "l1_small_size > 0 (16384 is sufficient) so semaphores are placed in L1_SMALL, above the mux.",
        mux_core_range_set.str());

    // Keep the mux below the floor of the L1_SMALL region, where carried semaphores live (#56769), and
    // below whatever the allocator has already handed out in L1. A live occupancy reading is always the
    // tighter of the two -- the L1 bank sits entirely below L1_SMALL -- but it is only a build-time
    // snapshot, and with nothing allocated yet the static floor is all there is to go on.
    const size_t usable_l1_end_address =
        mesh_device.lowest_occupied_compute_l1_address().value_or(ttnn::ccl::l1_small_floor_address(mesh_device));

    auto mux_kernel_config = get_fabric_mux_config(
        num_full_size_channels,
        num_header_only_channels,
        num_buffers_full_size_channels,
        num_buffers_header_only_channels,
        buffer_size_bytes_full_size_channel,
        l1_unreserved_base_address,
        usable_l1_end_address);

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

    // Mux is pushed after the reader and before the writer (CCL kernel order).
    const uint32_t mux_kernel_idx = desc.kernels.size();
    desc.kernels.push_back(tt::tt_metal::KernelDescriptor{
        .kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
        .core_ranges = needed_mux_core_range_set,
        .compile_time_args = mux_kernel_config.get_fabric_mux_compile_time_args(),
        .opt_level = tt::tt_metal::KernelBuildOptLevel::O3,
        .config =
            tt::tt_metal::DataMovementConfigDescriptor{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                .noc = tt::tt_metal::NOC::NOC_1,
            },
    });

    std::vector<std::map<ttnn::MeshCoordinate, CoreCoord>> mux_neigbor_core_maps;
    mux_neigbor_core_maps.reserve(num_links);

    const auto mux_cores = corerange_to_cores(needed_mux_core_range_set);
    auto mux_core_iter = mux_cores.begin();
    for (uint32_t link = 0; link < num_links; ++link) {
        std::map<ttnn::MeshCoordinate, CoreCoord> mux_neigbor_core_map;
        for (const auto& neighbor_coord : neighbors) {
            auto mux_logical_core = *(mux_core_iter++);
            const auto mux_virtual_core = mesh_device.worker_core_from_logical_core(mux_logical_core);

            const auto dst_node_id = mesh_device.get_fabric_node_id(neighbor_coord);
            tt::tt_metal::KernelDescriptor::RTArgList mux_rt_args;
            mux_rt_args.append(
                mux_kernel_config.get_fabric_mux_run_time_args(src_node_id, dst_node_id, link, desc, mux_logical_core));
            desc.kernels[mux_kernel_idx].emplace_runtime_args(mux_logical_core, mux_rt_args);
            mux_neigbor_core_map[neighbor_coord] = mux_virtual_core;
        }
        mux_neigbor_core_maps.push_back(mux_neigbor_core_map);
    }

    return std::make_tuple(mux_kernel_config, mux_neigbor_core_maps);
}

void add_termination_master_rt_args(
    const std::map<ttnn::MeshCoordinate, CoreCoord>& mux_neigbor_core_map, std::vector<uint32_t>& writer_runtime_args) {
    for (const auto& c : mux_neigbor_core_map) {
        const auto& mux_virtual_core = c.second;
        writer_runtime_args.push_back(mux_virtual_core.x);
        writer_runtime_args.push_back(mux_virtual_core.y);
    }
}

}  // namespace detail
namespace {

// Standalone UnifiedSelectReduce kernel order when append starts with no kernels:
// reader 0, local_combine writer 1, CCL mux 1, CCL writer 2.
constexpr uint32_t kLocalCombineWriterKernelIdx = 1;
constexpr uint32_t kCclWriterKernelIdx = 2;
constexpr uint32_t kWriterCrossDeviceSemaphoreArgIdx = 4;

}  // namespace

tt::tt_metal::WorkloadDescriptor UnifiedSelectReduce::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;

    auto* mesh_device = tensor_args.dense_input_tensor.device();
    const ttnn::CoreRangeSet worker_core_range_set(operation_attributes.worker_cores);
    // Carried counters: keep them above the mux's ceiling where an L1_SMALL region exists. This op
    // also TT_FATALs below if there is none, so the general-L1 fallback never reaches a mux.
    const auto sem_buffer_type = ttnn::ccl::prefer_l1_small_buffer_type(*mesh_device);
    auto init_barrier_semaphore =
        ttnn::global_semaphore::create_global_semaphore(mesh_device, worker_core_range_set, 0, sem_buffer_type);

    auto final_barrier_semaphore = operation_attributes.optional_cross_device_semaphore.value_or(
        ttnn::global_semaphore::create_global_semaphore(mesh_device, worker_core_range_set, 0, sem_buffer_type));

    workload.semaphores.push_back(init_barrier_semaphore);
    workload.semaphores.push_back(final_barrier_semaphore);

    tt::tt_metal::distributed::Synchronize(
        *mesh_device, std::nullopt, {});  // interaction with subdevice needs to be investigated

    for (const auto& coord : tensor_coords.coords()) {
        tt::tt_metal::ProgramDescriptor desc;
        const uint32_t metadata_sync_semaphore_id =
            detail::allocate_worker_semaphore(desc, CoreRangeSet(worker_core_range_set.bounding_box()), 1);
        const uint32_t compute_sync_semaphore_id = detail::allocate_worker_semaphore(desc, worker_core_range_set, 0);
        append_selective_reduce_combine_to_descriptor(
            desc,
            operation_attributes,
            coord,
            tensor_coords.coords(),
            tensor_args,
            tensor_return_value,
            std::optional<GlobalSemaphore>(init_barrier_semaphore),
            std::optional<GlobalSemaphore>(final_barrier_semaphore),
            metadata_sync_semaphore_id,
            compute_sync_semaphore_id);
        workload.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return workload;
}

void append_selective_reduce_combine_to_descriptor(
    tt::tt_metal::ProgramDescriptor& desc,
    const experimental::prim::SelectiveReduceCombineParams& operation_attributes,
    const MeshCoordinate& mesh_coordinate,
    const std::vector<MeshCoordinate>& all_mesh_coordinates,
    const experimental::prim::SelectiveReduceCombineTensors& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<GlobalSemaphore>& init_semaphore,
    const std::optional<GlobalSemaphore>& cross_device_semaphore,
    const uint32_t metadata_sync_semaphore_id,
    const uint32_t compute_sync_semaphore_id,
    const uint32_t compute_cores_per_combine_cores,
    const std::optional<std::vector<CoreCoord>>& compute_cores_by_ring_id) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    // 0 when the caller has no semaphore (fused moe_compute FullLocal path: the writer
    // compiles out all init/final barrier handling under LOCAL_COMBINE).
    const uint32_t init_semaphore_addr =
        init_semaphore.has_value() ? static_cast<uint32_t>(init_semaphore->address()) : 0u;
    const uint32_t cross_device_semaphore_addr =
        cross_device_semaphore.has_value() ? static_cast<uint32_t>(cross_device_semaphore->address()) : 0u;

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

    const auto num_links = operation_attributes.num_links;
    auto topology = operation_attributes.topology;

    auto* mesh_device = input_tensor.device();
    const auto& mesh_view = mesh_device->get_view();

    const auto axis = operation_attributes.axis;

    const auto fabric_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    const uint32_t src_chip_id = (uint32_t)fabric_node_id.chip_id;

    const uint32_t num_devices_total = mesh_view.num_devices();
    const bool double_buffer_source = compute_cores_by_ring_id.has_value();

    // physical experts per device, replicated shared experts are counted per device
    const uint32_t experts_per_device = dense_token_maps_tensor.logical_shape()[0];

    const auto input_dtype = input_tensor.dtype();
    const auto& dense_token_maps_tensor_spec = dense_token_maps_tensor.tensor_spec();

    // In local combine mode, there is no fabric packet-size constraint.
    const auto fabric_max_packet_size_bytes = operation_attributes.local_combine
                                                  ? std::numeric_limits<uint32_t>::max()
                                                  : get_tt_fabric_channel_buffer_size_bytes();
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

    const auto worker_layout = detail::compute_worker_layout(
        input_tensor,
        hidden_size,
        num_token_parallel_cores,
        num_data_parallel_cores,
        operation_attributes.local_combine);
    const auto& data_parallel_sizes_bytes = worker_layout.data_parallel_sizes_bytes;
    num_data_parallel_cores = worker_layout.num_data_parallel_cores;
    const auto num_worker_cores = worker_layout.num_worker_cores;
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
    // dense_token_maps_tensor page buffer
    // tensor pages are padded for alignment
    // Each expert row holds (total_tokens + 1) token indices -- the extra slot is the -1 terminator -- and each index
    // is padded to the alignment for NoC DMA. Divide by the real row count: dividing by total_tokens over-estimates
    // the stride for small token counts (total_tokens == 1 -> 8, == 2 -> 6, <= 4 -> 5; all of them should be 4).
    const uint32_t dense_token_maps_stride_elm = dense_token_maps_tensor.logical_shape()[-1] / (total_tokens + 1);
    constexpr auto dense_token_maps_cb_id = tt::CBIndex::c_4;
    const uint32_t aligned_dense_token_maps_buffer_size_bytes =
        tt::align(experts_per_device * aligned_dense_token_maps_page_size_bytes, l1_alignment);
    const auto dense_token_maps_data_format = datatype_to_dataformat_converter(dense_token_maps_tensor.dtype());

    // active token counts page buffer
    const auto token_counts_data_format = datatype_to_dataformat_converter(dense_token_counts_tensor.dtype());
    // offset into token maps, number of tokens, offset into activations
    const auto token_offset_count_bytes_per_expert = 3 * tt::datum_size(token_counts_data_format);
    constexpr auto token_counts_cb_id = tt::CBIndex::c_5;
    const auto token_counts_tensor_page_size_bytes = dense_token_counts_tensor.tensor_spec().compute_page_size_bytes();
    const uint32_t aligned_token_counts_buffer_size = tt::align(
        token_counts_tensor_page_size_bytes + token_offset_count_bytes_per_expert * experts_per_device, l1_alignment);

    // token activations metadata
    // page size: total tokens * (2 * experts_per_device + 1 + 3) * sizeof(uint32_t)
    const uint32_t activations_stride_elm = token_activations_tensor.logical_shape()[-1] / total_tokens;

    const auto token_activations_page_size_bytes = token_activations_tensor.tensor_spec().compute_page_size_bytes();
    const auto aligned_token_activations_page_size_bytes = tt::align(token_activations_page_size_bytes, l1_alignment);
    constexpr auto token_activations_cb_id = tt::CBIndex::c_6;

    // client interface
    constexpr auto num_headers = 3;  // data unicast headers and atomic inc multicast headers
    constexpr auto client_interface_cb_id = tt::CBIndex::c_7;

    auto push_cb =
        [&](uint32_t cb_index, uint32_t total_size, uint32_t page_size, tt::DataFormat data_format, Buffer* buffer) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = total_size,
                .core_ranges = needed_worker_core_range_set,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(cb_index),
                    .data_format = data_format,
                    .page_size = page_size,
                }}},
                .buffer = buffer,
            });
        };
    push_cb(data_cb_id, buffer_size_bytes, buffer_size_bytes, input_data_format, input_tensor.buffer());
    push_cb(
        dense_token_maps_cb_id,
        aligned_dense_token_maps_buffer_size_bytes,
        aligned_dense_token_maps_page_size_bytes,
        dense_token_maps_data_format,
        nullptr);
    push_cb(
        token_counts_cb_id,
        aligned_token_counts_buffer_size,
        aligned_token_counts_buffer_size,
        token_counts_data_format,
        nullptr);
    push_cb(
        token_activations_cb_id,
        static_cast<uint32_t>(aligned_token_activations_page_size_bytes),
        static_cast<uint32_t>(token_activations_page_size_bytes),
        tt::DataFormat::UInt32,
        nullptr);
    push_cb(
        client_interface_cb_id,
        num_headers * CLIENT_INTERFACE_SIZE,
        CLIENT_INTERFACE_SIZE,
        tt::DataFormat::UInt32,
        nullptr);

    const auto needed_worker_core_bounding_box = needed_worker_core_range_set.bounding_box();
    const auto start_coord = mesh_device->worker_core_from_logical_core(needed_worker_core_bounding_box.start_coord);
    const auto end_coord = mesh_device->worker_core_from_logical_core(needed_worker_core_bounding_box.end_coord);

    // launch reader kernel (same for both CCL and local modes — it only reads metadata locally)
    std::vector<std::pair<std::string, uint32_t>> reader_named_ct_args = {
        {"dense_token_maps_cb_id", dense_token_maps_cb_id},
        {"token_counts_cb_id", token_counts_cb_id},
        {"token_activations_cb_id", token_activations_cb_id},
        {"token_activations_page_size_bytes", static_cast<uint32_t>(token_activations_page_size_bytes)},
        {"aligned_token_activations_page_size_bytes", static_cast<uint32_t>(aligned_token_activations_page_size_bytes)},
        {"activations_stride_elm", activations_stride_elm},
        {"dense_token_maps_page_size_bytes", aligned_dense_token_maps_page_size_bytes},
        {"token_counts_page_size_bytes", static_cast<uint32_t>(token_counts_tensor_page_size_bytes)},
        {"dense_token_maps_stride_elm", dense_token_maps_stride_elm},
        {"num_local_experts", experts_per_device},
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        {"global_num_tokens", total_tokens},
        {"select_experts_k", select_experts_k},
        {"sync_semaphore_id", metadata_sync_semaphore_id},
        {"noc_x_start", static_cast<uint32_t>(start_coord.x)},
        {"noc_y_start", static_cast<uint32_t>(start_coord.y)},
        {"noc_x_end", static_cast<uint32_t>(end_coord.x)},
        {"noc_y_end", static_cast<uint32_t>(end_coord.y)},
        {"worker_bounding_box_size", static_cast<uint32_t>(needed_worker_core_bounding_box.size())},
    };

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(dense_token_maps_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(dense_token_counts_tensor.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(token_activations_tensor.buffer()).append_to(reader_compile_time_args);

    // Standalone reader kernel index is 0. local_combine writer is kLocalCombineWriterKernelIdx;
    // CCL writer is kCclWriterKernelIdx.
    const uint32_t reader_kernel_idx = desc.kernels.size();
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/reader.cpp",
        .core_ranges = needed_worker_core_range_set,
        .compile_time_args = std::move(reader_compile_time_args),
        .named_compile_time_args = std::move(reader_named_ct_args),
        .config =
            DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_1,
                .noc = NOC::NOC_1,
                .noc_mode = NOC_MODE::DM_DYNAMIC_NOC,
            },
    });

    // Writer compute sync: when used from MoE, use matmul's data-ready semaphore; else create local (standalone).
    const uint32_t writer_compute_sync_semaphore_id = compute_sync_semaphore_id;
    const bool use_init_semaphore = !tensor_args.optional_output_tensor.has_value() ||
                                    !operation_attributes.optional_cross_device_semaphore.has_value();

    auto append_compute_core_coords = [&](KernelDescriptor::RTArgList& writer_rt, auto& compute_cores_by_ring_iter) {
        if (!compute_cores_by_ring_iter.has_value()) {
            return;
        }
        auto coords =
            std::ranges::subrange(
                *compute_cores_by_ring_iter, (*compute_cores_by_ring_iter) + compute_cores_per_combine_cores) |
            std::views::transform([&](const auto& c) { return mesh_device->worker_core_from_logical_core(c); }) |
            std::ranges::views::transform([](const auto& c) { return std::array{c.x, c.y}; }) |
            std::ranges::views::join;
        std::vector<uint32_t> coord_args;
        std::ranges::copy(coords, std::back_inserter(coord_args));
        writer_rt.append(coord_args);
    };

    auto make_reader_runtime_args = [&](uint32_t token_parallel_idx, uint32_t is_init_sync_core) {
        KernelDescriptor::RTArgList reader_rt;
        reader_rt.push_back(dense_token_maps_tensor.buffer());
        reader_rt.push_back(dense_token_counts_tensor.buffer());
        reader_rt.push_back(token_activations_tensor.buffer());
        reader_rt.push_back(token_parallel_idx);
        reader_rt.push_back(is_init_sync_core);
        return reader_rt;
    };

    auto make_writer_runtime_args = [&](uint32_t source_token_segment_size_bytes,
                                        uint32_t dest_token_segment_offset_bytes,
                                        uint32_t is_init_sync_core) {
        KernelDescriptor::RTArgList writer_rt;
        writer_rt.push_back(output_tensor.buffer());
        writer_rt.push_back(source_token_segment_size_bytes);
        writer_rt.push_back(dest_token_segment_offset_bytes);
        writer_rt.push_back(
            init_semaphore_addr);  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the WorkloadDescriptor)
        writer_rt.push_back(cross_device_semaphore_addr);  // smuggled-rta-ok: persistent GlobalSemaphore (parked on the
                                                           // WorkloadDescriptor)
        writer_rt.push_back(is_init_sync_core);
        return writer_rt;
    };

    // ------------------------------------------------------------------------
    // Local combine path: single-device, no fabric/mux/CCL.
    // ------------------------------------------------------------------------
    if (operation_attributes.local_combine) {
        std::vector<std::pair<std::string, uint32_t>> writer_named_ct_args = {
            {"dense_token_maps_cb_id", dense_token_maps_cb_id},
            {"data_cb_id", data_cb_id},
            {"token_activations_cb_id", token_activations_cb_id},
            {"token_counts_cb_id", token_counts_cb_id},
            {"activations_stride_elm", activations_stride_elm},
            {"num_token_parallel_cores", num_token_parallel_cores},
            {"num_data_parallel_cores", num_data_parallel_cores},
            {"use_init_semaphore", static_cast<uint32_t>(use_init_semaphore)},
            {"num_local_experts", experts_per_device},
            {"global_num_tokens", total_tokens},
            {"source_token_segment_buffer_size_bytes", token_segment_buffer_size_bytes},
            {"source_expert_block_size_bytes", expert_token_segment_buffer_block_size_bytes},
            {"token_size_bytes", token_size_bytes},
            {"dense_token_maps_stride_elm", dense_token_maps_stride_elm},
            {"alignment", l1_alignment},
            {"compute_sync_semaphore_id", writer_compute_sync_semaphore_id},
            {"compute_cores_per_combine_core", compute_cores_per_combine_cores},
            {"double_buffer_source", static_cast<uint32_t>(double_buffer_source)},
        };

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

        // kLocalCombineWriterKernelIdx on the standalone program (reader, then writer).
        const uint32_t writer_kernel_idx = desc.kernels.size();
        desc.kernels.push_back(KernelDescriptor{
            .kernel_source = "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/"
                             "dataflow/writer.cpp",
            .core_ranges = needed_worker_core_range_set,
            .compile_time_args = std::move(writer_compile_time_args),
            .named_compile_time_args = std::move(writer_named_ct_args),
            .defines = {{"LOCAL_COMBINE", "1"}},
            .config =
                DataMovementConfigDescriptor{
                    .processor = DataMovementProcessor::RISCV_0,
                    .noc = NOC::NOC_1,
                    .noc_mode = NOC_MODE::DM_DYNAMIC_NOC,
                },
        });

        // Set runtime args for each combine worker core.
        uint32_t token_parallel_idx = 0;
        uint32_t dest_token_segment_offset_bytes = 0;
        auto data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
        auto compute_cores_by_ring_iter = (compute_cores_by_ring_id.has_value())
                                              ? std::make_optional(compute_cores_by_ring_id->cbegin())
                                              : std::nullopt;
        for (const auto& sender_core : sender_cores) {
            const bool is_init_sync_core = sender_core == sender_cores.at(0);
            const uint32_t is_init_sync_core_arg = static_cast<uint32_t>(is_init_sync_core);

            desc.kernels[reader_kernel_idx].emplace_runtime_args(
                sender_core, make_reader_runtime_args(token_parallel_idx, is_init_sync_core_arg));

            const auto source_token_segment_size_bytes = *(data_parallel_size_iter++);
            auto writer_rt = make_writer_runtime_args(
                source_token_segment_size_bytes, dest_token_segment_offset_bytes, is_init_sync_core_arg);
            // Double-buffered source (fused moe_compute): add compute core coordinates for
            // semaphore increments upon release of buffer segment.
            append_compute_core_coords(writer_rt, compute_cores_by_ring_iter);
            desc.kernels[writer_kernel_idx].emplace_runtime_args(sender_core, writer_rt);

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
                    (*compute_cores_by_ring_iter) += compute_cores_per_combine_cores;
                }
            }
        }
        return;
    }

    // ------------------------------------------------------------------------
    // CCL combine path: multi-device fabric-based combine.
    // ------------------------------------------------------------------------

    // fabric routing info
    std::vector<uint32_t> dest_mesh_id, dest_chip_id, route;
    for (const auto& coord : all_mesh_coordinates) {
        const auto dest_fabric_node_id = mesh_device->get_fabric_node_id(coord);
        dest_mesh_id.push_back(*dest_fabric_node_id.mesh_id);
        dest_chip_id.push_back((uint32_t)dest_fabric_node_id.chip_id);
    }
    const auto [neighbors, directions] =
        operations::ccl::common::get_neighbors(mesh_view, mesh_coordinate, topology, axis);

    // launch mux (reader already pushed; writer is pushed after this)
    const auto [mux_kernel_config, mux_neigbor_core_maps] = detail::launch_mux_workers(
        *mesh_device, mux_core_range_set, fabric_node_id, neighbors, num_links, num_worker_cores, desc);

    // launch writer kernel
    const uint32_t flat_mesh_idx = operations::ccl::common::get_linearized_index(mesh_coordinate, mesh_view);

    const uint32_t num_workers_per_link = num_worker_cores / num_links;

    std::vector<std::pair<std::string, uint32_t>> writer_named_ct_args = {
        {"dense_token_maps_cb_id", dense_token_maps_cb_id},
        {"data_cb_id", data_cb_id},
        {"token_activations_cb_id", token_activations_cb_id},
        {"token_counts_cb_id", token_counts_cb_id},
        {"activations_stride_elm", activations_stride_elm},
        {"packet_header_cb_id", client_interface_cb_id},
        {"num_token_parallel_cores", num_token_parallel_cores},
        {"num_data_parallel_cores", num_data_parallel_cores},
        {"num_workers_per_link", num_workers_per_link},
        {"use_init_semaphore", static_cast<uint32_t>(use_init_semaphore)},
        {"noc_x_start", static_cast<uint32_t>(start_coord.x)},
        {"noc_y_start", static_cast<uint32_t>(start_coord.y)},
        {"noc_x_end", static_cast<uint32_t>(end_coord.x)},
        {"noc_y_end", static_cast<uint32_t>(end_coord.y)},
        {"num_local_experts", experts_per_device},
        {"global_num_tokens", total_tokens},
        {"token_activations_page_size_bytes", static_cast<uint32_t>(aligned_token_activations_page_size_bytes)},
        {"source_token_segment_buffer_size_bytes", token_segment_buffer_size_bytes},
        {"source_expert_block_size_bytes", expert_token_segment_buffer_block_size_bytes},
        {"token_size_bytes", token_size_bytes},
        {"dense_token_maps_stride_elm", dense_token_maps_stride_elm},
        {"alignment", l1_alignment},
        {"num_devices", num_devices_total},
        {"src_chip_id", src_chip_id},
        {"mesh_rows", static_cast<uint32_t>(mesh_view.num_rows())},
        {"mesh_cols", static_cast<uint32_t>(mesh_view.num_cols())},
        {"fabric_max_packet_size_bytes", max_packet_size_bytes},
        {"linearized_mesh_coord", flat_mesh_idx},
        {"topology", static_cast<uint32_t>(topology)},
        {"num_mux_workers_per_link", static_cast<uint32_t>(neighbors.size())},
        {"compute_sync_semaphore_id", writer_compute_sync_semaphore_id},
        {"compute_cores_per_combine_core", compute_cores_per_combine_cores},
        {"double_buffer_source", static_cast<uint32_t>(compute_cores_by_ring_id.has_value())}};

    std::vector<uint32_t> writer_compile_time_args;
    ttnn::ccl::fabric_mux_connection_ct_args(
        num_data_parallel_cores * num_token_parallel_cores,
        tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
        mux_kernel_config,
        writer_compile_time_args);
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    using operations::ccl::common::stringify;
    std::vector<std::pair<std::string, std::string>> writer_defines = {
        {"DEST_CHIP_ID", stringify(dest_chip_id)},
        {"DEST_MESH_ID", stringify(dest_mesh_id)},
        {"DIRECTIONS", stringify(directions)}};

    writer_defines.emplace_back("REPLICATE_GROUP_AXIS", std::to_string(axis));

    // kCclWriterKernelIdx on the standalone program (reader, mux, writer).
    const uint32_t writer_kernel_idx = desc.kernels.size();
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source =
            "ttnn/cpp/ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/kernels/dataflow/writer.cpp",
        .core_ranges = needed_worker_core_range_set,
        .compile_time_args = std::move(writer_compile_time_args),
        .named_compile_time_args = std::move(writer_named_ct_args),
        .defines = std::move(writer_defines),
        .config =
            DataMovementConfigDescriptor{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::NOC_1,
                .noc_mode = NOC_MODE::DM_DYNAMIC_NOC,
            },
    });

    const auto termination_master_semaphore_id =
        detail::allocate_worker_semaphore(desc, needed_worker_core_range_set, 0);

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
        const uint32_t is_init_sync_core_arg = static_cast<uint32_t>(is_init_sync_core);

        const auto source_token_segment_size_bytes = *(data_parallel_size_iter++);
        auto writer_rt = make_writer_runtime_args(
            source_token_segment_size_bytes, dest_token_segment_offset_bytes, is_init_sync_core_arg);

        // if the input is double buffered, coming from fused moe_compute, add the core coordinates of the compute cores
        // which get semaphore increments upon release of buffer segment.
        append_compute_core_coords(writer_rt, compute_cores_by_ring_iter);

        const bool is_termination_master = (sender_core == *termination_master_core_iter);
        std::vector<uint32_t> mux_connection_args;
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
                desc,
                mesh_device->worker_core_from_logical_core(*termination_master_core_iter),
                mux_connection_args,
                termination_master_semaphore_id);
        }
        writer_rt.append(mux_connection_args);

        // termination master is responsible for tearing down mux workers for given link, needs their coordinates
        if (is_termination_master) {
            std::vector<uint32_t> termination_args;
            detail::add_termination_master_rt_args(*core_map_iter, termination_args);
            writer_rt.append(termination_args);
        }

        desc.kernels[reader_kernel_idx].emplace_runtime_args(
            sender_core, make_reader_runtime_args(token_parallel_idx, is_init_sync_core_arg));
        desc.kernels[writer_kernel_idx].emplace_runtime_args(sender_core, writer_rt);

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
                (*compute_cores_by_ring_iter) += compute_cores_per_combine_cores;
            }
        }

        if (++link_worker_idx == num_workers_per_link) {
            link_worker_idx = 0;
            ++core_map_iter;
            ++termination_master_core_iter;
        }
    }
}

void UnifiedSelectReduce::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*coord*/) {
    if (!operation_attributes.optional_cross_device_semaphore.has_value()) {
        return;
    }
    const uint32_t writer_kernel_idx =
        operation_attributes.local_combine ? kLocalCombineWriterKernelIdx : kCclWriterKernelIdx;
    const uint32_t cross_device_semaphore_addr =
        static_cast<uint32_t>(operation_attributes.optional_cross_device_semaphore->address());
    auto& writer_args_by_core = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_idx);
    for (auto& writer_args_column : writer_args_by_core) {
        for (auto& writer_args : writer_args_column) {
            if (writer_args.size() == 0) {
                continue;
            }
            writer_args[kWriterCrossDeviceSemaphoreArgIdx] = cross_device_semaphore_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
