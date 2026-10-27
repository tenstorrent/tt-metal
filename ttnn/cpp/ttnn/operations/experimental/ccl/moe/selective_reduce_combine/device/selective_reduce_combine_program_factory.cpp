// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <utility>
#include <vector>

#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/allocator.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include "ttnn/distributed/types.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_op_fusion.hpp"
#include "ttnn/operations/ccl/common/host/moe_utils.hpp"
#include "ttnn/operations/experimental/ccl/moe/selective_reduce_combine/device/selective_reduce_combine_program_factory.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/global_semaphore.hpp"

namespace ttnn::experimental::prim {

using ttnn::experimental::ccl::add_semaphore_descriptor;

namespace detail {

FusedSourceBufferLayout compute_fused_source_buffer_layout(
    const uint32_t source_shard_height,
    const uint32_t source_shard_width,
    const uint32_t token_segment_width,
    const uint32_t source_buffer_size_bytes,
    const uint32_t token_segment_size_bytes,
    const uint32_t num_buffers) {
    TT_FATAL(source_shard_height > 0, "Fused source shard height must be positive");
    TT_FATAL(source_shard_width > 0, "Fused source shard width must be positive");
    TT_FATAL(token_segment_width > 0, "Fused source token segment width must be positive");
    TT_FATAL(source_buffer_size_bytes > 0, "Fused source buffer size must be positive");
    TT_FATAL(token_segment_size_bytes > 0, "Fused source token segment size must be positive");
    TT_FATAL(num_buffers > 0, "Fused source buffer count must be positive");
    TT_FATAL(
        source_shard_height % num_buffers == 0,
        "Fused source shard height ({}) must be divisible by the buffer count ({})",
        source_shard_height,
        num_buffers);
    TT_FATAL(
        source_shard_width % token_segment_width == 0,
        "Fused source shard width ({}) must be divisible by the token segment width ({})",
        source_shard_width,
        token_segment_width);
    TT_FATAL(
        token_segment_size_bytes % token_segment_width == 0,
        "Fused source token segment size ({} B) must be a whole number of bytes per element over its width ({})",
        token_segment_size_bytes,
        token_segment_width);

    // A ring entry is source_shard_height / num_buffers shard rows; each shard row is
    // source_shard_width / token_segment_width token segments. Producer and writer both
    // step through the shard in token-segment rows, so that is the unit of rows_per_buffer.
    const uint64_t segments_per_shard_row = source_shard_width / token_segment_width;
    const uint64_t rows_per_buffer = (source_shard_height / num_buffers) * segments_per_shard_row;
    const uint64_t buffer_block_size_bytes = rows_per_buffer * token_segment_size_bytes;
    const uint64_t circular_buffer_size_bytes = buffer_block_size_bytes * num_buffers;
    const uint64_t element_size_bytes = token_segment_size_bytes / token_segment_width;
    const uint64_t shard_size_bytes = uint64_t(source_shard_height) * source_shard_width * element_size_bytes;
    TT_FATAL(
        circular_buffer_size_bytes == shard_size_bytes,
        "Fused source ring ({} B) must cover the shard exactly ({} B)",
        circular_buffer_size_bytes,
        shard_size_bytes);
    TT_FATAL(
        circular_buffer_size_bytes <= source_buffer_size_bytes,
        "Fused source circular buffer size ({} B) exceeds its L1 buffer bank ({} B)",
        circular_buffer_size_bytes,
        source_buffer_size_bytes);
    TT_FATAL(
        circular_buffer_size_bytes <= std::numeric_limits<uint32_t>::max(),
        "Fused source circular buffer size ({} B) exceeds uint32_t",
        circular_buffer_size_bytes);

    return {
        .rows_per_buffer = static_cast<uint32_t>(rows_per_buffer),
        .buffer_block_size_bytes = static_cast<uint32_t>(buffer_block_size_bytes),
        .circular_buffer_size_bytes = static_cast<uint32_t>(circular_buffer_size_bytes),
    };
}

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

    TT_FATAL(
        token_size_bytes == 0,
        "selective_reduce_combine: {} data-parallel core(s) cannot cover the token: {} bytes left over "
        "(each segment is limited to {} bytes by the fabric packet size); increase num_data_parallel_cores",
        num_data_parallel_cores,
        token_size_bytes,
        max_segment_size_bytes);

    return data_parallel_sizes_bytes;
}

// Writer runtime args for a double-buffered (fused moe_compute) source: the number of compute
// cores feeding this combine core, followed by their physical (x, y) coordinates.
void append_compute_core_rt_args(
    const MeshDevice& mesh_device,
    const std::vector<CoreCoord>& compute_cores,
    tt::tt_metal::KernelDescriptor::RTArgList& rt_args) {
    rt_args.push_back(static_cast<uint32_t>(compute_cores.size()));
    for (const auto& core : compute_cores) {
        const auto physical = mesh_device.worker_core_from_logical_core(core);
        rt_args.push_back(physical.x);
        rt_args.push_back(physical.y);
    }
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
        local_combine ? std::numeric_limits<uint32_t>::max()
                      : tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(*input_tensor.device());
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

// Fabric Mux V2 memory map fit: the largest per-channel buffer count whose mux L1 map ends at or below
// usable_l1_end_address (the lowest occupied L1 address, or the floor of L1_SMALL).
std::pair<tt::tt_fabric::FabricMuxV2Config, uint8_t> get_fabric_mux_config(
    const MeshDevice& mesh_device,
    const uint8_t num_channels,
    uint8_t num_buffers_per_channel,
    const size_t channel_buffer_size_bytes,
    const uint32_t l1_unreserved_base_address,
    const size_t usable_l1_end_address) {
    // Each candidate is sized without a ceiling, so this loop does the shrinking; the returned config carries the
    // ceiling and its constructor asserts the fit.
    while (true) {
        TT_FATAL(
            num_buffers_per_channel > 0,
            "Not enough L1 space for mux core memory requirements given current occupancy. Likely too many experts "
            "per device");
        const tt::tt_fabric::FabricMuxV2Config candidate(
            mesh_device, num_channels, num_buffers_per_channel, channel_buffer_size_bytes, l1_unreserved_base_address);
        if (candidate.get_memory_map_end_address() <= usable_l1_end_address) {
            break;
        }
        --num_buffers_per_channel;
    }
    return {
        tt::tt_fabric::FabricMuxV2Config(
            mesh_device,
            num_channels,
            num_buffers_per_channel,
            channel_buffer_size_bytes,
            l1_unreserved_base_address,
            usable_l1_end_address),
        num_buffers_per_channel};
}

struct MuxCorePlacement {
    CoreCoord logical_core;
    tt::tt_fabric::FabricNodeId dst_node_id;
    uint32_t link = 0;
};

// Plans Fabric Mux V2 cores without pushing kernels. Kernels are added later, after the writer, so the
// writer index stays reader+1 (each mux core contributes a forwarder and a manager).
auto plan_mux_workers(
    const MeshDevice& mesh_device,
    const CoreRangeSet& mux_core_range_set,
    const std::vector<ttnn::MeshCoordinate>& neighbors,
    const uint32_t num_links,
    const uint32_t num_workers) {
    TT_FATAL(
        num_links > 0 && num_workers % num_links == 0,
        "num_workers ({}) must be divisible by num_links ({})",
        num_workers,
        num_links);
    const uint32_t num_workers_per_link = num_workers / num_links;
    // One stream register per V2 logical channel; the FabricMuxV2Config ctor enforces the same bound.
    TT_FATAL(
        num_workers_per_link > 0 && num_workers_per_link <= 64,
        "Fabric Mux V2 supports 1..64 clients per mux core, got {} workers per link",
        num_workers_per_link);

    constexpr uint8_t initial_num_buffers_per_channel = 15;

    const size_t channel_buffer_size_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(mesh_device);
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

    auto [mux_config, num_buffers_per_channel] = get_fabric_mux_config(
        mesh_device,
        static_cast<uint8_t>(num_workers_per_link),
        initial_num_buffers_per_channel,
        channel_buffer_size_bytes,
        l1_unreserved_base_address,
        usable_l1_end_address);
    log_debug(
        tt::LogOp,
        "selective_reduce_combine fabric mux v2: {} channels x {} buffers x {} B, memory map [{}, {}) under {}, {} "
        "mux cores (num_links={} x neighbors={})",
        num_workers_per_link,
        num_buffers_per_channel,
        channel_buffer_size_bytes,
        l1_unreserved_base_address,
        mux_config.get_memory_map_end_address(),
        usable_l1_end_address,
        num_links * neighbors.size(),
        num_links,
        neighbors.size());

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

    std::vector<std::map<ttnn::MeshCoordinate, CoreCoord>> mux_neigbor_core_maps;
    mux_neigbor_core_maps.reserve(num_links);
    std::vector<MuxCorePlacement> placements;
    placements.reserve(needed_cores);

    const auto mux_cores = corerange_to_cores(needed_mux_core_range_set);
    auto mux_core_iter = mux_cores.begin();
    for (uint32_t link = 0; link < num_links; ++link) {
        std::map<ttnn::MeshCoordinate, CoreCoord> mux_neigbor_core_map;
        for (const auto& neighbor_coord : neighbors) {
            auto mux_logical_core = *(mux_core_iter++);
            const auto mux_virtual_core = mesh_device.worker_core_from_logical_core(mux_logical_core);
            const auto dst_node_id = mesh_device.get_fabric_node_id(neighbor_coord);
            placements.push_back(MuxCorePlacement{
                .logical_core = mux_logical_core,
                .dst_node_id = dst_node_id,
                .link = link,
            });
            mux_neigbor_core_map[neighbor_coord] = mux_virtual_core;
        }
        mux_neigbor_core_maps.push_back(mux_neigbor_core_map);
    }

    return std::make_tuple(mux_config, std::move(mux_neigbor_core_maps), std::move(placements));
}

void add_mux_worker_kernels(
    tt::tt_metal::ProgramDescriptor& desc,
    const tt::tt_fabric::FabricMuxV2Config& mux_config,
    const tt::tt_fabric::FabricNodeId& src_node_id,
    const std::vector<MuxCorePlacement>& placements) {
    for (const auto& placement : placements) {
        // Forwarder (RISCV_0) + manager (RISCV_1) on this core. The forwarder runs on NoC 1. On NoC 0 (the API
        // default) its posted VC 1 stream shares the read-request path of moe_compute's expert cores and
        // back-pressures their DRAM weight reads; on a Wormhole Galaxy that hung dm0 in
        // noc_async_read_barrier_with_trid (see moe_compute dm0.cpp).
        tt::tt_fabric::add_fabric_mux_v2_to_program(
            desc,
            mux_config,
            placement.logical_core,
            src_node_id,
            placement.dst_node_id,
            placement.link,
            /*forwarder_noc=*/tt::tt_metal::NOC::RISCV_1_default);
    }
}

}  // namespace detail
namespace {

// Standalone kernel order when append starts with no kernels: reader 0, writer 1, then Fabric Mux V2
// kernels on the CCL path. The writer stays at index 1 so override_runtime_arguments can patch it.
constexpr uint32_t kWriterKernelIdx = 1;
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
            add_semaphore_descriptor(desc, CoreRangeSet(worker_core_range_set.bounding_box()), 1);
        const uint32_t compute_sync_semaphore_id = add_semaphore_descriptor(desc, worker_core_range_set, 0);
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
    const std::optional<std::vector<std::vector<CoreCoord>>>& compute_cores_by_combine_column) {
    using namespace tt::tt_metal;
    using namespace tt::tt_fabric;
    using namespace ttnn::ccl;

    // 0 when the caller has no semaphore (fused moe_compute SingleDevice path: the writer
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
    const bool double_buffer_source = compute_cores_by_combine_column.has_value();

    // physical experts per device, replicated shared experts are counted per device
    const uint32_t experts_per_device = dense_token_maps_tensor.logical_shape()[0];

    const auto input_dtype = input_tensor.dtype();
    const auto& dense_token_maps_tensor_spec = dense_token_maps_tensor.tensor_spec();

    // In local combine mode, there is no fabric packet-size constraint.
    const auto fabric_max_packet_size_bytes = operation_attributes.local_combine
                                                  ? std::numeric_limits<uint32_t>::max()
                                                  : get_tt_fabric_channel_buffer_size_bytes(*mesh_device);
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

    // TODO (AFM) this is an ugly kludge until we can get GPT-OSS on the mainline op #43645
    uint32_t expert_token_segment_buffer_block_size_bytes;
    uint32_t buffer_size_bytes;
    if (double_buffer_source) {
        constexpr uint32_t num_buffers = 2;
        const auto& source_shard_shape = input_tensor.memory_config().shard_spec()->shape;
        // Block stride in token-segment rows (hidden_size / num_data_parallel_cores wide), the unit
        // the writer and moe_compute's dm1 both step in; must match the producer's call.
        const auto fused_source_layout = detail::compute_fused_source_buffer_layout(
            source_shard_shape[0],
            source_shard_shape[1],
            hidden_size / num_data_parallel_cores,
            input_tensor.buffer()->aligned_size_per_bank(),
            token_segment_buffer_size_bytes,
            num_buffers);
        expert_token_segment_buffer_block_size_bytes = fused_source_layout.buffer_block_size_bytes;
        buffer_size_bytes = fused_source_layout.circular_buffer_size_bytes;
    } else {
        expert_token_segment_buffer_block_size_bytes =
            token_segment_buffer_size_bytes * total_tokens / num_token_parallel_cores;
        buffer_size_bytes = expert_token_segment_buffer_block_size_bytes * experts_per_device;
    }

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

    // Reader is pushed here; the writer is the next kernel on both paths (kWriterKernelIdx when this
    // append starts empty). CCL mux kernels follow the writer.
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
            {"double_buffer_source", static_cast<uint32_t>(double_buffer_source)},
        };

        std::vector<uint32_t> writer_compile_time_args;
        TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

        // Writer follows the reader. On a program that started empty this is kWriterKernelIdx.
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
        auto compute_cores_by_column_iter = (compute_cores_by_combine_column.has_value())
                                                ? std::make_optional(compute_cores_by_combine_column->cbegin())
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
            if (compute_cores_by_column_iter.has_value()) {
                detail::append_compute_core_rt_args(*mesh_device, **compute_cores_by_column_iter, writer_rt);
            }
            desc.kernels[writer_kernel_idx].emplace_runtime_args(sender_core, writer_rt);

            if (data_parallel_size_iter == data_parallel_sizes_bytes.cend()) {
                data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
                dest_token_segment_offset_bytes = 0;
                ++token_parallel_idx;
                if (compute_cores_by_column_iter.has_value()) {
                    compute_cores_by_column_iter = std::make_optional(compute_cores_by_combine_column->cbegin());
                }
            } else {
                dest_token_segment_offset_bytes += source_token_segment_size_bytes;
                if (compute_cores_by_column_iter.has_value()) {
                    ++(*compute_cores_by_column_iter);
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

    // Plan mux cores now; their kernels are pushed after the writer so the writer index stays stable.
    const auto [mux_config, mux_neigbor_core_maps, mux_placements] = detail::plan_mux_workers(
        *mesh_device, mux_core_range_set, neighbors, num_links, num_worker_cores);

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
        {"compute_sync_semaphore_id", writer_compute_sync_semaphore_id},
        {"double_buffer_source", static_cast<uint32_t>(double_buffer_source)}};

    // Positional CT args: TensorAccessorArgs only. The V2 fabric mux client is built entirely from runtime args.
    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_time_args);

    using operations::ccl::common::stringify;
    std::vector<std::pair<std::string, std::string>> writer_defines = {
        {"DEST_CHIP_ID", stringify(dest_chip_id)},
        {"DEST_MESH_ID", stringify(dest_mesh_id)},
        {"DIRECTIONS", stringify(directions)}};

    writer_defines.emplace_back("REPLICATE_GROUP_AXIS", std::to_string(axis));

    // Writer follows the reader. Mux kernels are pushed after the runtime-arg loop.
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

    // Intra-link barrier: worker 0 of each link is the link sync core. With Mux V2 there is no termination
    // master: the mux terminates by itself after its last client closes.
    const auto link_sync_semaphore_id = add_semaphore_descriptor(desc, needed_worker_core_range_set, 0);

    // V2 client semaphores (flow control + teardown), one pair per direction, shared ids across worker cores.
    std::vector<tt::tt_fabric::FabricMuxV2Config::ClientSemaphores> client_semaphores;
    client_semaphores.reserve(neighbors.size());
    for (std::size_t d = 0; d < neighbors.size(); ++d) {
        client_semaphores.push_back(tt::tt_fabric::FabricMuxV2Config::ClientSemaphores{
            .flow_control_sem_id = add_semaphore_descriptor(desc, needed_worker_core_range_set, 0),
            .teardown_sem_id = add_semaphore_descriptor(desc, needed_worker_core_range_set, 0),
        });
    }

    uint32_t link_idx = 0, link_worker_idx = 0, token_parallel_idx = 0, dest_token_segment_offset_bytes = 0;
    auto core_map_iter = mux_neigbor_core_maps.cbegin();
    auto data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
    auto compute_cores_by_column_iter = (compute_cores_by_combine_column.has_value())
                                            ? std::make_optional(compute_cores_by_combine_column->cbegin())
                                            : std::nullopt;
    for (const auto& sender_core : sender_cores) {
        const bool is_init_sync_core = sender_core == sender_cores.at(0);
        const uint32_t is_init_sync_core_arg = static_cast<uint32_t>(is_init_sync_core);

        const auto source_token_segment_size_bytes = *(data_parallel_size_iter++);
        auto writer_rt = make_writer_runtime_args(
            source_token_segment_size_bytes, dest_token_segment_offset_bytes, is_init_sync_core_arg);

        // if the input is double buffered, coming from fused moe_compute, add the core coordinates of the compute cores
        // which get semaphore increments upon release of buffer segment.
        if (compute_cores_by_column_iter.has_value()) {
            detail::append_compute_core_rt_args(*mesh_device, **compute_cores_by_column_iter, writer_rt);
        }

        // Link sync args: is_link_sync_core, link_sync_semaphore_id, sync core virtual x, y.
        const bool is_link_sync_core = (link_worker_idx == 0);
        const auto link_sync_virtual_core =
            mesh_device->worker_core_from_logical_core(sender_cores.at(link_idx * num_workers_per_link));
        std::vector<uint32_t> mux_connection_args;
        mux_connection_args.push_back(static_cast<uint32_t>(is_link_sync_core));
        mux_connection_args.push_back(link_sync_semaphore_id);
        mux_connection_args.push_back(link_sync_virtual_core.x);
        mux_connection_args.push_back(link_sync_virtual_core.y);

        // This worker is logical channel link_worker_idx on each mux of its link. append_client_connection_rt_args
        // serializes the 11 args FabricMuxV2Sender::build_from_args consumes.
        for (std::size_t d = 0; d < neighbors.size(); ++d) {
            const auto& mux_virtual_core = core_map_iter->at(neighbors[d]);
            mux_config.append_client_connection_rt_args(
                mux_virtual_core, static_cast<uint8_t>(link_worker_idx), client_semaphores[d], mux_connection_args);
        }
        writer_rt.append(mux_connection_args);

        desc.kernels[reader_kernel_idx].emplace_runtime_args(
            sender_core, make_reader_runtime_args(token_parallel_idx, is_init_sync_core_arg));
        desc.kernels[writer_kernel_idx].emplace_runtime_args(sender_core, writer_rt);

        if (data_parallel_size_iter == data_parallel_sizes_bytes.cend()) {
            data_parallel_size_iter = data_parallel_sizes_bytes.cbegin();
            dest_token_segment_offset_bytes = 0;
            ++token_parallel_idx;
            if (compute_cores_by_column_iter.has_value()) {
                compute_cores_by_column_iter = std::make_optional(compute_cores_by_combine_column->cbegin());
            }
        } else {
            dest_token_segment_offset_bytes += source_token_segment_size_bytes;
            if (compute_cores_by_column_iter.has_value()) {
                ++(*compute_cores_by_column_iter);
            }
        }

        if (++link_worker_idx == num_workers_per_link) {
            link_worker_idx = 0;
            ++core_map_iter;
            ++link_idx;
        }
    }

    detail::add_mux_worker_kernels(desc, mux_config, fabric_node_id, mux_placements);
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
    const uint32_t writer_kernel_idx = kWriterKernelIdx;
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
