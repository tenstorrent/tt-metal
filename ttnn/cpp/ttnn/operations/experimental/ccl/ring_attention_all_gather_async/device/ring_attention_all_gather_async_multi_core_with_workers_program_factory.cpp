// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ring_attention_all_gather_async_multi_core_with_workers_program_factory.hpp"
#include "ring_attention_all_gather_async_device_operation_types.hpp"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "cpp/ttnn/operations/ccl/common/types/ccl_types_args_emitters.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_command_stream_builders.hpp"
#include "cpp/ttnn/operations/ccl/common/uops/command_lowering.hpp"
#include "cpp/ttnn/operations/ccl/common/host/ccl_worker_builder.hpp"
#include "cpp/ttnn/operations/ccl/common/host/command_backend_runtime_args_overrider.hpp"
#include <optional>
#include <tuple>

namespace ttnn::experimental::prim {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

// Per-coord ProgramDescriptor build. Pulled into an anonymous-namespace helper so
// create_workload_descriptor() can loop coords and reuse this body verbatim. The
// CMake-provided namespace keeps file-local names distinct in unity builds.
tt::tt_metal::ProgramDescriptor build_ring_attention_all_gather_program_descriptor(
    const RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::operation_attributes_t& operation_attributes,
    const RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::tensor_args_t& tensor_args,
    RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinate& mesh_coordinate) {
    tt::tt_metal::ProgramDescriptor desc;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> empty_fused_op_signaler;
    log_debug(tt::LogOp, "DEBUG: build_ring_attention_all_gather_program_descriptor is called");

    uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
        tensor_args.input_tensor[0], mesh_coordinate, operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor[0],
        mesh_coordinate,
        1,
        operation_attributes.topology,
        operation_attributes.cluster_axis);

    std::optional<MeshCoordinate> backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
        tensor_args.input_tensor[0],
        mesh_coordinate,
        -1,
        operation_attributes.topology,
        operation_attributes.cluster_axis);

    ring_attention_all_gather_async_multi_core_with_workers_helper(
        desc,
        tensor_args.input_tensor,
        mesh_coordinate,
        forward_coord,
        backward_coord,
        tensor_return_value,
        operation_attributes.dim,
        operation_attributes.num_links,
        operation_attributes.ring_size,
        device_index,
        operation_attributes.topology,
        operation_attributes.semaphore,
        operation_attributes.sub_device_id,
        empty_fused_op_signaler);

    return desc;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

// Returns a WorkloadDescriptor with one ProgramDescriptor per coord: device_index /
// forward_coord / backward_coord all depend on the mesh coordinate, so descriptors
// cannot be shared across coords.
tt::tt_metal::WorkloadDescriptor
RingAttentionAllGatherAsyncMultiCoreWithWorkersProgramFactory::create_workload_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor wd;
    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());
    for (const auto& coord : coords) {
        auto desc = CMAKE_UNIQUE_NAMESPACE::build_ring_attention_all_gather_program_descriptor(
            operation_attributes, tensor_args, tensor_return_value, coord);
        wd.programs.push_back({ttnn::MeshCoordinateRange(coord), std::move(desc)});
    }
    return wd;
}

}  // namespace ttnn::experimental::prim

namespace ttnn {

namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

constexpr uint32_t kBatchDimension = 0;
constexpr uint32_t kHeadDimension = 1;
constexpr uint32_t kSequenceDimension = 2;
constexpr uint32_t kWidthDimension = 3;

// Independent performance tunables. Keep these named rather than deriving one from another: the reader prefetch
// window, writer header pool, and Fabric mux credit window protect different pipelines.
constexpr uint32_t kDirectPrefetchPackets = 4;
constexpr uint32_t kMuxedPrefetchPackets = 8;
constexpr uint32_t kPacketHeaderSlots = 8;
constexpr uint32_t kDoubleBufferingFactor = 2;
constexpr uint32_t kScatterWritePages = 2;
constexpr uint32_t kMuxBuffersPerChannel = 2;

constexpr uint32_t kSingleWorkerPerLink = 1;
constexpr uint32_t kSingleMuxClient = 1;
constexpr uint32_t kFlatCoreAllocationGroupCount = 1;
constexpr uint32_t kContiguousPagesAdvanced = 1;
constexpr uint32_t kRingDirectionAlternationPeriod = 2;
constexpr uint32_t kFabricMuxTerminationSemaphoreArgIndex = 10;
constexpr bool kForwardDirection = true;
constexpr bool kBackwardDirection = false;

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

bool ring_attention_all_gather_async_detail::supports_output_bank_owned_schedule(
    const std::vector<Tensor>& output_tensors, int32_t dim, uint32_t transport_page_size) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    if (dim != kSequenceDimension || output_tensors.empty() || transport_page_size == 0) {
        return false;
    }
    const auto* device = output_tensors.front().device();
    const uint32_t num_dram_banks = device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    return num_dram_banks > 0 &&
           std::all_of(
               output_tensors.begin(),
               output_tensors.end(),
               [transport_page_size, device](const Tensor& tensor) {
                   return tensor.device() == device && tensor.buffer()->is_dram() &&
                          tensor.buffer()->buffer_layout() == tt::tt_metal::TensorMemoryLayout::INTERLEAVED &&
                          tensor.buffer()->aligned_page_size() == transport_page_size;
               }) &&
           tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / transport_page_size >= kScatterWritePages;
}

bool ring_attention_all_gather_async_detail::supports_muxed_multiworker_placement(
    const std::vector<Tensor>& output_tensors,
    int32_t dim,
    uint32_t transport_page_size,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    CoreCoord core_grid_offset,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy) {
    using namespace CMAKE_UNIQUE_NAMESPACE;

    if (!supports_output_bank_owned_schedule(output_tensors, dim, transport_page_size)) {
        return false;
    }
    auto* device = output_tensors.front().device();
    if (device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM) < kMuxedMultiworkerMinDramBanks) {
        return false;
    }
    const auto sub_device = sub_device_id.value_or(device->get_sub_device_ids().at(0));
    if (device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device).num_cores() <
        kMuxedMultiworkerNumReservedCores) {
        return false;
    }
    const auto selection = ttnn::ccl::try_choose_worker_cores(
        /*num_links=*/kFlatCoreAllocationGroupCount,
        kMuxedMultiworkerNumReservedCores,
        device,
        sub_device_id,
        core_grid_offset,
        std::nullopt,
        core_allocation_strategy);
    return selection.cores.size() == kMuxedMultiworkerNumReservedCores && selection.all_placeable();
}

void ring_attention_neighbor_halo_exchange_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const std::vector<Tensor>& input_tensors,
    const MeshCoordinate& target_device_coord,
    const MeshCoordinate& transport_device_coord,
    const MeshCoordinate& unicast_destination_coord,
    std::vector<Tensor>& output_tensors,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphores,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const ttnn::experimental::ccl::AllGatherFusedOpSignaler& fused_op_signaler,
    CoreCoord core_grid_offset,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    std::optional<uint32_t> input_batch_slice_idx,
    std::optional<uint32_t> gather_valid_Ht,
    const RingAttentionNeighborHaloConfig& halo) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ReaderConfigDescriptor;
    using tt::tt_metal::SemaphoreDescriptor;
    using tt::tt_metal::WriterConfigDescriptor;

    TT_FATAL(!input_tensors.empty() && input_tensors.size() == output_tensors.size(), "Invalid halo tensor list");
    TT_FATAL(!semaphores.empty(), "Neighbor halo requires an incoming-ready semaphore");

    auto* mesh_device = input_tensors.front().device();
    const bool wrap_endpoint = ring_index == 0 || ring_index + 1 == ring_size;
    const auto transport_topology = wrap_endpoint ? topology : ttnn::ccl::Topology::Linear;
    auto mutable_input_tensors = input_tensors;
    const auto op_config = ttnn::ccl::CCLOpConfig(mutable_input_tensors, output_tensors, transport_topology);
    auto unicast_forward_args = std::get<0>(ccl::get_forward_backward_line_unicast_configuration(
        target_device_coord, unicast_destination_coord, std::nullopt, mesh_device));
    if (tt::tt_fabric::is_1d_fabric_config(tt::tt_fabric::GetFabricConfig())) {
        unicast_forward_args[1] = halo.unicast_hops;
    }

    const auto [worker_core_range, worker_cores] = ttnn::ccl::choose_worker_cores(
        num_links,
        kSingleWorkerPerLink,
        mesh_device,
        sub_device_id,
        core_grid_offset,
        std::nullopt,
        core_allocation_strategy);
    const CoreRangeSet workers(worker_core_range);

    const uint32_t page_size = op_config.get_page_size();
    const uint32_t packet_buffer_bytes = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t pages_per_packet = std::min(packet_buffer_bytes / page_size, kScatterWritePages);
    const uint32_t cb_pages = kDoubleBufferingFactor * kDirectPrefetchPackets * pages_per_packet;
    const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.front().dtype());
    constexpr uint32_t data_cb = tt::CB::c_in2;
    constexpr uint32_t packet_header_cb = tt::CB::c_in1;
    const uint32_t packet_header_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_pages * page_size,
        .core_ranges = workers,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(data_cb), .data_format = data_format, .page_size = page_size}}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = kPacketHeaderSlots * packet_header_bytes * kDoubleBufferingFactor,
        .core_ranges = workers,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(packet_header_cb),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_bytes}}},
    });

    const uint32_t num_inputs = input_tensors.size();
    KernelDescriptor reader_kernel{};
    reader_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_neighbor_halo_reader.cpp";
    reader_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel.core_ranges = workers;
    reader_kernel.config = WriterConfigDescriptor{};
    reader_kernel.compile_time_args = {
        ring_index, ring_size, data_cb, pages_per_packet, page_size, num_inputs, kDirectPrefetchPackets};
    for (uint32_t input = 0; input < num_inputs; ++input) {
        reader_kernel.compile_time_args.push_back(page_size);
    }
    for (const auto& input : input_tensors) {
        tt::tt_metal::TensorAccessorArgs(input.buffer()).append_to(reader_kernel.compile_time_args);
    }

    KernelDescriptor writer_kernel{};
    writer_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_neighbor_halo_writer.cpp";
    writer_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel.core_ranges = workers;
    writer_kernel.config = ReaderConfigDescriptor{};
    writer_kernel.compile_time_args = {
        packet_header_cb,
        data_cb,
        pages_per_packet,
        page_size,
        num_inputs,
        unicast_forward_args[0],
        unicast_forward_args[1],
        static_cast<uint32_t>(halo.send_backward),
    };
    for (uint32_t input = 0; input < num_inputs; ++input) {
        writer_kernel.compile_time_args.push_back(page_size);
    }
    for (const auto& output : output_tensors) {
        tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_kernel.compile_time_args);
    }

    auto halo_signaler = fused_op_signaler;
    const auto worker_list = corerange_to_cores(workers, std::nullopt, true);
    if (worker_list.size() > 1) {
        const uint32_t sem_id = desc.semaphores.size();
        desc.semaphores.push_back(SemaphoreDescriptor{
            .id = sem_id,
            .core_type = tt::CoreType::WORKER,
            .core_ranges = workers,
            .initial_value = 0,
        });
        halo_signaler.all_gather_worker_sync_semaphore = sem_id;
    }
    halo_signaler.all_gather_worker_cores_noc.clear();
    for (const auto& worker : worker_list) {
        halo_signaler.all_gather_worker_cores_noc.push_back(mesh_device->worker_core_from_logical_core(worker));
    }
    halo_signaler.initialized_all_gather = true;

    for (uint32_t link = 0; link < num_links; ++link) {
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(static_cast<uint32_t>(
            semaphores.front().address()));  // smuggled-rta-ok: persistent GlobalSemaphore address
        KernelDescriptor::RTArgList writer_args;
        const CoreCoord worker_physical = mesh_device->worker_core_from_logical_core(worker_cores[link]);
        writer_args.push_back(worker_physical.x);
        writer_args.push_back(worker_physical.y);
        writer_args.push_back(static_cast<uint32_t>(
            semaphores.front().address()));  // smuggled-rta-ok: persistent GlobalSemaphore address

        for (uint32_t input = 0; input < num_inputs; ++input) {
            const auto input_shape = input_tensors[input].padded_shape();
            const auto output_shape = output_tensors[input].padded_shape();
            const uint32_t input_heads = input_shape[kHeadDimension];
            const uint32_t input_Wt = input_shape[kWidthDimension] / tt::constants::TILE_WIDTH;
            const uint32_t input_Ht = input_shape[kSequenceDimension] / tt::constants::TILE_HEIGHT;
            const uint32_t output_Wt = output_shape[kWidthDimension] / tt::constants::TILE_WIDTH;
            const uint32_t output_Ht = output_shape[kSequenceDimension] / tt::constants::TILE_HEIGHT;
            TT_FATAL(
                output_Wt == input_Wt,
                "Neighbor halo requires matching input/output tile widths, got input Wt={} and output Wt={}",
                input_Wt,
                output_Wt);
            TT_FATAL(
                output_Ht >= halo.send_to_next_count_Ht,
                "Neighbor halo output has {} tile rows but requires {}",
                output_Ht,
                halo.send_to_next_count_Ht);
            TT_FATAL(
                halo.send_to_next_start_Ht <= input_Ht &&
                    halo.send_to_next_count_Ht <= input_Ht - halo.send_to_next_start_Ht,
                "Neighbor halo [{}, {}) exceeds input Ht={}",
                halo.send_to_next_start_Ht,
                halo.send_to_next_start_Ht + halo.send_to_next_count_Ht,
                input_Ht);

            const uint32_t range_start_page = halo.send_to_next_start_Ht * input_Wt;
            const uint32_t range_page_count = halo.send_to_next_count_Ht * input_Wt;
            const uint32_t valid_pages = std::min(gather_valid_Ht.value_or(input_Ht), input_Ht) * input_Wt;
            TT_FATAL(
                range_start_page <= valid_pages && range_page_count <= valid_pages - range_start_page,
                "Neighbor halo [{}, {}) exceeds the valid per-head page prefix {}",
                range_start_page,
                range_start_page + range_page_count,
                valid_pages);
            TT_FATAL(
                range_page_count >= num_links,
                "Neighbor halo has {} pages for {} links; every worker must send at least one page",
                range_page_count,
                num_links);
            const uint32_t pages_per_worker = range_page_count / num_links;
            const uint32_t remainder = range_page_count % num_links;
            const uint32_t input_tile_start = range_start_page + link * pages_per_worker + std::min(link, remainder);
            const uint32_t input_tile_end =
                range_start_page + (link + 1) * pages_per_worker + std::min(link + 1, remainder);
            TT_FATAL(
                !input_batch_slice_idx.has_value() || *input_batch_slice_idx < input_shape[kBatchDimension],
                "input_batch_slice_idx={} out of range for input batch={}",
                input_batch_slice_idx.value_or(0),
                input_shape[kBatchDimension]);
            const uint32_t batch_head_count =
                input_batch_slice_idx.has_value() ? input_heads : input_shape[kBatchDimension] * input_heads;
            const uint32_t input_batch_base = ttnn::ring_attention_all_gather_async_detail::input_batch_base_pages(
                input_batch_slice_idx.value_or(0), input_heads, input_Ht, input_Wt);

            reader_args.push_back(input_Ht * input_Wt);
            reader_args.push_back(batch_head_count);
            reader_args.push_back(input_tile_start);
            reader_args.push_back(input_tile_end);
            reader_args.push_back(input_batch_base);

            writer_args.push_back(output_Ht * output_Wt);
            writer_args.push_back(batch_head_count);
            writer_args.push_back(input_tile_start);
            writer_args.push_back(input_tile_end);
            writer_args.push_back(range_start_page);
        }
        for (const auto& input : input_tensors) {
            reader_args.push_back(input.buffer());
        }
        std::vector<uint32_t> signaler_args;
        halo_signaler.push_all_gather_fused_op_rt_args(signaler_args, num_links, link, 0);
        reader_args.append(signaler_args);
        reader_kernel.emplace_runtime_args(worker_cores[link], reader_args);

        for (const auto& output : output_tensors) {
            writer_args.push_back(output.buffer());
        }
        std::vector<uint32_t> fabric_args;
        // FabricConnectionManager consumes flags in forward-then-backward order. The linear
        // wrap sender uses the backward connection, so its sender descriptor must follow the
        // backward flag instead of the forward one.
        writer_args.push_back(halo.send_backward ? 0u : 1u);
        if (halo.send_backward) {
            writer_args.push_back(1u);
        }
        tt::tt_fabric::append_fabric_connection_rt_args(
            mesh_device->get_fabric_node_id(target_device_coord),
            mesh_device->get_fabric_node_id(transport_device_coord),
            link,
            desc,
            worker_cores[link],
            fabric_args);
        writer_args.append(fabric_args);
        if (!halo.send_backward) {
            writer_args.push_back(0u);
        }
        writer_kernel.emplace_runtime_args(worker_cores[link], writer_args);
    }

    desc.kernels.push_back(std::move(reader_kernel));
    desc.kernels.push_back(std::move(writer_kernel));
}

void ring_attention_all_gather_async_multi_core_with_workers_helper(
    tt::tt_metal::ProgramDescriptor& desc,
    const std::vector<Tensor>& input_tensor,
    const MeshCoordinate& target_device_coord,
    std::optional<MeshCoordinate> forward_device_coord,
    std::optional<MeshCoordinate> backward_device_coord,
    std::vector<Tensor>& output_tensor,
    int32_t dim,
    uint32_t num_links,
    uint32_t ring_size,
    uint32_t ring_index,
    ttnn::ccl::Topology topology,
    const std::vector<GlobalSemaphore>& semaphore,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& fused_op_signaler,
    const CoreCoord core_grid_offset,
    ttnn::ccl::CoreAllocationStrategy core_allocation_strategy,
    bool enable_muxed_multiworker,
    bool enable_output_bank_owned_schedule,
    std::optional<uint32_t> input_batch_slice_idx,
    std::optional<uint32_t> gather_valid_Ht,
    std::optional<Tensor> slot_id,
    std::optional<Tensor> kv_actual_isl,
    uint32_t chunk_local_tiles,
    uint32_t kv_cache_num_layers,
    uint32_t kv_cache_layer_idx) {
    using namespace CMAKE_UNIQUE_NAMESPACE;
    using tt::tt_metal::CBDescriptor;
    using tt::tt_metal::CBFormatDescriptor;
    using tt::tt_metal::DataMovementConfigDescriptor;
    using tt::tt_metal::KernelDescriptor;
    using tt::tt_metal::ReaderConfigDescriptor;
    using tt::tt_metal::SemaphoreDescriptor;
    using tt::tt_metal::WriterConfigDescriptor;

    auto* mesh_device = input_tensor[0].device();
    [[maybe_unused]] const bool is_first_chip = ring_index == 0;
    [[maybe_unused]] const bool is_last_chip = ring_index == ring_size - 1;
    log_trace(
        tt::LogOp,
        "DEBUG: device: {}, is_first_chip: {}, is_last_chip: {}",
        input_tensor.at(0).device()->id(),
        is_first_chip,
        is_last_chip);

    /* All gather fusion */
    const bool fuse_op = fused_op_signaler.has_value();

    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_sender_workers;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_forward;
    std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler> fused_op_signaler_backward;

    if (fuse_op) {
        fused_op_signaler_backward = fused_op_signaler.value();
        fused_op_signaler_sender_workers = fused_op_signaler.value();
        fused_op_signaler_forward = fused_op_signaler.value();
    }

    // Get OP Config, topology config
    std::vector<Tensor> input_tensors = input_tensor;
    const std::vector<Tensor>& output_tensors = output_tensor;
    const auto& op_config = ttnn::ccl::CCLOpConfig(input_tensors, output_tensors, topology);
    auto [unicast_forward_args, unicast_backward_args] = ccl::get_forward_backward_line_unicast_configuration(
        target_device_coord, forward_device_coord, backward_device_coord, mesh_device);
    auto [num_targets_forward, num_targets_backward, dynamic_alternate] =
        ttnn::ccl::get_forward_backward_configuration(ring_size, ring_index, topology);
    if (topology == ttnn::ccl::Topology::Ring && ring_index % kRingDirectionAlternationPeriod == 0) {
        std::swap(num_targets_forward, num_targets_backward);
    }
    // L1 Scratch CB Creation
    const uint32_t max_payload_size_bytes = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    const uint32_t l1_scratch_cb_page_size_bytes = op_config.get_page_size();
    const uint32_t num_dram_banks = mesh_device->allocator()->get_num_banks(tt::tt_metal::BufferType::DRAM);
    const bool output_bank_owned_schedule = enable_output_bank_owned_schedule &&
                                            ring_attention_all_gather_async_detail::supports_output_bank_owned_schedule(
                                                output_tensor, dim, l1_scratch_cb_page_size_bytes);

    struct WorkerPlacement {
        CoreCoord core;
        uint32_t link;
        uint32_t worker_index_in_link;
        uint32_t workers_on_link;
    };
    enum MuxedCoreSlot : std::size_t {
        kForwardLink1Mux,
        kForwardLink1Worker0,
        kForwardLink1Worker1,
        kForwardLink0Mux,
        kForwardLink0Worker0,
        kBackwardLink1Mux,
        kBackwardLink1Worker0,
        kBackwardLink1Worker1,
        kBackwardLink0Mux,
        kBackwardLink0Worker0,
        kNumMuxedCoreSlots,
    };
    static_assert(kNumMuxedCoreSlots == ring_attention_all_gather_async_detail::kMuxedMultiworkerNumReservedCores);

    // The muxed schedule uses six CCL workers and four Fabric muxes. Each direction gets a doubled link and
    // a single-worker link; both directions double link 1 because alternating the doubled link adds about 3%
    // to the measured 512K critical path, while this placement also avoids the long tail of the mirrored
    // link-0 placement. Putting every link behind a mux keeps all four router links on one credit/teardown
    // protocol. The caller enables this only when the forwarded payload amortizes mux startup and teardown.
    if (enable_muxed_multiworker) {
        TT_FATAL(
            ring_attention_all_gather_async_detail::supports_muxed_multiworker_placement(
                output_tensor,
                dim,
                l1_scratch_cb_page_size_bytes,
                sub_device_id,
                core_grid_offset,
                core_allocation_strategy) &&
                fuse_op && topology == ttnn::ccl::Topology::Ring && forward_device_coord.has_value() &&
                backward_device_coord.has_value() &&
                num_links == ring_attention_all_gather_async_detail::kMuxedMultiworkerNumLinks,
            "Invalid configuration for the ring-attention muxed multi-worker schedule");
    }
    const uint32_t num_reserved_ccl_cores =
        enable_muxed_multiworker ? ring_attention_all_gather_async_detail::kMuxedMultiworkerNumReservedCores
                                 : num_links * ring_attention_all_gather_async_detail::kRingDirectionCount;
    auto worker_core_selection = ttnn::ccl::choose_worker_cores(
        kFlatCoreAllocationGroupCount,
        num_reserved_ccl_cores,
        mesh_device,
        sub_device_id,
        core_grid_offset,
        std::nullopt,
        core_allocation_strategy);
    auto& sender_reserved_cores = std::get<1>(worker_core_selection);
    TT_FATAL(
        sender_reserved_cores.size() == num_reserved_ccl_cores,
        "Ring attention requested {} CCL cores but only {} are available in the selected worker region",
        num_reserved_ccl_cores,
        sender_reserved_cores.size());

    std::vector<WorkerPlacement> forward_workers;
    std::vector<WorkerPlacement> backward_workers;
    std::vector<CoreCoord> mux_cores;
    if (enable_muxed_multiworker) {
        mux_cores = {
            sender_reserved_cores[kForwardLink0Mux],
            sender_reserved_cores[kForwardLink1Mux],
            sender_reserved_cores[kBackwardLink0Mux],
            sender_reserved_cores[kBackwardLink1Mux],
        };
        forward_workers = {
            {.core = sender_reserved_cores[kForwardLink0Worker0],
             .link = 0,
             .worker_index_in_link = 0,
             .workers_on_link = 1},
            {.core = sender_reserved_cores[kForwardLink1Worker0],
             .link = 1,
             .worker_index_in_link = 0,
             .workers_on_link = ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink},
            {.core = sender_reserved_cores[kForwardLink1Worker1],
             .link = 1,
             .worker_index_in_link = 1,
             .workers_on_link = ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink},
        };
        backward_workers = {
            {.core = sender_reserved_cores[kBackwardLink0Worker0],
             .link = 0,
             .worker_index_in_link = 0,
             .workers_on_link = 1},
            {.core = sender_reserved_cores[kBackwardLink1Worker0],
             .link = 1,
             .worker_index_in_link = 0,
             .workers_on_link = ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink},
            {.core = sender_reserved_cores[kBackwardLink1Worker1],
             .link = 1,
             .worker_index_in_link = 1,
             .workers_on_link = ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink},
        };
    } else {
        const auto direct_core_slot = [](uint32_t link, bool is_forward) {
            return link * ring_attention_all_gather_async_detail::kRingDirectionCount +
                   static_cast<uint32_t>(is_forward);
        };
        for (uint32_t link = 0; link < num_links; ++link) {
            backward_workers.push_back(WorkerPlacement{
                .core = sender_reserved_cores[direct_core_slot(link, kBackwardDirection)],
                .link = link,
                .worker_index_in_link = 0,
                .workers_on_link = kSingleWorkerPerLink});
            forward_workers.push_back(WorkerPlacement{
                .core = sender_reserved_cores[direct_core_slot(link, kForwardDirection)],
                .link = link,
                .worker_index_in_link = 0,
                .workers_on_link = kSingleWorkerPerLink});
        }
    }

    const auto placements_to_core_ranges = [](const std::vector<WorkerPlacement>& placements) {
        std::set<CoreRange> ranges;
        for (const auto& placement : placements) {
            ranges.insert(CoreRange(placement.core));
        }
        return CoreRangeSet(ranges);
    };
    const CoreRangeSet sender_forward_core_ranges = placements_to_core_ranges(forward_workers);
    const CoreRangeSet sender_backward_core_ranges = placements_to_core_ranges(backward_workers);

    tt::tt_fabric::FabricMuxConfig mux_config(
        /*num_full_size_channels=*/ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink,
        /*num_header_only_channels=*/0,
        /*num_buffers_full_size_channel=*/kMuxBuffersPerChannel,
        /*num_buffers_header_only_channel=*/0,
        /*buffer_size_bytes_full_size_channel=*/tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes(),
        /*base_l1_address=*/mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1));
    if (enable_muxed_multiworker) {
        TT_FATAL(
            mux_config.get_memory_map_end_address() <= mesh_device->l1_size_per_core(),
            "Ring attention Fabric mux requires L1 through address {:#x}, but each Tensix core has only {:#x} bytes",
            mux_config.get_memory_map_end_address(),
            mesh_device->l1_size_per_core());
    }
    const auto append_writer_fabric_compile_time_args =
        [&](KernelDescriptor& kernel, bool use_worker_mux, uint32_t num_mux_clients) {
            kernel.compile_time_args.push_back(static_cast<uint32_t>(use_worker_mux));
            ttnn::ccl::fabric_mux_connection_ct_args(
                num_mux_clients,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_config,
                kernel.compile_time_args);
        };
    const auto split_mux_worker_ranges = [](const std::vector<WorkerPlacement>& placements) {
        std::set<CoreRange> multi_client_ranges;
        std::set<CoreRange> single_client_ranges;
        for (const auto& placement : placements) {
            auto& ranges = placement.workers_on_link > 1 ? multi_client_ranges : single_client_ranges;
            ranges.insert(CoreRange(placement.core));
        }
        TT_FATAL(
            multi_client_ranges.size() == ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink &&
                single_client_ranges.size() == kSingleWorkerPerLink,
            "Ten-core mux schedule requires {} multi-client workers and one single-client worker per direction",
            ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink);
        return std::pair{CoreRangeSet(multi_client_ranges), CoreRangeSet(single_client_ranges)};
    };
    const uint32_t num_pages_per_packet =
        output_bank_owned_schedule
            ? max_payload_size_bytes / l1_scratch_cb_page_size_bytes
            : std::min(max_payload_size_bytes / l1_scratch_cb_page_size_bytes, kScatterWritePages);
    const uint32_t prefetch_packets = enable_muxed_multiworker ? kMuxedPrefetchPackets : kDirectPrefetchPackets;
    // Must be >= kDoubleBufferingFactor * prefetch_packets * num_pages_per_packet for deadlock-free buffering
    // (see PREFETCH_PACKETS in ring_attention_all_gather_reader.cpp).
    const uint32_t cb_num_pages = kDoubleBufferingFactor * prefetch_packets * num_pages_per_packet;
    const tt::DataFormat df = tt::tt_metal::datatype_to_dataformat_converter(input_tensor[0].dtype());

    // CBs for transferring data between sender_reader and sender_writer
    uint32_t sender_forward_cb_index = tt::CB::c_in0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_forward_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sender_forward_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });

    uint32_t sender_backward_cb_index = tt::CB::c_in2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_pages * l1_scratch_cb_page_size_bytes,
        .core_ranges = sender_backward_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(sender_backward_cb_index),
            .data_format = df,
            .page_size = l1_scratch_cb_page_size_bytes,
        }}},
    });

    // Set aside a buffer we can use for storing packet headers in (particularly for atomic incs)
    const auto reserved_packet_header_forward_CB_index = tt::CB::c_in1;
    const auto packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    desc.cbs.push_back(CBDescriptor{
        .total_size = kPacketHeaderSlots * packet_header_size_bytes * kDoubleBufferingFactor,
        .core_ranges = sender_forward_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_forward_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    const auto reserved_packet_header_backward_CB_index = tt::CB::c_in1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = kPacketHeaderSlots * packet_header_size_bytes * kDoubleBufferingFactor,
        .core_ranges = sender_backward_core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(reserved_packet_header_backward_CB_index),
            .data_format = tt::DataFormat::RawUInt32,
            .page_size = packet_header_size_bytes,
        }}},
    });

    // The host value is a structural placeholder for the indexed gather. On the
    // trace-safe path the reader derives the actual cache slot from slot_id.
    const bool has_metadata = slot_id.has_value();
    const uint32_t meta_cb_index = tt::CB::c_in3;
    if (has_metadata) {
        const uint32_t meta_cb_page_size_bytes = kv_actual_isl->buffer()->page_size();
        TT_FATAL(
            meta_cb_page_size_bytes >= sizeof(uint32_t),
            "Ring attention metadata CB page has {} bytes; a uint32 scalar requires at least {}",
            meta_cb_page_size_bytes,
            sizeof(uint32_t));
        for (const auto& core_ranges : {sender_forward_core_ranges, sender_backward_core_ranges}) {
            desc.cbs.push_back(CBDescriptor{
                .total_size = meta_cb_page_size_bytes,
                .core_ranges = core_ranges,
                .format_descriptors = {{CBFormatDescriptor{
                    .buffer_index = static_cast<uint8_t>(meta_cb_index),
                    .data_format = tt::DataFormat::RawUInt32,
                    .page_size = meta_cb_page_size_bytes,
                }}},
            });
        }
    }

    // Tensor Info
    const uint32_t num_inputs = input_tensor.size();
    constexpr const char* exchange_writer_kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_writer.cpp";

    // KERNEL CREATION
    // Forward Direction
    // Reader
    KernelDescriptor sender_reader_forward_kernel{};
    sender_reader_forward_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp";
    sender_reader_forward_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_reader_forward_kernel.core_ranges = sender_forward_core_ranges;
    sender_reader_forward_kernel.config = WriterConfigDescriptor{};
    sender_reader_forward_kernel.compile_time_args = {
        ring_index,                       // my_chip_id
        sender_forward_cb_index,          // cb_forward_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_slices_forward_direction
        num_targets_backward,             // num_slices_backward_direction
        static_cast<uint32_t>(topology),  // topology
        kContiguousPagesAdvanced,         // contig_pages_advanced
        num_inputs,                       // num_inputs
        kForwardDirection,                // direction
        fuse_op,                          // fused op
        static_cast<uint32_t>(has_metadata),
        meta_cb_index,
        num_links,
        static_cast<uint32_t>(output_bank_owned_schedule),
        num_dram_banks,
        prefetch_packets,
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_reader_forward_kernel.compile_time_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(input_tensor[i].buffer())
            .append_to(sender_reader_forward_kernel.compile_time_args);
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_reader_forward_kernel.compile_time_args);
    }
    if (has_metadata) {
        tt::tt_metal::TensorAccessorArgs(slot_id->buffer()).append_to(sender_reader_forward_kernel.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(kv_actual_isl->buffer())
            .append_to(sender_reader_forward_kernel.compile_time_args);
    }

    // Writer
    KernelDescriptor sender_writer_forward_kernel{};
    sender_writer_forward_kernel.kernel_source = exchange_writer_kernel_source;
    sender_writer_forward_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_writer_forward_kernel.core_ranges = sender_forward_core_ranges;
    sender_writer_forward_kernel.config = ReaderConfigDescriptor{};
    sender_writer_forward_kernel.compile_time_args = {
        ring_index,                               // my_chip_id
        reserved_packet_header_forward_CB_index,  // reserved_packet_header_cb_id
        kPacketHeaderSlots,                       // num_packet_headers_storable
        sender_forward_cb_index,                  // cb_forward_id
        num_pages_per_packet,                     // packet_size_in_pages
        op_config.get_page_size(),                // tensor0_page_size
        num_targets_forward,                      // num_targets_forward_direction
        num_targets_backward,                     // num_targets_backward_direction
        dynamic_alternate,                        // alternate
        fuse_op,                                  // fused op
        static_cast<uint32_t>(topology),          // topology
        kContiguousPagesAdvanced,                 // contig_pages_advanced
        num_inputs,                               // num_inputs
        kForwardDirection,                        // direction
        unicast_backward_args[0],                 // unicast route arg0 (dst_mesh_id or 0)
        unicast_backward_args[1],                 // unicast route arg1 (dst_chip_id or distance_in_hops)
        static_cast<uint32_t>(has_metadata),
        meta_cb_index,
        num_links,
        static_cast<uint32_t>(output_bank_owned_schedule),
        num_dram_banks,
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_writer_forward_kernel.compile_time_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_writer_forward_kernel.compile_time_args);
    }
    if (has_metadata) {
        tt::tt_metal::TensorAccessorArgs(kv_actual_isl->buffer())
            .append_to(sender_writer_forward_kernel.compile_time_args);
    }
    std::optional<KernelDescriptor> sender_writer_forward_single_mux_kernel;
    if (enable_muxed_multiworker) {
        const auto [multi_client_ranges, single_client_ranges] = split_mux_worker_ranges(forward_workers);
        sender_writer_forward_single_mux_kernel = sender_writer_forward_kernel;
        sender_writer_forward_single_mux_kernel->core_ranges = single_client_ranges;
        append_writer_fabric_compile_time_args(
            *sender_writer_forward_single_mux_kernel, /*use_worker_mux=*/true, kSingleMuxClient);

        sender_writer_forward_kernel.core_ranges = multi_client_ranges;
        append_writer_fabric_compile_time_args(
            sender_writer_forward_kernel,
            /*use_worker_mux=*/true,
            ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink);
    } else {
        append_writer_fabric_compile_time_args(
            sender_writer_forward_kernel, /*use_worker_mux=*/false, kSingleMuxClient);
    }

    // Backward Direction
    // Reader
    KernelDescriptor sender_reader_backward_kernel{};
    sender_reader_backward_kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/ring_attention_all_gather_async/device/kernels/"
        "ring_attention_all_gather_reader.cpp";
    sender_reader_backward_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_reader_backward_kernel.core_ranges = sender_backward_core_ranges;
    sender_reader_backward_kernel.config = WriterConfigDescriptor{};
    sender_reader_backward_kernel.compile_time_args = {
        ring_index,                       // my_chip_id
        sender_backward_cb_index,         // cb_backward_id
        num_pages_per_packet,             // packet_size_in_pages
        op_config.get_page_size(),        // tensor0_page_size
        num_targets_forward,              // num_slices_forward_direction
        num_targets_backward,             // num_slices_backward_direction
        static_cast<uint32_t>(topology),  // topology
        kContiguousPagesAdvanced,         // contig_pages_advanced
        num_inputs,                       // num_inputs
        kBackwardDirection,               // direction
        fuse_op,                          // fused op
        static_cast<uint32_t>(has_metadata),
        meta_cb_index,
        num_links,
        static_cast<uint32_t>(output_bank_owned_schedule),
        num_dram_banks,
        prefetch_packets,
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_reader_backward_kernel.compile_time_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(input_tensor[i].buffer())
            .append_to(sender_reader_backward_kernel.compile_time_args);
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_reader_backward_kernel.compile_time_args);
    }
    if (has_metadata) {
        tt::tt_metal::TensorAccessorArgs(slot_id->buffer()).append_to(sender_reader_backward_kernel.compile_time_args);
        tt::tt_metal::TensorAccessorArgs(kv_actual_isl->buffer())
            .append_to(sender_reader_backward_kernel.compile_time_args);
    }

    // Writer
    KernelDescriptor sender_writer_backward_kernel{};
    sender_writer_backward_kernel.kernel_source = exchange_writer_kernel_source;
    sender_writer_backward_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
    sender_writer_backward_kernel.core_ranges = sender_backward_core_ranges;
    sender_writer_backward_kernel.config = ReaderConfigDescriptor{};
    sender_writer_backward_kernel.compile_time_args = {
        ring_index,                                // my_chip_id
        reserved_packet_header_backward_CB_index,  // reserved_packet_header_cb_id
        kPacketHeaderSlots,                        // num_packet_headers_storable
        sender_backward_cb_index,                  // cb_backward_id
        num_pages_per_packet,                      // packet_size_in_pages
        op_config.get_page_size(),                 // tensor0_page_size
        num_targets_forward,                       // num_targets_forward_direction
        num_targets_backward,                      // num_targets_backward_direction
        dynamic_alternate,                         // alternate
        fuse_op,                                   // fused op
        static_cast<uint32_t>(topology),           // topology
        kContiguousPagesAdvanced,                  // contig_pages_advanced
        num_inputs,                                // num_inputs
        kBackwardDirection,                        // direction
        unicast_forward_args[0],                   // unicast route arg0 (dst_mesh_id or 0)
        unicast_forward_args[1],                   // unicast route arg1 (dst_chip_id or distance_in_hops)
        static_cast<uint32_t>(has_metadata),
        meta_cb_index,
        num_links,
        static_cast<uint32_t>(output_bank_owned_schedule),
        num_dram_banks,
    };
    for (uint32_t i = 0; i < num_inputs; i++) {
        sender_writer_backward_kernel.compile_time_args.push_back(op_config.get_page_size());
    }
    for (uint32_t i = 0; i < num_inputs; i++) {
        tt::tt_metal::TensorAccessorArgs(output_tensor[i].buffer())
            .append_to(sender_writer_backward_kernel.compile_time_args);
    }
    if (has_metadata) {
        tt::tt_metal::TensorAccessorArgs(kv_actual_isl->buffer())
            .append_to(sender_writer_backward_kernel.compile_time_args);
    }

    std::optional<KernelDescriptor> sender_writer_backward_single_mux_kernel;
    if (enable_muxed_multiworker) {
        const auto [multi_client_ranges, single_client_ranges] = split_mux_worker_ranges(backward_workers);
        sender_writer_backward_single_mux_kernel = sender_writer_backward_kernel;
        sender_writer_backward_single_mux_kernel->core_ranges = single_client_ranges;
        append_writer_fabric_compile_time_args(
            *sender_writer_backward_single_mux_kernel, /*use_worker_mux=*/true, kSingleMuxClient);

        sender_writer_backward_kernel.core_ranges = multi_client_ranges;
        append_writer_fabric_compile_time_args(
            sender_writer_backward_kernel,
            /*use_worker_mux=*/true,
            ring_attention_all_gather_async_detail::kMuxedMultiworkerMaxWorkersPerLink);
    } else {
        append_writer_fabric_compile_time_args(
            sender_writer_backward_kernel, /*use_worker_mux=*/false, kSingleMuxClient);
    }

    /* All gather fusion */
    // Inline equivalent of AllGatherFusedOpSignaler::init_all_gather for the descriptor
    // pattern. The original init_all_gather mutates a Program; here we instead append
    // to `desc.semaphores` and update the signaler's noc-coord and semaphore-id state
    // directly. Semaphore IDs are sequential and start at the current desc.semaphores.size().
    if (fuse_op) {
        auto sender_workers_backward = corerange_to_cores(sender_backward_core_ranges, std::nullopt, true);

        auto init_all_gather_descriptor =
            [&](std::optional<ttnn::experimental::ccl::AllGatherFusedOpSignaler>& signaler,
                const CoreRangeSet& workers_range,
                const std::vector<CoreCoord>& worker_cores) {
                // Mirror AllGatherFusedOpSignaler::init_all_gather: only allocate the sync semaphore
                // when there is more than one worker core (otherwise no inter-worker sync is needed).
                if (worker_cores.size() > 1) {
                    const uint32_t sem_id = static_cast<uint32_t>(desc.semaphores.size());
                    desc.semaphores.push_back(SemaphoreDescriptor{
                        .id = sem_id,
                        .core_type = tt::CoreType::WORKER,
                        .core_ranges = workers_range,
                        .initial_value = 0,
                    });
                    signaler->all_gather_worker_sync_semaphore = sem_id;
                }
                signaler->all_gather_worker_cores_noc.clear();
                for (const auto& core : worker_cores) {
                    signaler->all_gather_worker_cores_noc.push_back(mesh_device->worker_core_from_logical_core(core));
                }
                signaler->initialized_all_gather = true;
            };

        init_all_gather_descriptor(fused_op_signaler_backward, sender_backward_core_ranges, sender_workers_backward);
        auto sender_workers_forward = corerange_to_cores(sender_forward_core_ranges, std::nullopt, true);
        init_all_gather_descriptor(fused_op_signaler_forward, sender_forward_core_ranges, sender_workers_forward);
        init_all_gather_descriptor(
            fused_op_signaler_sender_workers, sender_forward_core_ranges, sender_workers_forward);
    }
    // Kernel Runtime Args
    const auto build_tensor_descriptor_args = [&](const WorkerPlacement& placement) {
        std::vector<uint32_t> tensor_descriptor_args;
        for (uint32_t i = 0; i < num_inputs; i++) {
            const auto input_tensor_shape = input_tensor[i].padded_shape();
            const auto output_tensor_shape = output_tensor[i].padded_shape();
            const uint32_t num_heads = input_tensor_shape[kHeadDimension];
            // single_batch_head_num_pages is always pages-per-(batch,head); independent of slicing.
            const uint32_t full_batch_head_size = input_tensor_shape[kBatchDimension] * num_heads;

            const uint32_t input_tensor_Wt = input_tensor_shape[kWidthDimension] / tt::constants::TILE_WIDTH;
            const uint32_t input_tensor_Ht = input_tensor_shape[kSequenceDimension] / tt::constants::TILE_HEIGHT;
            const uint32_t output_tensor_Wt = output_tensor_shape[kWidthDimension] / tt::constants::TILE_WIDTH;
            const uint32_t output_tensor_Ht = output_tensor_shape[kSequenceDimension] / tt::constants::TILE_HEIGHT;
            TT_ASSERT(!(input_tensor_shape[kWidthDimension] % tt::constants::TILE_WIDTH));
            TT_ASSERT(!(output_tensor_shape[kWidthDimension] % tt::constants::TILE_WIDTH));

            // TensorAccessor page IDs address the logical padded tensor volume. For ND-sharded buffers,
            // buffer()->num_pages() can additionally include physical padding in the final shard; treating
            // those pages as tensor data would overrun a selected batch/head slice and can write past the
            // corresponding logical output band.
            const uint32_t single_batch_head_num_pages = input_tensor_Ht * input_tensor_Wt;

            // Single-slot gather: read only slot `input_batch_slice_idx`'s `num_heads` blocks (from
            // `input_batch_base`); the writer emits them to output slot 0. A batch-1 output suffices,
            // but a full-batch output also works (only slot 0 written). std::nullopt => full batch.
            uint32_t batch_head_size = full_batch_head_size;
            uint32_t input_batch_base = 0;
            if (input_batch_slice_idx.has_value()) {
                TT_FATAL(
                    *input_batch_slice_idx < input_tensor_shape[kBatchDimension],
                    "input_batch_slice_idx={} out of range for input batch={}",
                    *input_batch_slice_idx,
                    input_tensor_shape[kBatchDimension]);
                batch_head_size = num_heads;
                input_batch_base = ttnn::ring_attention_all_gather_async_detail::input_batch_base_pages(
                    *input_batch_slice_idx, num_heads, input_tensor_Ht, input_tensor_Wt);
            }

            tensor_descriptor_args.push_back(input_tensor_Wt);   // 0 == input_tensor_Wt
            tensor_descriptor_args.push_back(input_tensor_Ht);   // 1 == input_tensor_Ht
            tensor_descriptor_args.push_back(output_tensor_Wt);  // 2 == output_tensor_Wt
            tensor_descriptor_args.push_back(output_tensor_Ht);  // 3 == output_tensor_Ht
            tensor_descriptor_args.push_back(batch_head_size);   // 4 == batch_head_size (bh-loop count)
            // The device derives the actual per-link range from valid_pages. These are retained as
            // structural placeholders for the descriptor ABI and are overwritten before use.
            tensor_descriptor_args.push_back(0u);                           // 5 == input_tile_id_start
            tensor_descriptor_args.push_back(single_batch_head_num_pages);  // 6 == input_tile_id_end
            tensor_descriptor_args.push_back(input_batch_base);  // 7 == input_batch_base (phase-1 input page offset)
            // 8 == valid pages per (batch,head) to gather. Default: full input slab (no clamp). When
            // gather_valid_Ht is set (fused ring_joint_sdpa with an oversized cache), bound it to the
            // first gather_valid_Ht tile-rows so only kv_actual-sized data moves. The fused path also
            // re-patches this per dispatch on cache hits (apply_ring_joint_scalar_runtime_args); setting
            // it here makes the cache-miss (first) dispatch bounded too.
            const uint32_t valid_pages_per_batch_head =
                gather_valid_Ht.has_value() ? std::min(*gather_valid_Ht, input_tensor_Ht) * input_tensor_Wt
                                            : single_batch_head_num_pages;
            tensor_descriptor_args.push_back(valid_pages_per_batch_head);      // 8 == valid_pages_per_batch_head
            tensor_descriptor_args.push_back(placement.link);                  // 9 == worker_link
            tensor_descriptor_args.push_back(placement.worker_index_in_link);  // 10 == worker_index_in_link
            tensor_descriptor_args.push_back(placement.workers_on_link);       // 11 == workers_on_link
        }
        return tensor_descriptor_args;
    };

    const auto forward_signaler_cores = corerange_to_cores(sender_forward_core_ranges, std::nullopt, true);
    const auto backward_signaler_cores = corerange_to_cores(sender_backward_core_ranges, std::nullopt, true);
    const auto signaler_index = [](const std::vector<CoreCoord>& cores, const CoreCoord& core) {
        const auto it = std::find(cores.begin(), cores.end(), core);
        TT_FATAL(it != cores.end(), "Ring attention worker core is missing from its signaler core range");
        return static_cast<uint32_t>(std::distance(cores.begin(), it));
    };
    std::vector<std::optional<uint32_t>> forward_mux_termination_sem_ids(num_links);
    std::vector<std::optional<uint32_t>> backward_mux_termination_sem_ids(num_links);

    const auto emit_worker_runtime_args = [&](const WorkerPlacement& placement, bool is_forward) {
        const auto tensor_descriptor_args = build_tensor_descriptor_args(placement);
        const uint32_t sem_index = is_forward ? 1 : 0;
        const auto& direction_signaler_cores = is_forward ? forward_signaler_cores : backward_signaler_cores;
        const uint32_t worker_signaler_index = signaler_index(direction_signaler_cores, placement.core);

        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(static_cast<uint32_t>(dim));
        reader_args.push_back(ring_size);
        reader_args.push_back(static_cast<uint32_t>(
            semaphore.at(sem_index).address()));  // smuggled-rta-ok: persistent GlobalSemaphore address
        reader_args.append(tensor_descriptor_args);
        for (uint32_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            reader_args.push_back(input_tensor[input_idx].buffer());
        }
        for (uint32_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            reader_args.push_back(output_tensor[input_idx].buffer());
        }
        if (has_metadata) {
            reader_args.push_back(slot_id->buffer());
            reader_args.push_back(kv_actual_isl->buffer());
            reader_args.push_back(chunk_local_tiles);
            reader_args.push_back(kv_cache_num_layers);
            reader_args.push_back(kv_cache_layer_idx);
        }
        if (fuse_op) {
            std::vector<uint32_t> signaler_args;
            auto& signaler = is_forward ? fused_op_signaler_forward : fused_op_signaler_backward;
            signaler->push_all_gather_fused_op_rt_args(
                signaler_args,
                static_cast<uint32_t>(direction_signaler_cores.size()),
                worker_signaler_index,
                is_forward ? 1 : 0);
            reader_args.append(signaler_args);
        }
        auto& reader_kernel = is_forward ? sender_reader_forward_kernel : sender_reader_backward_kernel;
        reader_kernel.emplace_runtime_args(placement.core, reader_args);

        const CoreCoord worker_virtual_core = mesh_device->worker_core_from_logical_core(placement.core);
        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(static_cast<uint32_t>(dim));
        writer_args.push_back(static_cast<uint32_t>(worker_virtual_core.x));
        writer_args.push_back(static_cast<uint32_t>(worker_virtual_core.y));
        writer_args.push_back(ring_size);
        writer_args.push_back(static_cast<uint32_t>(
            semaphore.at(sem_index).address()));  // smuggled-rta-ok: persistent GlobalSemaphore address
        writer_args.append(tensor_descriptor_args);
        for (uint32_t input_idx = 0; input_idx < num_inputs; ++input_idx) {
            writer_args.push_back(output_tensor[input_idx].buffer());
        }
        if (has_metadata) {
            writer_args.push_back(kv_actual_isl->buffer());
            writer_args.push_back(chunk_local_tiles);
        }

        const auto neighbor_coord = is_forward ? backward_device_coord : forward_device_coord;
        std::vector<uint32_t> writer_extra_args;
        if (enable_muxed_multiworker) {
            const CoreCoord mux_logical_core =
                is_forward ? mux_cores[placement.link] : mux_cores[num_links + placement.link];
            const CoreCoord mux_virtual_core = mesh_device->worker_core_from_logical_core(mux_logical_core);
            const auto& direction_workers = is_forward ? forward_workers : backward_workers;
            const auto master_it =
                std::find_if(direction_workers.begin(), direction_workers.end(), [&](const WorkerPlacement& candidate) {
                    return candidate.link == placement.link && candidate.worker_index_in_link == 0;
                });
            TT_FATAL(master_it != direction_workers.end(), "Fabric mux group has no termination master");
            const CoreCoord termination_master_virtual_core =
                mesh_device->worker_core_from_logical_core(master_it->core);
            auto& termination_sem_ids = is_forward ? forward_mux_termination_sem_ids : backward_mux_termination_sem_ids;
            const bool is_termination_master = placement.worker_index_in_link == 0;
            if (!is_termination_master) {
                TT_FATAL(
                    termination_sem_ids[placement.link].has_value(),
                    "Fabric mux termination slave was emitted before its master");
            }
            ttnn::ccl::fabric_mux_connection_rt_args(
                neighbor_coord.has_value(),
                is_termination_master,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                mux_virtual_core,
                placement.worker_index_in_link,
                placement.core,
                mux_config,
                desc,
                termination_master_virtual_core,
                writer_extra_args,
                termination_sem_ids[placement.link]);
            if (is_termination_master) {
                TT_FATAL(
                    writer_extra_args.size() > kFabricMuxTerminationSemaphoreArgIndex,
                    "Fabric mux runtime arguments omitted the termination semaphore");
                termination_sem_ids[placement.link] = writer_extra_args[kFabricMuxTerminationSemaphoreArgIndex];
            }
        } else {
            if (is_forward) {
                writer_args.push_back(0u);
            }
            writer_args.push_back(static_cast<uint32_t>(neighbor_coord.has_value()));
            if (neighbor_coord.has_value()) {
                tt::tt_fabric::append_fabric_connection_rt_args(
                    mesh_device->get_fabric_node_id(target_device_coord),
                    mesh_device->get_fabric_node_id(*neighbor_coord),
                    placement.link,
                    desc,
                    placement.core,
                    writer_extra_args);
            }
            if (!is_forward) {
                writer_extra_args.push_back(0u);
            }
        }
        if (fuse_op) {
            if (is_forward) {
                fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(
                    writer_extra_args, static_cast<uint32_t>(forward_signaler_cores.size()), worker_signaler_index, 1);
            } else {
                // Backward writers never issue the local-slice signal, but retain the writer ABI.
                fused_op_signaler_sender_workers->push_all_gather_fused_op_rt_args(writer_extra_args, 1, 0, 0);
            }
        }
        writer_args.append(writer_extra_args);
        if (enable_muxed_multiworker && placement.workers_on_link == kSingleWorkerPerLink) {
            auto& single_mux_kernel =
                is_forward ? sender_writer_forward_single_mux_kernel : sender_writer_backward_single_mux_kernel;
            TT_FATAL(single_mux_kernel.has_value(), "Missing single-client Fabric mux writer kernel");
            single_mux_kernel->emplace_runtime_args(placement.core, writer_args);
        } else {
            auto& writer_kernel = is_forward ? sender_writer_forward_kernel : sender_writer_backward_kernel;
            writer_kernel.emplace_runtime_args(placement.core, writer_args);
        }
    };

    for (const auto& placement : forward_workers) {
        emit_worker_runtime_args(placement, true);
    }
    for (const auto& placement : backward_workers) {
        emit_worker_runtime_args(placement, false);
    }

    // Kernel descriptors are pushed last, with their runtime args fully populated.
    // The descriptor framework allocates KernelHandles when materializing the
    // descriptor into a Program; runtime-arg auto-patching on cache hits removes
    // the need to expose those handles back to the caller.
    desc.kernels.push_back(std::move(sender_reader_forward_kernel));
    desc.kernels.push_back(std::move(sender_writer_forward_kernel));
    desc.kernels.push_back(std::move(sender_reader_backward_kernel));
    desc.kernels.push_back(std::move(sender_writer_backward_kernel));
    if (sender_writer_forward_single_mux_kernel.has_value()) {
        desc.kernels.push_back(std::move(*sender_writer_forward_single_mux_kernel));
    }
    if (sender_writer_backward_single_mux_kernel.has_value()) {
        desc.kernels.push_back(std::move(*sender_writer_backward_single_mux_kernel));
    }

    if (enable_muxed_multiworker) {
        KernelDescriptor mux_kernel{};
        mux_kernel.kernel_source = "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp";
        mux_kernel.source_type = KernelDescriptor::SourceType::FILE_PATH;
        std::set<CoreRange> mux_ranges;
        for (const auto& core : mux_cores) {
            mux_ranges.insert(CoreRange(core));
        }
        mux_kernel.core_ranges = CoreRangeSet(mux_ranges);
        mux_kernel.compile_time_args = mux_config.get_fabric_mux_compile_time_args();
        mux_kernel.opt_level = tt::tt_metal::KernelBuildOptLevel::O3;
        mux_kernel.config = DataMovementConfigDescriptor{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::RISCV_0_default,
        };

        const auto src_node_id = mesh_device->get_fabric_node_id(target_device_coord);
        for (uint32_t link = 0; link < num_links; ++link) {
            const auto forward_dst_node_id = mesh_device->get_fabric_node_id(*backward_device_coord);
            auto forward_mux_args =
                mux_config.get_fabric_mux_run_time_args(src_node_id, forward_dst_node_id, link, desc, mux_cores[link]);
            KernelDescriptor::RTArgList forward_mux_rt_args;
            forward_mux_rt_args.append(forward_mux_args);
            mux_kernel.emplace_runtime_args(mux_cores[link], forward_mux_rt_args);
        }
        const auto backward_dst_node_id = mesh_device->get_fabric_node_id(*forward_device_coord);
        for (uint32_t link = 0; link < num_links; ++link) {
            auto backward_mux_args = mux_config.get_fabric_mux_run_time_args(
                src_node_id, backward_dst_node_id, link, desc, mux_cores[num_links + link]);
            KernelDescriptor::RTArgList backward_mux_rt_args;
            backward_mux_rt_args.append(backward_mux_args);
            mux_kernel.emplace_runtime_args(mux_cores[num_links + link], backward_mux_rt_args);
        }
        desc.kernels.push_back(std::move(mux_kernel));
    }
}

}  // namespace ttnn
