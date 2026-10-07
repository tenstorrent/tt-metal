// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <algorithm>
#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/deepseek_moe_reduce_scatter_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/deepseek_moe_reduce_scatter_program_factory.hpp"

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"
#include "ttnn/operations/math.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr uint32_t kRingSize = 8;
constexpr uint32_t kNumDirectionsPerLink = 2;
constexpr uint32_t kTileGranularity = 2;
// Push order of desc.kernels. Fabric define injection indexes the writer by this handle.
constexpr KernelHandle kReaderKernelIdx = 0;
constexpr KernelHandle kWriterKernelIdx = 1;
constexpr KernelHandle kReduceKernelIdx = 2;

CoreCoord choose_additional_core(
    ttnn::MeshDevice* mesh_device, const std::vector<CoreCoord>& cores_already_selected, uint32_t clamped_num_links) {
    /*
     * - optimal core to use as the additional core (when necessary), so that each used link has both a forward and
     * backward worker
     * - respective core is only optimal when the optimal shard grid is used for the input tensors
     */
    constexpr std::array optimal_supplemental_core_per_link = {
        CoreCoord(2, 5),
        CoreCoord(3, 5),
        CoreCoord(6, 5),
        CoreCoord(0, 5),
    };

    // try optimal core first
    CoreCoord optimal_supplemental_core = optimal_supplemental_core_per_link.at(clamped_num_links - 1);
    if (std::find(cores_already_selected.begin(), cores_already_selected.end(), optimal_supplemental_core) ==
        cores_already_selected.end()) {
        return optimal_supplemental_core;
    }

    // try to find any other available core
    auto available_cores = mesh_device->worker_cores(
        tt::tt_metal::HalProgrammableCoreType::TENSIX, mesh_device->get_sub_device_ids().at(0));
    for (const auto& cr : available_cores.ranges()) {
        auto start = cr.start_coord;
        auto end = cr.end_coord;
        for (size_t y = start.y; y <= end.y; y++) {
            for (size_t x = start.x; x <= end.x; x++) {
                CoreCoord core = CoreCoord(x, y);
                if (std::find(cores_already_selected.begin(), cores_already_selected.end(), core) ==
                    cores_already_selected.end()) {
                    return core;
                }
            }
        }
    }

    TT_FATAL(false, "deepseek_moe_reduce_scatter requires an even number of worker cores");
}

std::tuple<uint32_t, CoreRangeSet, std::vector<CoreCoord>> get_cores(
    ttnn::MeshDevice* mesh_device,
    const NdShardSpec& input_nd_shard_spec,
    uint32_t num_shards,
    uint32_t num_directions_per_link) {
    uint32_t clamped_num_links = tt::div_up(num_shards, num_directions_per_link);

    std::vector<CoreCoord> worker_cores = corerange_to_cores(
        input_nd_shard_spec.grid, num_shards, input_nd_shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    TT_FATAL(
        worker_cores.size() == num_shards,
        "deepseek_moe_reduce_scatter requires each shard to be located on a different core");

    // always need a forward and backward core for each link being used (for in op synchronization), even if the forward
    // worker isn't being used for data transfer due to an odd number of shards
    if (num_shards % 2 != 0) {
        worker_cores.emplace_back(choose_additional_core(mesh_device, worker_cores, clamped_num_links));
    }

    std::vector<CoreRange> worker_core_ranges;
    worker_core_ranges.reserve(worker_cores.size());
    for (const CoreCoord& worker_core : worker_cores) {
        worker_core_ranges.emplace_back(worker_core);
    }
    CoreRangeSet worker_core_range_set = CoreRangeSet(worker_core_ranges);

    return {clamped_num_links, worker_core_range_set, worker_cores};
}

void push_tensor_cb(
    ProgramDescriptor& desc,
    uint32_t cb_id,
    uint32_t num_pages,
    uint32_t page_size,
    DataFormat data_format,
    const CoreRangeSet& cores,
    Buffer* buffer) {
    TT_FATAL(buffer != nullptr, "deepseek_moe_reduce_scatter circular buffer {} requires a device buffer", cb_id);
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_pages * page_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_id),
            .data_format = data_format,
            .page_size = page_size,
        }}},
        .buffer = buffer,
    });
}

ProgramDescriptor build_program_descriptor(
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::vector<ttnn::Tensor>& intermediate_slice_tensors,
    const ttnn::Tensor& output_tensor,
    const ttnn::MeshCoordinate& sender_coord,
    const std::optional<ttnn::MeshCoordinate>& forward_coord,
    const std::optional<ttnn::MeshCoordinate>& backward_coord,
    uint32_t ring_index,
    const GlobalSemaphore& op_semaphore,
    const GlobalSemaphore& pre_op_barrier_semaphore,
    uint32_t num_links) {
    auto* mesh_device = input_tensors.at(0).device();
    TT_FATAL(input_tensors.at(0).buffer() != nullptr, "deepseek_moe_reduce_scatter input tensor must have a buffer");

    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    const NdShardSpec& input_nd_shard_spec = input_tensors.at(0).nd_shard_spec().value();
    const uint32_t num_pages_per_shard = input_nd_shard_spec.shard_shape.volume() / num_tile_elements;
    const uint32_t num_shards = input_tensors.at(0).physical_volume() / (num_tile_elements * num_pages_per_shard);
    const uint32_t num_pages_per_slice = static_cast<uint32_t>(input_tensors.at(0).buffer()->num_pages());
    const uint32_t page_size = static_cast<uint32_t>(input_tensors.at(0).buffer()->page_size());

    const auto [clamped_num_links, worker_core_range_set, worker_cores] =
        get_cores(mesh_device, input_nd_shard_spec, num_shards, kNumDirectionsPerLink);
    TT_FATAL(clamped_num_links <= num_links, "{} links available, but {} requested", num_links, clamped_num_links);

    // NOTE: writer kernel hardcoded to always use scatter_write with 2 tiles
    const uint32_t compute_input_cb_num_pages = num_pages_per_shard;    // entire shard
    const uint32_t compute_output_cb_num_pages = 2 * kTileGranularity;  // double buffer

    DataFormat data_format = datatype_to_dataformat_converter(input_tensors.at(0).dtype());

    const uint32_t input_cb_ids[] = {
        tt::CBIndex::c_0,
        tt::CBIndex::c_1,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        tt::CBIndex::c_4,
        tt::CBIndex::c_5,
        tt::CBIndex::c_6,
        tt::CBIndex::c_7};
    const uint32_t intermediate_cb_ids[] = {
        tt::CBIndex::c_8,
        tt::CBIndex::c_9,
        tt::CBIndex::c_10,
        tt::CBIndex::c_11,
        tt::CBIndex::c_12,
        tt::CBIndex::c_13,
        tt::CBIndex::c_14,
        tt::CBIndex::c_15};
    const uint32_t compute_cb_id = tt::CBIndex::c_16;

    ProgramDescriptor desc;
    TT_FATAL(input_tensors.size() == kRingSize, "deepseek_moe_reduce_scatter expects {} input slices", kRingSize);
    TT_FATAL(
        intermediate_slice_tensors.size() == kRingSize,
        "deepseek_moe_reduce_scatter expects {} intermediate slices",
        kRingSize);
    for (uint32_t slice = 0; slice < kRingSize; ++slice) {
        push_tensor_cb(
            desc,
            input_cb_ids[slice],
            compute_input_cb_num_pages,
            page_size,
            data_format,
            worker_core_range_set,
            input_tensors.at(slice).buffer());
        push_tensor_cb(
            desc,
            intermediate_cb_ids[slice],
            compute_input_cb_num_pages,
            page_size,
            data_format,
            worker_core_range_set,
            intermediate_slice_tensors.at(slice).buffer());
    }
    desc.cbs.push_back(CBDescriptor{
        .total_size = compute_output_cb_num_pages * page_size,
        .core_ranges = worker_core_range_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(compute_cb_id),
            .data_format = data_format,
            .page_size = page_size,
        }}},
    });

    std::vector<uint32_t> reader_ct_args = {
        ring_index,
        kRingSize,
        kTileGranularity,
        input_cb_ids[0],
        input_cb_ids[1],
        input_cb_ids[2],
        input_cb_ids[3],
        input_cb_ids[4],
        input_cb_ids[5],
        input_cb_ids[6],
        input_cb_ids[7],
        intermediate_cb_ids[0],
        intermediate_cb_ids[1],
        intermediate_cb_ids[2],
        intermediate_cb_ids[3],
        intermediate_cb_ids[4],
        intermediate_cb_ids[5],
        intermediate_cb_ids[6],
        intermediate_cb_ids[7],
    };
    std::vector<uint32_t> writer_ct_args = {
        ring_index,
        kRingSize,
        page_size,
        kTileGranularity,
        input_cb_ids[0],
        input_cb_ids[1],
        input_cb_ids[2],
        input_cb_ids[3],
        input_cb_ids[4],
        input_cb_ids[5],
        input_cb_ids[6],
        input_cb_ids[7],
        compute_cb_id,
    };
    for (uint32_t slice = 0; slice < kRingSize; ++slice) {
        TT_FATAL(
            intermediate_slice_tensors.at(slice).buffer() != nullptr,
            "deepseek_moe_reduce_scatter intermediate slice {} must have a buffer",
            slice);
        TensorAccessorArgs(intermediate_slice_tensors.at(slice).buffer()).append_to(writer_ct_args);
    }
    TT_FATAL(output_tensor.buffer() != nullptr, "deepseek_moe_reduce_scatter output tensor must have a buffer");
    TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);

    std::vector<uint32_t> reduce_ct_args = {
        ring_index,
        kRingSize,
        kTileGranularity,
        input_cb_ids[0],
        input_cb_ids[1],
        input_cb_ids[2],
        input_cb_ids[3],
        input_cb_ids[4],
        input_cb_ids[5],
        input_cb_ids[6],
        input_cb_ids[7],
        intermediate_cb_ids[0],
        intermediate_cb_ids[1],
        intermediate_cb_ids[2],
        intermediate_cb_ids[3],
        intermediate_cb_ids[4],
        intermediate_cb_ids[5],
        intermediate_cb_ids[6],
        intermediate_cb_ids[7],
        compute_cb_id,
    };

    const std::string kernel_dir =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/";
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source = kernel_dir + "deepseek_moe_reduce_scatter_reader.cpp",
        .core_ranges = worker_core_range_set,
        .compile_time_args = std::move(reader_ct_args),
        .config = ReaderConfigDescriptor{},
    });
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source = kernel_dir + "deepseek_moe_reduce_scatter_writer.cpp",
        .core_ranges = worker_core_range_set,
        .compile_time_args = std::move(writer_ct_args),
        .config = WriterConfigDescriptor{},
    });
    desc.kernels.push_back(KernelDescriptor{
        .kernel_source = kernel_dir + "deepseek_moe_reduce_scatter_reduction.cpp",
        .core_ranges = worker_core_range_set,
        .compile_time_args = std::move(reduce_ct_args),
        .config = ComputeConfigDescriptor{},
    });

    for (uint32_t link = 0; link < clamped_num_links; link++) {
        for (uint32_t direction = 0; direction < kNumDirectionsPerLink; direction++) {
            uint32_t worker_id = (link * kNumDirectionsPerLink) + direction;
            uint32_t opposite_direction_worker_id =
                (link * kNumDirectionsPerLink) + ((direction + 1) % kNumDirectionsPerLink);

            CoreCoord core = worker_cores[worker_id];
            CoreCoord virtual_core = mesh_device->worker_core_from_logical_core(core);

            CoreCoord opposite_direction_core = worker_cores[opposite_direction_worker_id];
            CoreCoord opposition_direction_virtual_core =
                mesh_device->worker_core_from_logical_core(opposite_direction_core);

            /*
             * NOTE
             * - need to create kernels even if worker not processing tiles, required for pre and post op barrier/sync
             * - min so that we don't try to process non-existent tiles on that dummy worker
             */
            uint32_t start_tiles_read = num_pages_per_shard * worker_id;
            uint32_t start_tiles_to_read = num_pages_per_shard * (worker_id + 1);
            start_tiles_to_read = std::min(start_tiles_to_read, num_pages_per_slice);

            KernelDescriptor::RTArgList reader_rt_args;
            reader_rt_args.push_back(static_cast<uint32_t>(
                op_semaphore
                    .address()));  // smuggled-rta-ok: persistent GlobalSemaphore parked on the WorkloadDescriptor
            reader_rt_args.push_back(direction);
            reader_rt_args.push_back(start_tiles_read);
            reader_rt_args.push_back(start_tiles_to_read);
            desc.kernels[kReaderKernelIdx].emplace_runtime_args(core, reader_rt_args);

            std::vector<uint32_t> writer_fabric_args = {
                static_cast<uint32_t>(virtual_core.x),
                static_cast<uint32_t>(virtual_core.y),
                static_cast<uint32_t>(op_semaphore.address()),  // smuggled-rta-ok: persistent GlobalSemaphore parked on
                                                                // the WorkloadDescriptor
                static_cast<uint32_t>(opposition_direction_virtual_core.x),
                static_cast<uint32_t>(opposition_direction_virtual_core.y),
                static_cast<uint32_t>(
                    pre_op_barrier_semaphore
                        .address()),  // smuggled-rta-ok: persistent GlobalSemaphore parked on the WorkloadDescriptor
                direction,
                start_tiles_read,
                start_tiles_to_read,
            };

            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_coord);
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            dst_nodes.reserve(1);
            if (direction == 0) {
                const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                dst_nodes.push_back(backward_coord_fabric_node_id);
            } else {
                const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                dst_nodes.push_back(forward_coord_fabric_node_id);
            }
            KernelHandle writer_kernel_handle = kWriterKernelIdx;
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args<ProgramDescriptor>(
                sender_fabric_node_id, dst_nodes, {link}, desc, writer_kernel_handle, core, writer_fabric_args);

            KernelDescriptor::RTArgList writer_rt_args;
            for (uint32_t slice = 0; slice < kRingSize; ++slice) {
                writer_rt_args.push_back(intermediate_slice_tensors.at(slice).buffer());
            }
            writer_rt_args.push_back(output_tensor.buffer());
            writer_rt_args.append(writer_fabric_args);
            desc.kernels[kWriterKernelIdx].emplace_runtime_args(core, writer_rt_args);

            desc.kernels[kReduceKernelIdx].emplace_runtime_args(
                core, {start_tiles_read, start_tiles_to_read, direction});
        }
    }

    return desc;
}

}  // namespace

namespace ttnn::experimental::prim {

tt::tt_metal::WorkloadDescriptor DeepseekMoEReduceScatterMeshWorkloadFactory::create_workload_descriptor(
    const DeepseekMoEReduceScatterParams& operation_attributes,
    const DeepseekMoEReduceScatterInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload_descriptor;

    auto* mesh_device = tensor_args.input_tensors.at(0).device();
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    auto& sems = workload_descriptor.semaphores;
    sems.reserve(SemaphoreIndex::count);
    // 1 semaphore used for within op synchronizations
    sems.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    // 1 semaphore used for pre op synchronization to ensure intermediate/output tensors are allocated
    sems.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));

    ttsl::SmallVector<tt::tt_metal::SubDeviceId> sub_device_ids = {sd_id};
    tt::tt_metal::distributed::Synchronize(*mesh_device, std::nullopt, sub_device_ids);

    TT_FATAL(tensor_return_value.size() == 9, "deepseek_moe_reduce_scatter returns 8 intermediates and 1 output");
    const std::vector<ttnn::Tensor> intermediate_slice_tensors(
        tensor_return_value.begin(), tensor_return_value.end() - 1);
    const ttnn::Tensor& output_tensor = tensor_return_value.back();

    for (const auto& coord : tensor_coords.coords()) {
        const std::optional<ttnn::MeshCoordinate> forward_coordinate =
            ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
                tensor_args.input_tensors.at(0),
                coord,
                1,
                tt::tt_fabric::Topology::Ring,
                operation_attributes.cluster_axis);
        const std::optional<ttnn::MeshCoordinate> backward_coordinate =
            ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
                tensor_args.input_tensors.at(0),
                coord,
                -1,
                tt::tt_fabric::Topology::Ring,
                operation_attributes.cluster_axis);
        TT_FATAL(
            forward_coordinate.has_value() && backward_coordinate.has_value(),
            "DEBUG: forward_coord or backward_coord is null");

        uint32_t device_index = ttnn::ccl::get_linearized_index_from_physical_coord(
            tensor_args.input_tensors.at(0), coord, operation_attributes.cluster_axis);
        log_debug(tt::LogOp, "Device index for {} is {}", coord, device_index);

        workload_descriptor.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_program_descriptor(
                 tensor_args.input_tensors,
                 intermediate_slice_tensors,
                 output_tensor,
                 coord,
                 forward_coordinate,
                 backward_coordinate,
                 device_index,
                 sems[SemaphoreIndex::op],
                 sems[SemaphoreIndex::pre_op_barrier],
                 operation_attributes.num_links)});
    }

    return workload_descriptor;
}

}  // namespace ttnn::experimental::prim
