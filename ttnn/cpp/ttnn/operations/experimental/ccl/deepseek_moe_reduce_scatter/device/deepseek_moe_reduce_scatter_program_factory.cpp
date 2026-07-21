// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
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

// Build a ProgramDescriptor for one coord.  Dynamic CBs (input/intermediate
// slice buffers) are wired up via CBDescriptor::buffer; per-core runtime args
// carry intermediate / output Buffer* via the framework's Buffer* binding
// mechanism.
ProgramDescriptor build_program_descriptor(
    const std::vector<ttnn::Tensor>& input_tensors,
    const std::vector<ttnn::Tensor>& intermediate_slice_tensors,
    const ttnn::Tensor& output_tensor,
    const ttnn::MeshCoordinate& sender_coord,
    const std::optional<ttnn::MeshCoordinate>& forward_coord,
    const std::optional<ttnn::MeshCoordinate>& backward_coord,
    uint32_t ring_index,
    const tt::tt_metal::GlobalSemaphore& op_semaphore,
    const tt::tt_metal::GlobalSemaphore& pre_op_barrier_semaphore,
    uint32_t num_links) {
    auto* mesh_device = input_tensors.at(0).device();

    // hardcoded constants
    const uint32_t ring_size = 8;
    const uint32_t num_directions_per_link = 2;
    const uint32_t num_tile_elements = tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH;

    // tensor details
    const NdShardSpec& input_nd_shard_spec = input_tensors.at(0).nd_shard_spec().value();
    const uint32_t num_pages_per_shard = input_nd_shard_spec.shard_shape.volume() / num_tile_elements;
    const uint32_t num_shards = input_tensors.at(0).physical_volume() / (num_tile_elements * num_pages_per_shard);
    const uint32_t num_pages_per_slice = input_tensors.at(0).buffer()->num_pages();
    const uint32_t page_size = input_tensors.at(0).buffer()->page_size();

    // choose cores
    const auto [clamped_num_links, worker_core_range_set, worker_cores] =
        get_cores(mesh_device, input_nd_shard_spec, num_shards, num_directions_per_link);
    TT_FATAL(clamped_num_links <= num_links, "{} links available, but {} requested", num_links, clamped_num_links);

    // NOTE: writer kernel hardcoded to always use scatter_write with 2 tiles
    const uint32_t tile_granularity = 2;

    // L1 scratch CB creation
    const uint32_t compute_input_cb_num_pages = num_pages_per_shard;   // entire shard
    const uint32_t compute_ouput_cb_num_pages = 2 * tile_granularity;  // double buffer

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensors.at(0).dtype());

    const std::array<uint32_t, 8> input_slice_cb_ids = {
        tt::CBIndex::c_0,
        tt::CBIndex::c_1,
        tt::CBIndex::c_2,
        tt::CBIndex::c_3,
        tt::CBIndex::c_4,
        tt::CBIndex::c_5,
        tt::CBIndex::c_6,
        tt::CBIndex::c_7,
    };
    const std::array<uint32_t, 8> intermediate_slice_cb_ids = {
        tt::CBIndex::c_8,
        tt::CBIndex::c_9,
        tt::CBIndex::c_10,
        tt::CBIndex::c_11,
        tt::CBIndex::c_12,
        tt::CBIndex::c_13,
        tt::CBIndex::c_14,
        tt::CBIndex::c_15,
    };
    const uint32_t compute_cb_id = tt::CBIndex::c_16;

    ProgramDescriptor desc;

    // input CBs — globally-allocated over input tensor buffers.  Setting
    // CBDescriptor::buffer wires the framework's dynamic-CB patcher.
    for (uint32_t i = 0; i < 8; i++) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = compute_input_cb_num_pages * page_size,
            .core_ranges = worker_core_range_set,
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(input_slice_cb_ids[i]),
                .data_format = data_format,
                .page_size = page_size}},
            .buffer = input_tensors.at(i).buffer(),
        });
    }

    // intermediate CBs — globally-allocated over intermediate tensor buffers.
    for (uint32_t i = 0; i < 8; i++) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = compute_input_cb_num_pages * page_size,
            .core_ranges = worker_core_range_set,
            .format_descriptors = {CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(intermediate_slice_cb_ids[i]),
                .data_format = data_format,
                .page_size = page_size}},
            .buffer = intermediate_slice_tensors.at(i).buffer(),
        });
    }

    // compute CB (scratch, not globally allocated)
    desc.cbs.push_back(CBDescriptor{
        .total_size = compute_ouput_cb_num_pages * page_size,
        .core_ranges = worker_core_range_set,
        .format_descriptors = {CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(compute_cb_id), .data_format = data_format, .page_size = page_size}},
    });

    // reader
    std::vector<uint32_t> reader_ct_args = {
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        tile_granularity,  // tile_granularity
        input_slice_cb_ids[0],
        input_slice_cb_ids[1],
        input_slice_cb_ids[2],
        input_slice_cb_ids[3],
        input_slice_cb_ids[4],
        input_slice_cb_ids[5],
        input_slice_cb_ids[6],
        input_slice_cb_ids[7],
        intermediate_slice_cb_ids[0],
        intermediate_slice_cb_ids[1],
        intermediate_slice_cb_ids[2],
        intermediate_slice_cb_ids[3],
        intermediate_slice_cb_ids[4],
        intermediate_slice_cb_ids[5],
        intermediate_slice_cb_ids[6],
        intermediate_slice_cb_ids[7],
    };

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = worker_core_range_set;
    reader_desc.compile_time_args = std::move(reader_ct_args);
    reader_desc.config = ReaderConfigDescriptor{};
    desc.kernels.push_back(std::move(reader_desc));
    const auto reader_kernel_id = desc.kernels.size() - 1;

    // writer
    std::vector<uint32_t> writer_ct_args = {
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        page_size,         // page_size
        tile_granularity,  // tile_granularity
        input_slice_cb_ids[0],
        input_slice_cb_ids[1],
        input_slice_cb_ids[2],
        input_slice_cb_ids[3],
        input_slice_cb_ids[4],
        input_slice_cb_ids[5],
        input_slice_cb_ids[6],
        input_slice_cb_ids[7],
        compute_cb_id,
    };
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(0).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(1).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(2).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(3).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(4).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(5).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(6).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(intermediate_slice_tensors.at(7).buffer()).append_to(writer_ct_args);
    tt::tt_metal::TensorAccessorArgs(output_tensor.buffer()).append_to(writer_ct_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = worker_core_range_set;
    writer_desc.compile_time_args = std::move(writer_ct_args);
    writer_desc.config = WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(writer_desc));
    const auto writer_kernel_id = desc.kernels.size() - 1;

    // reduce
    KernelDescriptor reduce_desc;
    reduce_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/kernels/"
        "deepseek_moe_reduce_scatter_reduction.cpp";
    reduce_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reduce_desc.core_ranges = worker_core_range_set;
    reduce_desc.compile_time_args = {
        ring_index,        // my_chip_id
        ring_size,         // ring_size
        tile_granularity,  // tile_granularity
        input_slice_cb_ids[0],
        input_slice_cb_ids[1],
        input_slice_cb_ids[2],
        input_slice_cb_ids[3],
        input_slice_cb_ids[4],
        input_slice_cb_ids[5],
        input_slice_cb_ids[6],
        input_slice_cb_ids[7],
        intermediate_slice_cb_ids[0],
        intermediate_slice_cb_ids[1],
        intermediate_slice_cb_ids[2],
        intermediate_slice_cb_ids[3],
        intermediate_slice_cb_ids[4],
        intermediate_slice_cb_ids[5],
        intermediate_slice_cb_ids[6],
        intermediate_slice_cb_ids[7],
        compute_cb_id,
    };
    reduce_desc.config = ComputeConfigDescriptor{};
    desc.kernels.push_back(std::move(reduce_desc));
    const auto reduce_kernel_id = desc.kernels.size() - 1;

    // runtime args
    for (uint32_t link = 0; link < clamped_num_links; link++) {
        for (uint32_t direction = 0; direction < num_directions_per_link; direction++) {
            uint32_t worker_id = (link * num_directions_per_link) + direction;
            uint32_t opposite_direction_worker_id =
                (link * num_directions_per_link) + ((direction + 1) % num_directions_per_link);

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

            // reader
            std::vector<uint32_t> reader_rt_args = {
                static_cast<uint32_t>(op_semaphore.address()),  // op_semaphore
                direction,                                      // direction
                start_tiles_read,                               // start_tiles_read
                start_tiles_to_read,                            // start_tiles_to_read
            };
            desc.kernels[reader_kernel_id].runtime_args.emplace_back(core, std::move(reader_rt_args));

            // writer — intermediate slice and output Buffer*s are wired up
            // via emplace_runtime_args so the framework patches them on every
            // dispatch.
            KernelDescriptor::RTArgList writer_rt_args;
            writer_rt_args.push_back(intermediate_slice_tensors.at(0).buffer());      // intermediate_slice_0_address
            writer_rt_args.push_back(intermediate_slice_tensors.at(1).buffer());      // intermediate_slice_1_address
            writer_rt_args.push_back(intermediate_slice_tensors.at(2).buffer());      // intermediate_slice_2_address
            writer_rt_args.push_back(intermediate_slice_tensors.at(3).buffer());      // intermediate_slice_3_address
            writer_rt_args.push_back(intermediate_slice_tensors.at(4).buffer());      // intermediate_slice_4_address
            writer_rt_args.push_back(intermediate_slice_tensors.at(5).buffer());      // intermediate_slice_5_address
            writer_rt_args.push_back(intermediate_slice_tensors.at(6).buffer());      // intermediate_slice_6_address
            writer_rt_args.push_back(intermediate_slice_tensors.at(7).buffer());      // intermediate_slice_7_address
            writer_rt_args.push_back(output_tensor.buffer());                         // output_address
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_core.x));          // op_semaphore_noc0_x
            writer_rt_args.push_back(static_cast<uint32_t>(virtual_core.y));          // op_semaphore_noc0_y
            writer_rt_args.push_back(static_cast<uint32_t>(op_semaphore.address()));  // op_semaphore
            writer_rt_args.push_back(
                static_cast<uint32_t>(opposition_direction_virtual_core.x));  // pre_op_barrier_semaphore_noc0_x
            writer_rt_args.push_back(
                static_cast<uint32_t>(opposition_direction_virtual_core.y));  // pre_op_barrier_semaphore_noc0_y
            writer_rt_args.push_back(
                static_cast<uint32_t>(pre_op_barrier_semaphore.address()));  // pre_op_barrier_semaphore
            writer_rt_args.push_back(direction);                             // direction
            writer_rt_args.push_back(start_tiles_read);                      // start_tiles_read
            writer_rt_args.push_back(start_tiles_to_read);                   // tiles_to_read

            const auto sender_fabric_node_id = mesh_device->get_fabric_node_id(sender_coord);
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            dst_nodes.reserve(1);
            if (direction == 0) {
                // backward
                const auto backward_coord_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
                dst_nodes.push_back(backward_coord_fabric_node_id);
            } else {
                // forward
                const auto forward_coord_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
                dst_nodes.push_back(forward_coord_fabric_node_id);
            }
            // append_routing_plane_connection_manager_rt_args expects a raw
            // uint32_t vector; build one, run the helper, then splice it back
            // into the descriptor's RTArgList.
            std::vector<uint32_t> extra_rt_args;
            tt::tt_metal::KernelHandle writer_kernel_handle = static_cast<tt::tt_metal::KernelHandle>(writer_kernel_id);
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args<ProgramDescriptor>(
                sender_fabric_node_id, dst_nodes, {link}, desc, writer_kernel_handle, core, extra_rt_args);
            for (uint32_t v : extra_rt_args) {
                writer_rt_args.push_back(v);
            }

            desc.kernels[writer_kernel_id].emplace_runtime_args(core, writer_rt_args);

            // reduce
            std::vector<uint32_t> reduce_rt_args = {
                start_tiles_read,     // start_tiles_read
                start_tiles_to_read,  // start_tiles_to_read
                direction};           // direction
            desc.kernels[reduce_kernel_id].runtime_args.emplace_back(core, std::move(reduce_rt_args));
        }
    }

    return desc;
}

}  // namespace

namespace ttnn::experimental::prim {

WorkloadDescriptor DeepseekMoEReduceScatterMeshWorkloadFactory::create_workload_descriptor(
    const DeepseekMoEReduceScatterParams& operation_attributes,
    const DeepseekMoEReduceScatterInputs& tensor_args,
    std::vector<ttnn::Tensor>& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    const std::vector<ttnn::Tensor>& input_tensors = tensor_args.input_tensors;
    const std::vector<ttnn::Tensor> intermediate_slice_tensors(
        tensor_return_value.begin(), tensor_return_value.end() - 1);  // first 8 are intermediate tensors
    const ttnn::Tensor& output_tensor = tensor_return_value.back();   // last is the output tensor

    auto* mesh_device = input_tensors.at(0).device();
    auto sd_id = mesh_device->get_sub_device_ids().at(0);
    auto available_cores = mesh_device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sd_id);

    WorkloadDescriptor wd;

    // Allocate workload-scoped GlobalSemaphores once at cache miss and park
    // them in wd.semaphores so they outlive the cached MeshWorkload.
    // [0] op_semaphore — used for within-op synchronization.
    // [1] pre_op_barrier_semaphore — ensures intermediate/output tensors are allocated.
    wd.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));
    wd.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh_device, available_cores, 0));

    ttsl::SmallVector<tt::tt_metal::SubDeviceId> sub_device_ids = {sd_id};
    tt::tt_metal::distributed::Synchronize(mesh_device, std::nullopt, sub_device_ids);

    const auto& op_semaphore = wd.semaphores[0];
    const auto& pre_op_barrier_semaphore = wd.semaphores[1];

    std::optional<uint32_t> cluster_axis = operation_attributes.cluster_axis;

    const auto coords = tensor_coords.coords();
    wd.programs.reserve(coords.size());

    for (const auto& mesh_coordinate : coords) {
        const std::optional<ttnn::MeshCoordinate> forward_coordinate =
            ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
                input_tensors.at(0), mesh_coordinate, 1, tt::tt_fabric::Topology::Ring, cluster_axis);
        const std::optional<ttnn::MeshCoordinate> backward_coordinate =
            ::ttnn::ccl::get_physical_neighbor_from_physical_coord(
                input_tensors.at(0), mesh_coordinate, -1, tt::tt_fabric::Topology::Ring, cluster_axis);
        TT_FATAL(
            forward_coordinate.has_value() && backward_coordinate.has_value(),
            "DEBUG: forward_coord or backward_coord is null");

        uint32_t device_index =
            ttnn::ccl::get_linearized_index_from_physical_coord(input_tensors.at(0), mesh_coordinate, cluster_axis);
        log_debug(tt::LogOp, "Device index for {} is {}", mesh_coordinate, device_index);

        ProgramDescriptor desc = build_program_descriptor(
            input_tensors,
            intermediate_slice_tensors,
            output_tensor,
            mesh_coordinate,
            forward_coordinate,
            backward_coordinate,
            device_index,
            op_semaphore,
            pre_op_barrier_semaphore,
            operation_attributes.num_links);

        wd.programs.push_back({ttnn::MeshCoordinateRange(mesh_coordinate), std::move(desc)});
    }

    return wd;
}

}  // namespace ttnn::experimental::prim
