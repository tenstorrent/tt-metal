// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/select_final_carry/device/select_final_carry_program_factory.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t tiles_cb = 0;
constexpr uint32_t mode_cb = 1;
// One DRAM-aligned scratch page receives each chronology scalar before it carries the derived mode.
constexpr uint32_t mode_page_bytes = 64;
// A few tiles per copy worker keep the unsplit copy latency-bound.
constexpr uint32_t tiles_per_copy_worker = 8;
constexpr const char* kernel_dir =
    "ttnn/cpp/ttnn/operations/experimental/kda/select_final_carry/device/kernels/dataflow/";

struct TileRange {
    uint32_t begin = 0;
    uint32_t end = 0;
};

TileRange split_range(uint32_t tiles, uint32_t parts, uint32_t index) {
    const uint32_t per_part = (tiles + parts - 1) / parts;
    return {std::min(tiles, index * per_part), std::min(tiles, (index + 1) * per_part)};
}

tt::tt_metal::ProgramDescriptor build_program(
    const SelectFinalCarryParams& attrs,
    const SelectFinalCarryInputs& in,
    const Tensor& output,
    const ttnn::MeshCoordinate& coord,
    const tt::tt_metal::GlobalSemaphore& barrier,
    const tt::tt_metal::GlobalSemaphore& arrival) {
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh = in.prefix_final.device();
    const uint32_t axis = attrs.sequence_parallel_axis;
    const uint32_t sp_size = mesh->shape()[axis];
    const uint32_t sp_rank = coord[axis];
    const uint32_t state_tiles = in.prefix_final.physical_volume() / tt::constants::TILE_HW;
    const auto& prefix_shape = in.prefix_final.padded_shape();
    const uint32_t head_tiles = prefix_shape[-2] * prefix_shape[-1] / tt::constants::TILE_HW;
    const uint32_t tail_groups = in.rank_final.logical_shape().rank() == 4 ? in.rank_final.logical_shape()[1] : 1;
    const uint32_t page_size = in.prefix_final.buffer()->aligned_page_size();
    const uint32_t max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    TT_FATAL(page_size <= max_payload, "select_final_carry: a {} B tile exceeds the fabric payload", page_size);
    const uint32_t packet_pages = std::min<uint32_t>(2, max_payload / page_size);

    const auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t copy_workers =
        std::min<uint32_t>(grid.x * grid.y, (state_tiles + tiles_per_copy_worker - 1) / tiles_per_copy_worker);
    const auto copy_cores = tt::tt_metal::corerange_to_cores(
        tt::tt_metal::num_cores_to_corerangeset(copy_workers, grid, /*row_wise=*/true), std::nullopt, true);

    // The tail can only separate from the prefix on a line of several ranks.
    std::optional<ttnn::MeshCoordinate> forward, backward;
    std::vector<CoreCoord> fabric_cores;
    CoreRangeSet all_cores = tt::tt_metal::num_cores_to_corerangeset(copy_workers, grid, /*row_wise=*/true);
    if (sp_size > 1) {
        forward = ttnn::ccl::get_physical_neighbor_from_physical_coord(in.prefix_final, coord, 1, attrs.topology, axis);
        backward =
            ttnn::ccl::get_physical_neighbor_from_physical_coord(in.prefix_final, coord, -1, attrs.topology, axis);
        TT_FATAL(forward || backward, "select_final_carry: no fabric neighbor on the sequence-parallel line");
        const auto sub_device = mesh->get_sub_device_ids().at(0);
        CoreRangeSet fabric_range;
        std::tie(fabric_range, fabric_cores) =
            ttnn::ccl::choose_worker_cores(attrs.num_links, 1, mesh, sub_device, CoreCoord(0, 0), std::nullopt);
        all_cores = all_cores.merge(fabric_range);
    }
    const uint32_t ring_index = ttnn::ccl::get_linearized_index_from_physical_coord(in.prefix_final, coord, axis);
    const auto [targets_forward, targets_backward] =
        sp_size > 1 ? ttnn::ccl::get_forward_backward_line_mcast_distance(sp_size, ring_index, attrs.topology, true)
                    : std::tuple<uint32_t, uint32_t>{0, 0};

    const auto state_format = tt::tt_metal::datatype_to_dataformat_converter(in.prefix_final.dtype());
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 4 * packet_pages * page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = tiles_cb, .data_format = state_format, .page_size = page_size}}},
    });
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = mode_page_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = mode_cb, .data_format = tt::DataFormat::UInt32, .page_size = mode_page_bytes}}},
    });

    const auto& end_tensor = in.actual_end.value_or(in.actual_start);
    std::vector<uint32_t> reader_args = {
        tiles_cb,
        mode_cb,
        packet_pages,
        sp_rank,
        sp_size,
        attrs.local_rows,
        in.actual_end.has_value(),
        tail_groups,
        head_tiles};
    tt::tt_metal::TensorAccessorArgs(in.rank_final.buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(in.prefix_final.buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(in.actual_start.buffer()).append_to(reader_args);
    tt::tt_metal::TensorAccessorArgs(end_tensor.buffer()).append_to(reader_args);
    std::vector<uint32_t> writer_args = {
        tiles_cb,
        mode_cb,
        packet_pages,
        page_size,
        targets_forward + targets_backward,
        uint32_t(forward.has_value()),
        forward ? targets_forward : 0u,
        uint32_t(backward.has_value()),
        backward ? targets_backward : 0u};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_args);

    tt::tt_metal::KernelDescriptor reader;
    reader.kernel_source = std::string(kernel_dir) + "select_final_carry_reader.cpp";
    reader.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_args);
    reader.config = tt::tt_metal::ReaderConfigDescriptor{};
    tt::tt_metal::KernelDescriptor writer;
    writer.kernel_source = std::string(kernel_dir) + "select_final_carry_writer.cpp";
    writer.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_args);
    writer.config = tt::tt_metal::WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    tt::tt_metal::KernelHandle reader_id = 0;
    tt::tt_metal::KernelHandle writer_id = 1;

    const auto drain_core =
        fabric_cores.empty() ? CoreCoord{} : mesh->worker_core_from_logical_core(fabric_cores.front());
    for (const auto& range : all_cores.ranges()) {
        for (const auto& core : range) {
            const auto copy_index = std::find(copy_cores.begin(), copy_cores.end(), core) - copy_cores.begin();
            const auto link = std::find(fabric_cores.begin(), fabric_cores.end(), core) - fabric_cores.begin();
            const bool fabric_worker = link < static_cast<std::ptrdiff_t>(fabric_cores.size());
            const TileRange copy = copy_index < static_cast<std::ptrdiff_t>(copy_cores.size())
                                       ? split_range(state_tiles, copy_workers, copy_index)
                                       : TileRange{};
            const TileRange send = fabric_worker ? split_range(state_tiles, attrs.num_links, link) : TileRange{};

            tt::tt_metal::KernelDescriptor::RTArgList reader_rt;
            reader_rt.push_back(in.rank_final.buffer());
            reader_rt.push_back(in.prefix_final.buffer());
            reader_rt.push_back(in.actual_start.buffer());
            reader_rt.push_back(end_tensor.buffer());
            for (uint32_t value : {copy.begin, copy.end, send.begin, send.end}) {
                reader_rt.push_back(value);
            }
            desc.kernels[reader_id].emplace_runtime_args(core, reader_rt);

            // Index 0 is replaced with the output buffer binding below.
            std::vector<uint32_t> writer_rt = {0, copy.begin, copy.end, send.begin, send.end, fabric_worker};
            if (fabric_worker) {
                const auto worker = mesh->worker_core_from_logical_core(core);
                const uint32_t connections = uint32_t(forward.has_value()) + uint32_t(backward.has_value());
                writer_rt.insert(
                    writer_rt.end(),
                    {static_cast<uint32_t>(arrival.address()),
                     static_cast<uint32_t>(drain_core.x),
                     static_cast<uint32_t>(drain_core.y),
                     uint32_t(link == 0),
                     attrs.num_links,
                     static_cast<uint32_t>(barrier.address()),
                     static_cast<uint32_t>(worker.x),
                     static_cast<uint32_t>(worker.y),
                     connections});
                std::vector<tt::tt_fabric::FabricNodeId> destinations;
                for (const auto& neighbor : {forward, backward}) {
                    if (neighbor) {
                        destinations.push_back(mesh->get_fabric_node_id(*neighbor));
                    }
                }
                tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                    mesh->get_fabric_node_id(coord),
                    destinations,
                    {static_cast<uint32_t>(link)},
                    desc,
                    writer_id,
                    core,
                    writer_rt);
            }
            tt::tt_metal::KernelDescriptor::RTArgList writer_list;
            writer_list.reserve(writer_rt.size());
            writer_list.push_back(output.buffer());
            for (size_t index = 1; index < writer_rt.size(); ++index) {
                writer_list.push_back(writer_rt[index]);
            }
            desc.kernels[writer_id].emplace_runtime_args(core, writer_list);
        }
    }
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor SelectFinalCarryProgramFactory::create_workload_descriptor(
    const SelectFinalCarryParams& attrs,
    const SelectFinalCarryInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    auto* mesh = in.prefix_final.device();
    const auto sub_device = mesh->get_sub_device_ids().at(0);
    const auto cores = mesh->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device);
    // Both counters persist across replays, so they live where a fabric mux cannot clobber them.
    const auto semaphore_memory = ttnn::ccl::prefer_l1_small_buffer_type(*mesh);
    workload.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh, cores, 0, semaphore_memory));
    workload.semaphores.push_back(ttnn::global_semaphore::create_global_semaphore(mesh, cores, 0, semaphore_memory));
    tt::tt_metal::distributed::Synchronize(
        *mesh, std::nullopt, ttsl::SmallVector<tt::tt_metal::SubDeviceId>{sub_device});
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_program(attrs, in, outputs[0], coord, workload.semaphores[0], workload.semaphores[1])});
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
