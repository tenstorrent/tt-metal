// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/kda/exchange_histories/device/exchange_histories_program_factory.hpp"

#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/experimental/kda/chronological_selections/device/kernels/chronology.hpp"

namespace ttnn::experimental::prim {

namespace {

constexpr uint32_t rows_cb = 0;
constexpr uint32_t scalar_cb = 1;
constexpr uint32_t spans_cb = 2;
// One DRAM-aligned page receives each chronology scalar.
constexpr uint32_t scalar_page_bytes = 64;
// DRAM reads start 64-byte aligned, so each 32-byte face row of a tile is fetched within its aligned span.
constexpr uint32_t span_bytes = 64;
// A few column tiles per gather worker keep the row gather latency-bound.
constexpr uint32_t tiles_per_worker = 8;

// The largest fabric payload that divides a row into equal, 16-byte aligned chunks.
uint32_t chunk_bytes_for(uint32_t row_bytes) {
    const uint32_t max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    for (uint32_t chunk = std::min(row_bytes, max_payload); chunk >= 16; chunk -= 16) {
        if (row_bytes % chunk == 0) {
            return chunk;
        }
    }
    TT_THROW("exchange_histories: no 16-byte aligned chunk divides a {} B row", row_bytes);
}

tt::tt_metal::ProgramDescriptor build_program(
    const ExchangeHistoriesParams& attrs,
    const ExchangeHistoriesInputs& in,
    const Tensor& predecessor,
    const Tensor& final_history,
    const ttnn::MeshCoordinate& coord,
    const tt::tt_metal::GlobalSemaphore& barrier,
    const tt::tt_metal::GlobalSemaphore& arrival) {
    tt::tt_metal::ProgramDescriptor desc;
    auto* mesh = in.projected.device();
    const uint32_t axis = attrs.sequence_parallel_axis;
    const uint32_t sp_size = mesh->shape()[axis];
    const uint32_t sp_rank = coord[axis];
    const uint32_t row_bytes = attrs.width * sizeof(uint16_t);
    TT_FATAL(
        row_bytes == predecessor.buffer()->aligned_page_size(),
        "exchange_histories: output rows must be unpadded {} B rows",
        row_bytes);
    const uint32_t chunk_bytes = chunk_bytes_for(row_bytes);

    const auto forward =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(in.projected, coord, 1, attrs.topology, axis);
    const auto backward =
        ttnn::ccl::get_physical_neighbor_from_physical_coord(in.projected, coord, -1, attrs.topology, axis);
    TT_FATAL(forward || backward, "exchange_histories: no fabric neighbor on the sequence-parallel line");
    const uint32_t ring_index = ttnn::ccl::get_linearized_index_from_physical_coord(in.projected, coord, axis);
    const auto [targets_forward, targets_backward] =
        ttnn::ccl::get_forward_backward_line_mcast_distance(sp_size, ring_index, attrs.topology, true);
    // The successor is the next physical rank: one hop forward, or across the line on an unwrapped last rank.
    auto successor = coord;
    successor[axis] = (sp_rank + 1) % sp_size;
    const bool successor_forward = forward.has_value() && *forward == successor;
    const uint32_t successor_hops = successor_forward ? 1 : sp_size - 1;

    const auto sub_device = mesh->get_sub_device_ids().at(0);
    auto [fabric_range, fabric_cores] =
        ttnn::ccl::choose_worker_cores(1, 1, mesh, sub_device, CoreCoord(0, 0), std::nullopt);
    const auto core = fabric_cores.front();
    // The gather workers take the other worker cores in row-major order, a few column tiles each.
    const uint32_t width_tiles = attrs.width / tt::constants::TILE_WIDTH;
    const uint32_t input_row_tiles = in.projected.padded_shape()[-1] / tt::constants::TILE_WIDTH;
    const auto grid = mesh->compute_with_storage_grid_size();
    const uint32_t gather_workers = std::min<uint32_t>(grid.x * grid.y - 1, tt::div_up(width_tiles, tiles_per_worker));
    const uint32_t tiles_per_core = tt::div_up(width_tiles, gather_workers);
    std::vector<CoreCoord> gather_cores;
    for (uint32_t index = 0; gather_cores.size() < tt::div_up(width_tiles, tiles_per_core); ++index) {
        const CoreCoord candidate{index % grid.x, index / grid.x};
        if (candidate != core) {
            gather_cores.push_back(candidate);
        }
    }
    const CoreRangeSet gather_range{ttsl::Span<const CoreCoord>(gather_cores)};
    const CoreRangeSet cores = gather_range.merge(fabric_range);

    // The staged rows share one address on every core, so each gather worker writes its columns to the fabric
    // worker's rows at the offsets it gathered them at.
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = 2 * kda_chronology::selection::history_rows * row_bytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = rows_cb,
            .data_format = tt::DataFormat::Float16_b,
            .page_size = 2 * kda_chronology::selection::history_rows * row_bytes}}},
    });
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = scalar_page_bytes,
        .core_ranges = cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = scalar_cb, .data_format = tt::DataFormat::UInt32, .page_size = scalar_page_bytes}}},
    });
    const uint32_t spans_bytes = 2 * kda_chronology::selection::history_rows * tiles_per_core * 2 * span_bytes;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = spans_bytes,
        .core_ranges = gather_range,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = spans_cb, .data_format = tt::DataFormat::Float16_b, .page_size = spans_bytes}}},
    });
    // Each gather worker announces its landed columns on the fabric worker.
    desc.semaphores.push_back(tt::tt_metal::SemaphoreDescriptor{
        .id = 0, .core_type = tt::CoreType::WORKER, .core_ranges = fabric_range, .initial_value = 0});
    constexpr uint32_t gathered_semaphore = 0;

    const auto& end_tensor = in.actual_end.value_or(in.actual_start);
    std::vector<uint32_t> compile_args = {
        rows_cb,
        scalar_cb,
        row_bytes,
        chunk_bytes,
        sp_rank,
        sp_size,
        attrs.local_rows,
        in.actual_end.has_value(),
        targets_forward + targets_backward,
        forward ? 1u : 0u,
        forward ? targets_forward : 0u,
        backward ? 1u : 0u,
        backward ? targets_backward : 0u,
        successor_hops,
        successor_forward ? 1u : 0u};
    compile_args.push_back(static_cast<uint32_t>(gather_cores.size()));
    compile_args.push_back(gathered_semaphore);
    tt::tt_metal::TensorAccessorArgs(predecessor.buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(final_history.buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(in.actual_start.buffer()).append_to(compile_args);
    tt::tt_metal::TensorAccessorArgs(end_tensor.buffer()).append_to(compile_args);

    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/exchange_histories/device/kernels/dataflow/exchange_histories.cpp";
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.core_ranges = fabric_range;
    kernel.compile_time_args = std::move(compile_args);
    kernel.config = tt::tt_metal::WriterConfigDescriptor{};
    desc.kernels.push_back(std::move(kernel));
    tt::tt_metal::KernelHandle kernel_id = 0;

    const auto worker = mesh->worker_core_from_logical_core(core);
    std::vector<tt::tt_fabric::FabricNodeId> line_destinations;
    for (const auto& neighbor : {forward, backward}) {
        if (neighbor) {
            line_destinations.push_back(mesh->get_fabric_node_id(*neighbor));
        }
    }
    // Buffer addresses are bound below; the scalar arguments follow them.
    std::vector<uint32_t> runtime_args = {
        0,
        0,
        0,
        0,
        static_cast<uint32_t>(barrier.address()),
        static_cast<uint32_t>(arrival.address()),
        static_cast<uint32_t>(worker.x),
        static_cast<uint32_t>(worker.y),
        static_cast<uint32_t>(line_destinations.size())};
    tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
        mesh->get_fabric_node_id(coord), line_destinations, {0u}, desc, kernel_id, core, runtime_args);
    if (!successor_forward) {
        runtime_args.push_back(1);
        tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
            mesh->get_fabric_node_id(coord),
            {mesh->get_fabric_node_id(successor)},
            {0u},
            desc,
            kernel_id,
            core,
            runtime_args);
    }

    tt::tt_metal::KernelDescriptor::RTArgList list;
    list.reserve(runtime_args.size());
    list.push_back(predecessor.buffer());
    list.push_back(final_history.buffer());
    list.push_back(in.actual_start.buffer());
    list.push_back(end_tensor.buffer());
    for (size_t index = 4; index < runtime_args.size(); ++index) {
        list.push_back(runtime_args[index]);
    }
    desc.kernels[kernel_id].emplace_runtime_args(core, list);

    std::vector<uint32_t> gather_args = {
        rows_cb,
        spans_cb,
        scalar_cb,
        row_bytes,
        input_row_tiles,
        width_tiles,
        tiles_per_core,
        sp_rank,
        sp_size,
        attrs.local_rows,
        in.actual_end.has_value(),
        static_cast<uint32_t>(worker.x),
        static_cast<uint32_t>(worker.y),
        gathered_semaphore};
    tt::tt_metal::TensorAccessorArgs(in.projected.buffer()).append_to(gather_args);
    tt::tt_metal::TensorAccessorArgs(in.actual_start.buffer()).append_to(gather_args);
    tt::tt_metal::TensorAccessorArgs(end_tensor.buffer()).append_to(gather_args);
    tt::tt_metal::KernelDescriptor gather;
    gather.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/kda/exchange_histories/device/kernels/dataflow/gather_histories.cpp";
    gather.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    gather.core_ranges = gather_range;
    gather.compile_time_args = std::move(gather_args);
    gather.config = tt::tt_metal::ReaderConfigDescriptor{};
    for (uint32_t index = 0; index < gather_cores.size(); ++index) {
        tt::tt_metal::KernelDescriptor::RTArgList gather_list;
        gather_list.push_back(in.projected.buffer());
        gather_list.push_back(in.actual_start.buffer());
        gather_list.push_back(end_tensor.buffer());
        gather_list.push_back(index * tiles_per_core);
        gather.emplace_runtime_args(gather_cores[index], gather_list);
    }
    desc.kernels.push_back(std::move(gather));
    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor ExchangeHistoriesProgramFactory::create_workload_descriptor(
    const ExchangeHistoriesParams& attrs,
    const ExchangeHistoriesInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    auto* mesh = in.projected.device();
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
             build_program(attrs, in, outputs[0], outputs[1], coord, workload.semaphores[0], workload.semaphores[1])});
    }
    return workload;
}

}  // namespace ttnn::experimental::prim
