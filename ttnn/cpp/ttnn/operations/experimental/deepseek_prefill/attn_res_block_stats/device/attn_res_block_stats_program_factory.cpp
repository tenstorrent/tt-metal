// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/attn_res_block_stats_program_factory.hpp"

#include <algorithm>
#include <bit>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/ccl/common/host/descriptor_runtime_args.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_block_stats/device/kernels/";

CBDescriptor make_cb(
    CBIndex index, const CoreRangeSet& cores, uint32_t num_tiles, uint32_t tile_size, DataFormat format) {
    return CBDescriptor{
        .total_size = num_tiles * tile_size,
        .core_ranges = cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(index),
            .data_format = format,
            .page_size = tile_size,
        }}},
    };
}

// The cores that hold the gather's fabric connections: the grid's last, one per link.
CoreRangeSet gather_cores_of(const CoreCoord& grid, uint32_t num_gather_cores) {
    if (num_gather_cores == 0) {
        return CoreRangeSet{};
    }
    return CoreRangeSet{CoreRange({grid.x - num_gather_cores, grid.y - 1}, {grid.x - 1, grid.y - 1})};
}

// The most runs a row is cut into: a row's leader holds a partial per run and candidate.
constexpr uint32_t kMaxChunksPerRow = 8;
// Pages the gather core stages behind one read barrier: its staging buffer's size.
constexpr uint32_t kGatherBatch = 16;

// Semaphores: a row leader's per-run counts, and the gather cores' per-row counts of
// parked candidates, which take as many semaphore words as their fields need.
constexpr uint32_t kChunkSem = 0;
constexpr uint32_t kRowSem = 1;
// Semaphore ids a program can hold.
constexpr uint32_t kMaxSemaphores = 16;

// Circular buffers. The kernels take every id as a named compile-time arg.
constexpr auto cb_v = CBIndex::c_0;
constexpr auto cb_queries = CBIndex::c_1;
constexpr auto cb_ones = CBIndex::c_2;
constexpr auto cb_squares = CBIndex::c_3;
constexpr auto cb_local = CBIndex::c_4;
constexpr auto cb_received = CBIndex::c_5;
constexpr auto cb_headers = CBIndex::c_8;
constexpr auto cb_stage = CBIndex::c_9;
constexpr auto cb_out = CBIndex::c_16;

// The gather kernel is the program's fourth, after the reader, writer and compute kernels.
constexpr tt::tt_metal::KernelHandle kGatherKernel = 3;

ProgramDescriptor build_descriptor_at(
    const AttnResBlockStatsParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coord,
    const AttnResBlockStatsInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto* device = tensor_args.queries.device();
    const auto& first = tensor_args.sealed.front();

    const auto in_format = datatype_to_dataformat_converter(first.dtype());
    const auto in_tile_size = tt::tile_size(in_format);
    const auto out_format = datatype_to_dataformat_converter(tensor_return_value.dtype());
    const auto out_tile_size = tt::tile_size(out_format);

    const auto& shape = first.padded_shape();
    const uint32_t Ht = shape[2] / TILE_HEIGHT;
    const uint32_t Wt = shape[3] / TILE_WIDTH;
    const uint32_t num_candidates = tensor_return_value.padded_shape()[1];

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    // Where the op gathers across an axis: this chip's rank on it, and its peers.
    const bool gathers = operation_attributes.cluster_axis.has_value();
    const auto& semaphore = operation_attributes.semaphore;
    const auto mesh_view = device->get_view();
    const uint32_t cluster_axis = operation_attributes.cluster_axis.value_or(0);
    const uint32_t ring_size = !gathers ? 1 : cluster_axis == 0 ? mesh_view.num_rows() : mesh_view.num_cols();
    const uint32_t rank = gathers ? mesh_coord[cluster_axis] : 0;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    // A unit is a run of one token row across every candidate (see
    // `kernels/attn_res_block_stats_items.hpp`). Rows are cut into as many runs as the grid
    // seats so every core reads the same share; where rows outnumber the cores, a core takes
    // several whole rows.
    const auto grid = device->compute_with_storage_grid_size();
    // Where the op gathers, the grid's last cores hold the fabric connections instead, one
    // per link: a core opens one connection per direction, a link admits a single sender, and
    // a core's dataflow RISCs are otherwise a worker's reader and writer.
    const uint32_t num_gather_cores = gathers ? operation_attributes.num_links : 0;
    const uint32_t grid_cores = grid.x * grid.y - num_gather_cores;
    const uint32_t chunks_per_row = std::clamp(grid_cores / Ht, 1u, std::min(Wt, kMaxChunksPerRow));
    const uint32_t num_units = Ht * chunks_per_row;
    const uint32_t units_per_core = (num_units + grid_cores - 1) / grid_cores;
    const uint32_t num_cores = (num_units + units_per_core - 1) / units_per_core;
    const uint32_t max_run = (Wt + chunks_per_row - 1) / chunks_per_row;
    const CoreRangeSet all_cores = num_cores_to_corerangeset(num_cores, grid, /*row_wise=*/true);
    const auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/true);
    TT_FATAL(chunks_per_row == 1 || units_per_core == 1, "a cut row must take one core per run");

    ProgramDescriptor desc;

    ////////////////////////////////////////////////////////////////////////////
    //                         CircularBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    // A candidate's run is one CB unit, double-buffered so the next candidate's run arrives
    // while this one is reduced. The squares hold a run too: the sum of squares is a second
    // matmul over them, into the same accumulator as the dots.
    desc.cbs.push_back(make_cb(cb_v, all_cores, 2 * max_run, in_tile_size, in_format));
    desc.cbs.push_back(make_cb(cb_queries, all_cores, max_run, in_tile_size, in_format));  // resident
    desc.cbs.push_back(make_cb(cb_ones, all_cores, 1, in_tile_size, in_format));           // ones in column 0
    desc.cbs.push_back(make_cb(cb_squares, all_cores, max_run, in_tile_size, in_format));
    desc.cbs.push_back(make_cb(cb_out, all_cores, 2, out_tile_size, out_format));
    if (chunks_per_row > 1) {
        // A row leader's own partials, and the row's other runs', written in by them. On
        // every core, so a sender finds the leader's at its own copy's address.
        desc.cbs.push_back(make_cb(cb_local, all_cores, num_candidates, out_tile_size, out_format));
        desc.cbs.push_back(
            make_cb(cb_received, all_cores, (chunks_per_row - 1) * num_candidates, out_tile_size, out_format));
    }
    // A row leader counts each other run's candidates, one field per run. A count reaches at
    // most num_candidates, so a field of its bit width never carries.
    const uint32_t field_width = std::bit_width(num_candidates);
    TT_FATAL(
        (chunks_per_row - 1) * field_width <= 32,
        "a row leader counts {} runs' candidates in one semaphore",
        chunks_per_row - 1);
    desc.semaphores.push_back(SemaphoreDescriptor{.id = kChunkSem, .core_ranges = all_cores, .initial_value = 0});
    // The gather cores count each row's parked candidates, one field per row, in the semaphore
    // ids left over. A field counts in steps of `row_step` candidates (the last step may be
    // short), so where the rows outnumber what full-width fields fit, narrower fields trade how
    // early a row's first pages go out for room: at one bit a row is announced once, whole.
    uint32_t row_field_width = field_width;
    const auto row_sem_words_at = [&](uint32_t width) { return (Ht + 32 / width - 1) / (32 / width); };
    while (row_field_width > 1 && kRowSem + row_sem_words_at(row_field_width) > kMaxSemaphores) {
        --row_field_width;
    }
    const uint32_t row_sem_words = row_sem_words_at(row_field_width);
    const uint32_t row_levels = (1u << row_field_width) - 1;
    const uint32_t row_step = (num_candidates + row_levels - 1) / row_levels;
    const CoreRangeSet gather_core_set = gather_cores_of(grid, num_gather_cores);
    const auto gather_cores = corerange_to_cores(gather_core_set, std::nullopt, /*row_wise=*/true);
    if (gathers) {
        // The gather kernel keeps its per-row sent counts in as many words.
        TT_FATAL(
            kRowSem + row_sem_words <= kMaxSemaphores,
            "AttnResBlockStats gathers at most {} row tiles, got {}",
            (kMaxSemaphores - kRowSem) * 32,
            Ht);
        TT_FATAL(!all_cores.intersects(gather_core_set), "the gather cores must hold no rows");
        for (uint32_t w = 0; w < row_sem_words; ++w) {
            desc.semaphores.push_back(
                SemaphoreDescriptor{.id = kRowSem + w, .core_ranges = gather_core_set, .initial_value = 0});
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernels
    ////////////////////////////////////////////////////////////////////////////
    // Every kernel takes the shape, the split and the CB and semaphore ids by name; only the
    // tensor accessors are positional. A core finds its units from its own position, so a
    // launch ships no per-core args.
    const KernelDescriptor::NamedCompileTimeArgs named_ct_args = {
        {"num_candidates", num_candidates},
        {"Ht", Ht},
        {"Wt", Wt},
        {"grid_x", grid.x},
        {"units_per_core", units_per_core},
        {"num_units", num_units},
        {"chunks_per_row", chunks_per_row},
        {"ring_size", ring_size},
        {"field_width", field_width},
        {"row_field_width", row_field_width},
        {"row_step", row_step},
        {"num_gather_cores", num_gather_cores},
        {"gather_batch", kGatherBatch},
        {"chunk_sem_id", kChunkSem},
        {"row_sem_id", kRowSem},
        {"cb_v", cb_v},
        {"cb_queries", cb_queries},
        {"cb_ones", cb_ones},
        {"cb_squares", cb_squares},
        {"cb_local", cb_local},
        {"cb_received", cb_received},
        {"cb_headers", cb_headers},
        {"cb_stage", cb_stage},
        {"cb_out", cb_out},
    };

    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*first.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*tensor_args.queries.buffer()).append_to(reader_ct_args);

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_ct_args);

    // Each candidate is a (snapshot address, first page) pair: the snapshot that holds it and
    // the offset of its plane there. The snapshots share one accessor configuration, which
    // the device operation checks, so the address is all that differs between them.
    KernelDescriptor::RTArgList reader_common;
    reader_common.push_back(tensor_args.queries.buffer());
    for (const auto& snapshot : tensor_args.sealed) {
        for (uint32_t plane = 0; plane < snapshot.padded_shape()[1]; ++plane) {
            reader_common.push_back(snapshot.buffer());
            reader_common.push_back(plane * Ht * Wt);
        }
    }

    // This chip's rank and its gather cores, then where rows are cut, each row's leader — its
    // chunk 0 core — that the row's runs send to. Cores go as (x << 16) | y.
    KernelDescriptor::RTArgList writer_common;
    writer_common.push_back(tensor_return_value.buffer());
    writer_common.push_back(rank);
    for (const auto& core : gather_cores) {
        const auto physical = device->worker_core_from_logical_core(core);
        writer_common.push_back((physical.x << 16) | physical.y);
    }
    if (chunks_per_row > 1) {
        for (uint32_t g = 0; g < Ht; ++g) {
            const auto leader = device->worker_core_from_logical_core(cores.at(g * chunks_per_row));
            writer_common.push_back((leader.x << 16) | leader.y);
        }
    }

    KernelDescriptor reader;
    reader.kernel_source = std::string(kKernelDir) + "reader_attn_res_block_stats.cpp";
    reader.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader.core_ranges = all_cores;
    reader.compile_time_args = std::move(reader_ct_args);
    reader.named_compile_time_args = named_ct_args;
    reader.config = ReaderConfigDescriptor{};
    reader.emplace_common_runtime_args(reader_common);

    KernelDescriptor writer;
    writer.kernel_source = std::string(kKernelDir) + "writer_attn_res_block_stats.cpp";
    writer.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer.core_ranges = all_cores;
    writer.compile_time_args = std::move(writer_ct_args);
    writer.named_compile_time_args = named_ct_args;
    writer.config = WriterConfigDescriptor{};
    writer.emplace_common_runtime_args(writer_common);

    // A leader's partials are only ever copied into dest, so they unpack straight to it
    // rather than being cut to tf32 on the way through srcA.
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[cb_local] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[cb_received] = UnpackToDestMode::UnpackToDestFp32;

    KernelDescriptor compute;
    compute.kernel_source = std::string(kKernelDir) + "attn_res_block_stats.cpp";
    compute.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute.core_ranges = all_cores;
    compute.named_compile_time_args = named_ct_args;
    compute.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = dst_full_sync_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
        .math_approx_mode = math_approx_mode,
    };

    desc.kernels.push_back(std::move(reader));
    desc.kernels.push_back(std::move(writer));
    desc.kernels.push_back(std::move(compute));

    if (gathers) {
        // A payload header and an increment header per direction, and a batch of pages staged.
        const auto header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
        desc.cbs.push_back(make_cb(cb_headers, gather_core_set, 2 * 2, header_size, DataFormat::UInt32));
        desc.cbs.push_back(make_cb(cb_stage, gather_core_set, kGatherBatch, out_tile_size, out_format));

        // A multicast route each way along the axis, as far as the peers that lie that way:
        // a page crosses each link once however many peers are beyond it, and every peer is
        // reached by one route from each sender. Gather core l takes link l.
        const auto topology = operation_attributes.topology;
        const auto [forward_targets, backward_targets] =
            ttnn::ccl::get_forward_backward_line_mcast_distance(ring_size, rank, topology, /*static_alternate=*/true);
        const auto forward = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.queries, mesh_coord, 1, topology, cluster_axis);
        const auto backward = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            tensor_args.queries, mesh_coord, -1, topology, cluster_axis);
        const auto [forward_route, backward_route] = ttnn::ccl::get_forward_backward_line_mcast_configuration(
            mesh_coord, forward, backward, forward_targets, backward_targets, device);

        std::vector<uint32_t> gather_ct_args;
        TensorAccessorArgs(*tensor_return_value.buffer()).append_to(gather_ct_args);
        KernelDescriptor gather;
        gather.kernel_source = std::string(kKernelDir) + "gather_attn_res_block_stats.cpp";
        gather.source_type = KernelDescriptor::SourceType::FILE_PATH;
        gather.core_ranges = gather_core_set;
        gather.compile_time_args = std::move(gather_ct_args);
        gather.named_compile_time_args = named_ct_args;
        gather.config = ReaderConfigDescriptor{};
        // The caller's arrival semaphore, which `override_runtime_arguments` rewrites on a hit.
        gather.emplace_common_runtime_args({static_cast<uint32_t>(
            semaphore->address())});  // smuggled-rta-ok: caller's GlobalSemaphore, patched per hit
        desc.kernels.push_back(std::move(gather));
        tt::tt_metal::KernelHandle gather_kernel = desc.kernels.size() - 1;
        TT_FATAL(gather_kernel == kGatherKernel, "the gather kernel must be the program's fourth");

        const auto src_node = device->get_fabric_node_id(mesh_coord);
        for (uint32_t link = 0; link < num_gather_cores; ++link) {
            std::vector<tt::tt_fabric::FabricNodeId> dst_nodes;
            std::vector<uint32_t> route_args;
            const auto add_route = [&](const std::optional<ttnn::MeshCoordinate>& neighbor,
                                       uint32_t targets,
                                       const std::array<uint32_t, 6>& route) {
                if (neighbor.has_value() && targets > 0) {
                    dst_nodes.push_back(device->get_fabric_node_id(*neighbor));
                    route_args.insert(route_args.end(), route.begin(), route.end());
                }
            };
            add_route(forward, forward_targets, forward_route);
            add_route(backward, backward_targets, backward_route);

            // The gathered tensor's address (bound, so a cache hit patches it), this rank, this
            // core's link, the routes' connections, then each route.
            std::vector<uint32_t> gather_rt_args = {0u, rank, link};
            gather_rt_args.push_back(static_cast<uint32_t>(dst_nodes.size()));
            tt::tt_fabric::append_routing_plane_connection_manager_rt_args(
                *device,
                src_node,
                dst_nodes,
                std::vector<uint32_t>(dst_nodes.size(), link),
                desc,
                gather_kernel,
                gather_cores.at(link),
                gather_rt_args);
            gather_rt_args.insert(gather_rt_args.end(), route_args.begin(), route_args.end());
            ttnn::ccl::emplace_runtime_args_with_leading_buffers(
                desc.kernels[gather_kernel], gather_cores.at(link), {tensor_return_value.buffer()}, gather_rt_args);
        }
    }

    return desc;
}

}  // namespace

tt::tt_metal::WorkloadDescriptor AttnResBlockStatsProgramFactory::create_workload_descriptor(
    const AttnResBlockStatsParams& operation_attributes,
    const AttnResBlockStatsInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    tt::tt_metal::WorkloadDescriptor workload;
    for (const auto& coord : tensor_coords.coords()) {
        workload.programs.push_back(
            {ttnn::MeshCoordinateRange(coord),
             build_descriptor_at(operation_attributes, coord, tensor_args, tensor_return_value)});
    }
    return workload;
}

void AttnResBlockStatsProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const AttnResBlockStatsParams& operation_attributes,
    const AttnResBlockStatsInputs& /*tensor_args*/,
    Tensor& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_coordinate*/) {
    // The semaphore is left out of the program hash, so a hit may carry another one.
    if (operation_attributes.semaphore.has_value()) {
        GetCommonRuntimeArgs(program, kGatherKernel)[0] = operation_attributes.semaphore->address();
    }
}

}  // namespace ttnn::experimental::prim
