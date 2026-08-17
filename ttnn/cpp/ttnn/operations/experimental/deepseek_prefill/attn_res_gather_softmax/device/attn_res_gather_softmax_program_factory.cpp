// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "attn_res_gather_softmax_program_factory.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <optional>
#include <vector>

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

constexpr auto kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/attn_res_gather_softmax/device/kernels/";

// The row weights `a` and `b`, and the shift and mass they are derived from.
constexpr uint32_t kRowWeights = 2;
constexpr uint32_t kFixedScalars = 2;
// A rank's contribution to the live score: its sum of squares and its dots.
constexpr uint32_t kStatsPerPartial = 2;
constexpr uint32_t kStatsPerRow = 2;
// Ceiling on the gather's staging buffer, in tiles.
constexpr uint32_t kMaxStageTiles = 64;

// Fold cores per token row-tile. Two puts a core's share at half a row, which is
// the widest split whose share still divides the row width.
constexpr uint32_t kFoldCoresPerRow = 2;
// Where the site-derived page offsets sit in each kernel's common runtime args.
constexpr uint32_t kReaderSiteArgIdx = 0;
constexpr uint32_t kWriterSiteArgIdx = 0;
// Where the gather kernel's statistics-tensor address and semaphore address sit.
constexpr uint32_t kGatherStatsAddrIdx = 0;
constexpr uint32_t kGatherSemAddrIdx = 2;

}  // namespace

AttnResGatherSoftmaxSiteOffsets attn_res_gather_softmax_site_offsets(
    const AttnResGatherSoftmaxParams& operation_attributes, const AttnResGatherSoftmaxInputs& tensor_args) {
    const auto& input_shape = tensor_args.partial.padded_shape();
    const uint32_t Wt = input_shape[-1] / TILE_WIDTH;
    const uint32_t Ht = input_shape[-2] / TILE_HEIGHT;

    // A scalar operand is one tile column wide, so its dim-0 plane is Ht pages and
    // selecting a read site is a page offset. Resolved here rather than in the kernel
    // because a shared operand ignores the site, and that test belongs where the
    // shapes are.
    const auto scalar_page_offset = [&](const Tensor& scalar) {
        return scalar.padded_shape()[0] == 1 ? 0u : operation_attributes.site * Ht;
    };

    return {
        .shift = scalar_page_offset(tensor_args.shift),
        .mass = scalar_page_offset(tensor_args.mass),
        .partial = input_shape[0] == 1 ? 0u : operation_attributes.site * Ht * Wt,
    };
}

AttnResGatherSoftmaxMeshWorkloadFactory::cached_program_t AttnResGatherSoftmaxMeshWorkloadFactory::create_at(
    const AttnResGatherSoftmaxParams& operation_attributes,
    const ttnn::MeshCoordinate& mesh_coord,
    const AttnResGatherSoftmaxInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    ttnn::MeshDevice* mesh_device = tensor_args.partial.device();
    auto* const target_device = mesh_device->get_device(mesh_coord);
    const auto mesh_view = mesh_device->get_view();
    TT_FATAL(mesh_view.is_mesh_2d(), "attn_res_gather_softmax requires a 2D mesh to resolve the cluster axis");

    const uint32_t ring_size = operation_attributes.ring_size;
    const uint32_t cluster_axis = operation_attributes.cluster_axis;

    const std::vector<IDevice*> devices = (cluster_axis == 0) ? mesh_view.get_devices_on_column(mesh_coord[1])
                                                              : mesh_view.get_devices_on_row(mesh_coord[0]);
    const auto fabric_node_ids = (cluster_axis == 0) ? mesh_view.get_fabric_node_ids_on_column(mesh_coord[1])
                                                     : mesh_view.get_fabric_node_ids_on_row(mesh_coord[0]);
    TT_FATAL(
        devices.size() == ring_size,
        "ring_size {} does not match the {} devices on cluster axis {}",
        ring_size,
        devices.size(),
        cluster_axis);

    // This chip's rank names the slot of the statistics tensor it fills, and every
    // peer's direction follows from the difference of ranks.
    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id;
    uint32_t my_rank = 0;
    for (uint32_t i = 0; i < ring_size; ++i) {
        if (devices.at(i) != target_device) {
            continue;
        }
        my_rank = i;
        if (i != 0) {
            backward_fabric_node_id = fabric_node_ids.at(i - 1);
        } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
            backward_fabric_node_id = fabric_node_ids.at(ring_size - 1);
        }
        if (i != ring_size - 1) {
            forward_fabric_node_id = fabric_node_ids.at(i + 1);
        } else if (operation_attributes.topology == ttnn::ccl::Topology::Ring) {
            forward_fabric_node_id = fabric_node_ids.at(0);
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto& partial = tensor_args.partial;
    const auto& running_sum = tensor_args.running_sum;
    const auto& shift = tensor_args.shift;
    const auto& mass = tensor_args.mass;
    const auto& q = tensor_args.q;
    const auto& stats = tensor_args.stats;
    auto& output = tensor_return_value.at(0);

    // A deferred residual write, settled inside pass one against the very rows it is
    // about to be reduced over. Pass two then folds the settled stream by distributing
    // the row weight over the two addends — `b*(prefix + pending)` as `b*prefix +
    // b*pending`, two MACs into the accumulator it already uses — so nothing has to wait
    // on the settled tiles reaching DRAM. Only the reduce needs them summed for real:
    // a sum of squares does not distribute.
    const bool fuse_add = tensor_args.pending.has_value();
    const auto& pending = fuse_add ? tensor_args.pending.value() : running_sum;
    auto& total = fuse_add ? tensor_return_value.at(1) : output;

    const auto wide_data_format = datatype_to_dataformat_converter(partial.dtype());
    const auto wide_tile_size = tt::tile_size(wide_data_format);
    const auto scalar_data_format = datatype_to_dataformat_converter(shift.dtype());
    const auto scalar_tile_size = tt::tile_size(scalar_data_format);
    const auto output_data_format = datatype_to_dataformat_converter(output.dtype());
    const auto output_tile_size = tt::tile_size(output_data_format);

    const auto& input_shape = partial.padded_shape();
    const uint32_t Wt = input_shape[-1] / TILE_WIDTH;
    const uint32_t Ht = input_shape[-2] / TILE_HEIGHT;
    const uint32_t kScalars = kFixedScalars + kStatsPerPartial * ring_size;

    const auto site_offsets = attn_res_gather_softmax_site_offsets(operation_attributes, tensor_args);

    // The statistics cross packed by column rather than tile-shaped, so what a packet
    // carries is a page of them — a thousand tokens' worth of one rank's statistic
    // instead of thirty-two. The fabric charges per packet almost regardless of payload,
    // so this is most of what the exchange costs. Kernel-side layout in
    // `kernels/dataflow/attn_res_stats_layout.hpp`.
    // A packed token row is one value per token of a row-tile, so a page holds as many
    // of them as a tile has columns. A worker's scratch has to hold the whole gathered
    // set, which is the larger of the two things it packs.
    const uint32_t packed_row_size = scalar_tile_size / TILE_WIDTH;
    const uint32_t pages_per_plane = (Ht + TILE_WIDTH - 1) / TILE_WIDTH;
    const uint32_t packed_tiles =
        (kStatsPerPartial * ring_size * packed_row_size + scalar_tile_size - 1) / scalar_tile_size;

    const auto max_payload = tt::tt_fabric::get_tt_fabric_max_payload_size_bytes();
    TT_FATAL(
        scalar_tile_size <= max_payload,
        "A {} B statistics page exceeds the {} B fabric payload limit; narrow the statistics dtype",
        scalar_tile_size,
        max_payload);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(target_device->arch(), operation_attributes.compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                            Core Assignment
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::Program program{};
    const auto grid = target_device->compute_with_storage_grid_size();

    // Two splits, because the two passes carry different amounts of parallelism. A
    // row's statistics reduce along the whole of its width, so the core that reduces a
    // row is the only one that can produce it: pass one is capped at Ht cores however
    // large the grid. The fold is tied to nothing once the row weights exist — every
    // output tile is independent — so pass two splits by tile across the rest of the
    // grid. Splitting the fold by row as well would leave 90 of 110 cores idle through
    // the part of the op that moves all the data.
    const uint32_t num_output_tiles = Ht * Wt;
    const auto grid_cores =
        corerange_to_cores(CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1})), std::nullopt, /*row_wise=*/true);
    TT_FATAL(grid_cores.size() > 1, "attn_res_gather_softmax needs a grid with more than one core");

    // One core is held back for the fabric connection. It cannot double as a worker:
    // a link admits a single sender, and that core's two dataflow RISCs are already
    // the reader and the writer.
    const auto max_fold_cores = static_cast<uint32_t>(grid_cores.size() - 1);

    // The fold is not run on the whole grid, because a core's share of the output has
    // to divide the row width to be worth having: the scalars a row needs are fetched
    // once per row a core touches, so a share that straddles a boundary pays twice for
    // the same run of tiles. Only a share that divides `Wt` avoids it, and of those the
    // half-row is the widest — beyond it the split adds cores without shortening the
    // work each one still has to fetch for. Falls back to a whole row per core, then to
    // a ragged split, where the grid cannot seat the aligned choice.
    uint32_t fold_core_budget = kFoldCoresPerRow * Ht;
    if (fold_core_budget > max_fold_cores) {
        fold_core_budget = Ht;
    }
    fold_core_budget = std::min(fold_core_budget, max_fold_cores);
    const CoreRangeSet fold_grid = num_cores_to_corerangeset(fold_core_budget, grid, /*row_wise=*/true);
    const auto [num_fold_cores, all_fold_cores, fold_group_1, fold_group_2, tiles_per_core_1, tiles_per_core_2] =
        tt::tt_metal::split_work_to_cores(fold_grid, num_output_tiles, /*row_wise=*/true);
    const auto [num_stat_cores, all_stat_cores, stat_group_1, stat_group_2, rows_per_core_1, rows_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid, Ht, /*row_wise=*/true);

    // Both splits hand out cores row-wise from the origin and Wt >= 1, so the
    // statistics cores are a prefix of the fold cores. The kernels rely on that: a core
    // that produced statistics is the one that waits for the gather, and a core that
    // did not still has to be released by it.
    TT_FATAL(
        num_stat_cores <= num_fold_cores && all_fold_cores.contains(all_stat_cores),
        "the {} statistics cores must be a subset of the {} fold cores",
        num_stat_cores,
        num_fold_cores);

    const auto fold_cores = corerange_to_cores(all_fold_cores, std::nullopt, /*row_wise=*/true);
    const CoreCoord gather_core = grid_cores.back();
    const CoreRangeSet gather_core_set{CoreRange(gather_core, gather_core)};
    TT_FATAL(!all_fold_cores.contains(gather_core), "the gather core must hold no fold work");
    const CoreRangeSet all_cores = all_fold_cores.merge(gather_core_set);

    ////////////////////////////////////////////////////////////////////////////
    //                            Validation
    ////////////////////////////////////////////////////////////////////////////
    // The fold alternates its srcA operand between the two full-width CBs, which one
    // `bcast_init` configures for both.
    TT_FATAL(
        partial.dtype() == running_sum.dtype(),
        "partial ({}) and running_sum ({}) share one unpacker configuration and must share a dtype",
        partial.dtype(),
        running_sum.dtype());
    // shift, mass and the gathered statistics reach the derivation through one CB.
    TT_FATAL(
        shift.dtype() == mass.dtype() && shift.dtype() == stats.dtype(),
        "shift ({}), mass ({}) and stats ({}) share one circular buffer and must share a dtype",
        shift.dtype(),
        mass.dtype(),
        stats.dtype());
    const auto& stats_shape = stats.padded_shape();
    TT_FATAL(
        stats_shape[1] == kStatsPerPartial * ring_size && stats_shape[-2] == input_shape[-2],
        "stats is {} but the exchange needs [1, {}, {}, 1]: one sum-of-squares and one dots plane per rank",
        stats_shape,
        kStatsPerPartial * ring_size,
        input_shape[-2]);

    // Sized for a statistics core, and allocated on every fold core: the pass-one
    // buffers sit unused on the cores that only fold, which is what lets one kernel
    // binary and one buffer layout serve both roles.
    const uint32_t l1_per_core = Wt * wide_tile_size                         // partial
                                 + Wt * wide_tile_size                       // running_sum
                                 + Wt * wide_tile_size                       // q, resident
                                 + Wt * wide_tile_size                       // the reduce's transformed row
                                 + (fuse_add ? 3 * Wt * wide_tile_size : 0)  // pending, the sum, the sum again
                                 + scalar_tile_size                          // the reduce scaler
                                 + kScalars * scalar_tile_size               // shift, mass, gathered statistics
                                 + kRowWeights * scalar_tile_size + kStatsPerRow * scalar_tile_size +
                                 packed_tiles * scalar_tile_size  // the packed statistics
                                 + 2 * output_tile_size;
    TT_FATAL(
        l1_per_core <= target_device->l1_size_per_core(),
        "attn_res_gather_softmax needs {} B of L1 per core at Wt {} but the core has {} B",
        l1_per_core,
        Wt,
        target_device->l1_size_per_core());

    ////////////////////////////////////////////////////////////////////////////
    //                      Circular Buffers and Semaphores
    ////////////////////////////////////////////////////////////////////////////
    const auto make_cb =
        [&program](
            uint32_t index, const CoreRangeSet& cores, uint32_t num_tiles, uint32_t tile_size, tt::DataFormat format) {
            tt::tt_metal::CircularBufferConfig config =
                tt::tt_metal::CircularBufferConfig(num_tiles * tile_size, {{index, format}})
                    .set_page_size(index, tile_size);
            return tt::tt_metal::CreateCircularBuffer(program, cores, config);
        };

    make_cb(tt::CBIndex::c_0, all_fold_cores, Wt, wide_tile_size, wide_data_format);     // partial
    make_cb(tt::CBIndex::c_1, all_fold_cores, 1, scalar_tile_size, scalar_data_format);  // reduce scaler
    make_cb(tt::CBIndex::c_2, all_fold_cores, Wt, wide_tile_size, wide_data_format);     // q
    make_cb(tt::CBIndex::c_3, all_fold_cores, Wt, wide_tile_size, wide_data_format);     // running_sum
    make_cb(tt::CBIndex::c_4, all_fold_cores, kRowWeights, scalar_tile_size, scalar_data_format);
    make_cb(tt::CBIndex::c_5, all_fold_cores, kScalars, scalar_tile_size, scalar_data_format);
    make_cb(tt::CBIndex::c_6, all_fold_cores, Wt, wide_tile_size, wide_data_format);  // reduce input
    make_cb(tt::CBIndex::c_7, all_fold_cores, kStatsPerRow, scalar_tile_size, scalar_data_format);
    make_cb(tt::CBIndex::c_16, all_fold_cores, 2, output_tile_size, output_data_format);
    // A worker's scratch for the packed statistics: the two columns it parks in pass
    // one, and the whole gathered set it collects in pass two.
    make_cb(tt::CBIndex::c_13, all_fold_cores, packed_tiles, scalar_tile_size, scalar_data_format);

    if (fuse_add) {
        make_cb(tt::CBIndex::c_10, all_fold_cores, Wt, wide_tile_size, wide_data_format);  // pending
        // The settled row, packed twice from one dest register: a circular buffer has a
        // single consumer, and the row is needed by the two reduces here and by the
        // writer that parks it in DRAM. Both hold a whole row so the writer barriers
        // once per row rather than once per tile.
        //
        // Only summing needs them, so they are the one buffer pair not allocated on the
        // fold-only cores: the fold reads the two addends and distributes the weight.
        make_cb(tt::CBIndex::c_11, all_stat_cores, Wt, wide_tile_size, wide_data_format);
        make_cb(tt::CBIndex::c_12, all_stat_cores, Wt, wide_tile_size, wide_data_format);
    }

    // The gather core holds no operand — one route header and one increment header per
    // peer, and a staging buffer for the plane it sends.
    const uint32_t kPeers = ring_size - 1;
    const auto packet_header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    make_cb(tt::CBIndex::c_8, gather_core_set, kStatsPerRow * kPeers, packet_header_size, tt::DataFormat::UInt32);

    // The gather stages a run of pages and sends them behind a single read barrier.
    // A barrier per page would put the exchange on the DRAM latency ladder — one round
    // trip per page, serialized — at exactly the moment every fold core is prefetching
    // against the same DRAM. The cap bounds the buffer for sequences long enough that
    // the whole plane would not fit L1, which the packed layout puts far out of reach:
    // a page carries a thousand tokens.
    const uint32_t stage_tiles = std::min(kStatsPerRow * pages_per_plane, kMaxStageTiles);
    make_cb(tt::CBIndex::c_9, gather_core_set, stage_tiles, scalar_tile_size, scalar_data_format);

    // Local counters, so a program-launch reset is what we want: the workers that
    // increment `ready` and the gather core that increments `done` are all in this
    // program on this chip.
    const auto ready_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);
    const auto done_sem_id = tt::tt_metal::CreateSemaphore(program, all_cores, 0);

    ////////////////////////////////////////////////////////////////////////////
    //                              Kernels
    ////////////////////////////////////////////////////////////////////////////
    // `pending` aliases running_sum where no write was handed in, so the accessor list is
    // the same length either way and the kernel's compile-time offsets do not move. The
    // kernel never reads through it there.
    std::vector<uint32_t> reader_ct_args = {Wt, static_cast<uint32_t>(fuse_add)};
    TensorAccessorArgs(*running_sum.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*partial.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*q.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*pending.buffer()).append_to(reader_ct_args);

    const auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "dataflow/reader_attn_res_gather_softmax.cpp",
        all_fold_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args = {
        Wt, ring_size, Ht, ready_sem_id, done_sem_id, static_cast<uint32_t>(fuse_add)};
    TensorAccessorArgs(*stats.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*shift.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*mass.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);
    TensorAccessorArgs(*total.buffer()).append_to(writer_ct_args);

    const auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "dataflow/writer_attn_res_gather_softmax.cpp",
        all_fold_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_ct_args));

    const std::vector<uint32_t> compute_ct_args = {
        Wt,
        ring_size,
        std::bit_cast<uint32_t>(operation_attributes.inv_hidden_size),
        std::bit_cast<uint32_t>(operation_attributes.eps),
        static_cast<uint32_t>(fuse_add)};

    const auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "compute/attn_res_gather_softmax.cpp",
        all_fold_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_ct_args});

    std::vector<uint32_t> gather_ct_args = {
        ring_size, Ht, static_cast<uint32_t>(num_stat_cores), ready_sem_id, done_sem_id, stage_tiles};
    TensorAccessorArgs(*stats.buffer()).append_to(gather_ct_args);

    const auto gather_kernel_id = tt::tt_metal::CreateKernel(
        program,
        std::string(kKernelDir) + "dataflow/gather_attn_res_gather_softmax.cpp",
        gather_core_set,
        tt::tt_metal::ReaderDataMovementConfig(gather_ct_args));

    ////////////////////////////////////////////////////////////////////////////
    //                           Runtime Arguments
    ////////////////////////////////////////////////////////////////////////////
    tt::tt_metal::SetCommonRuntimeArgs(program, reader_kernel_id, std::vector<uint32_t>{site_offsets.partial});
    tt::tt_metal::SetCommonRuntimeArgs(
        program, writer_kernel_id, std::vector<uint32_t>{site_offsets.shift, site_offsets.mass});

    const auto gather_physical = mesh_device->worker_core_from_logical_core(gather_core);
    uint32_t start_row = 0;
    uint32_t start_tile = 0;
    for (uint32_t i = 0; i < num_fold_cores; ++i) {
        const CoreCoord& core = fold_cores.at(i);
        const uint32_t rows =
            (i < num_stat_cores) ? (stat_group_1.contains(core) ? rows_per_core_1 : rows_per_core_2) : 0;
        const uint32_t tiles = fold_group_1.contains(core) ? tiles_per_core_1 : tiles_per_core_2;

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {running_sum.buffer()->address(),
             partial.buffer()->address(),
             q.buffer()->address(),
             rows,
             start_row,
             tiles,
             start_tile,
             pending.buffer()->address()});

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {stats.buffer()->address(),
             shift.buffer()->address(),
             mass.buffer()->address(),
             output.buffer()->address(),
             rows,
             start_row,
             tiles,
             start_tile,
             my_rank,
             gather_physical.x,
             gather_physical.y,
             total.buffer()->address()});

        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {rows, tiles, start_tile});
        start_row += rows;
        start_tile += tiles;
    }

    // The release is a multicast per rectangle of the fold set rather than one
    // increment per core, so the fold's cost in cores does not become a cost in
    // wake-up latency. The gather core sits outside every rectangle by construction,
    // so it is not among the destinations it counts.
    std::vector<uint32_t> release_args;
    const auto fold_ranges = all_fold_cores.merge_ranges().ranges();
    for (const auto& range : fold_ranges) {
        const auto start_physical = mesh_device->worker_core_from_logical_core(range.start_coord);
        const auto end_physical = mesh_device->worker_core_from_logical_core(range.end_coord);
        release_args.push_back(start_physical.x);
        release_args.push_back(start_physical.y);
        release_args.push_back(end_physical.x);
        release_args.push_back(end_physical.y);
        release_args.push_back(static_cast<uint32_t>(range.size()));
    }

    std::vector<uint32_t> gather_rt_args = {
        stats.buffer()->address(),
        my_rank,
        static_cast<uint32_t>(operation_attributes.semaphore.address()),
        static_cast<uint32_t>(fold_ranges.size())};
    gather_rt_args.insert(gather_rt_args.end(), release_args.begin(), release_args.end());

    // Forward then backward, each preceded by its presence flag: the order
    // FabricConnectionManager::build_from_args reads them in.
    const auto src_fabric_node_id = mesh_device->get_fabric_node_id(mesh_coord);
    gather_rt_args.push_back(forward_fabric_node_id.has_value());
    if (forward_fabric_node_id.has_value()) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fabric_node_id, forward_fabric_node_id.value(), 0, program, gather_core, gather_rt_args);
    }
    gather_rt_args.push_back(backward_fabric_node_id.has_value());
    if (backward_fabric_node_id.has_value()) {
        tt::tt_fabric::append_fabric_connection_rt_args(
            src_fabric_node_id, backward_fabric_node_id.value(), 0, program, gather_core, gather_rt_args);
    }

    // A peer's route, in the order the gather kernel takes its peers: the peer's node
    // outright, so the connection above only chooses the first hop. The fabric config
    // also decides the packet header type the kernel reads these back with, so host and
    // device stay in step without a flag — and only a 2D config pairs with the header
    // this kernel is compiled against.
    const auto fabric_config = tt::tt_fabric::GetFabricConfig();
    TT_FATAL(
        tt::tt_fabric::is_2d_fabric_config(fabric_config),
        "attn_res_gather_softmax needs a 2D fabric, got {}",
        fabric_config);
    for (uint32_t p = 0; p < ring_size; ++p) {
        if (p == my_rank) {
            continue;
        }
        gather_rt_args.push_back(*fabric_node_ids.at(p).mesh_id);
        gather_rt_args.push_back(fabric_node_ids.at(p).chip_id);
    }
    tt::tt_metal::SetRuntimeArgs(program, gather_kernel_id, gather_core, gather_rt_args);

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .gather_kernel_id = gather_kernel_id,
         .fold_cores = fold_cores,
         .gather_core = gather_core}};
}

AttnResGatherSoftmaxMeshWorkloadFactory::cached_mesh_workload_t
AttnResGatherSoftmaxMeshWorkloadFactory::create_mesh_workload(
    const AttnResGatherSoftmaxParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const AttnResGatherSoftmaxInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& mesh_coord_range : tensor_coords.ranges()) {
        for (const auto& mesh_coord : mesh_coord_range) {
            const ttnn::MeshCoordinateRange single_coord_range{mesh_coord, mesh_coord};
            auto cached_program = create_at(operation_attributes, mesh_coord, tensor_args, tensor_return_value);
            shared_variables[single_coord_range] = std::move(cached_program.shared_variables);
            mesh_workload.add_program(single_coord_range, std::move(cached_program.program));
        }
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void AttnResGatherSoftmaxMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const AttnResGatherSoftmaxParams& operation_attributes,
    const AttnResGatherSoftmaxInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const auto site_offsets = attn_res_gather_softmax_site_offsets(operation_attributes, tensor_args);
    const bool fuse_add = tensor_args.pending.has_value();
    const auto pending_addr =
        fuse_add ? tensor_args.pending->buffer()->address() : tensor_args.running_sum.buffer()->address();
    const auto total_addr =
        fuse_add ? tensor_return_value.at(1).buffer()->address() : tensor_return_value.at(0).buffer()->address();

    for (auto& [range, shared_vars] : cached_workload.shared_variables) {
        auto& program = cached_workload.workload.get_programs().at(range);

        // The site is a page offset, so a new site is a common-arg patch rather than a
        // new program. That is what keeps one cached program serving every read site.
        auto& reader_common = tt::tt_metal::GetCommonRuntimeArgs(program, shared_vars.reader_kernel_id);
        reader_common[kReaderSiteArgIdx] = site_offsets.partial;
        auto& writer_common = tt::tt_metal::GetCommonRuntimeArgs(program, shared_vars.writer_kernel_id);
        writer_common[kWriterSiteArgIdx] = site_offsets.shift;
        writer_common[kWriterSiteArgIdx + 1] = site_offsets.mass;

        auto& reader_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
        auto& writer_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);
        for (const auto& core : shared_vars.fold_cores) {
            auto& reader_args = reader_args_by_core[core.x][core.y];
            reader_args[0] = tensor_args.running_sum.buffer()->address();
            reader_args[1] = tensor_args.partial.buffer()->address();
            reader_args[2] = tensor_args.q.buffer()->address();
            reader_args[7] = pending_addr;

            auto& writer_args = writer_args_by_core[core.x][core.y];
            writer_args[0] = tensor_args.stats.buffer()->address();
            writer_args[1] = tensor_args.shift.buffer()->address();
            writer_args[2] = tensor_args.mass.buffer()->address();
            writer_args[3] = tensor_return_value.at(0).buffer()->address();
            writer_args[11] = total_addr;
        }

        auto& gather_args_by_core = GetRuntimeArgs(program, shared_vars.gather_kernel_id);
        auto& gather_args = gather_args_by_core[shared_vars.gather_core.x][shared_vars.gather_core.y];
        gather_args[kGatherStatsAddrIdx] = tensor_args.stats.buffer()->address();
        gather_args[kGatherSemAddrIdx] = operation_attributes.semaphore.address();
    }
}

}  // namespace ttnn::experimental::prim
