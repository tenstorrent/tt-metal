// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "wan_fused_distributed_rmsnorm_program_factory.hpp"

#include <algorithm>
#include <cstring>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"

using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::experimental::prim {

// =============================================================================
// Program factory for the fused Wan2.2 distributed RMSNorm op.
//
// Two execution modes:
//
//   TP=1 (ring_size==1): the compute kernel pushes per-row stats directly into
//     stats_gathered_cb (is_tp_1 compile-time flag); the writer kernel just
//     drains output_cb to DRAM. No fabric setup. Multiple worker cores per
//     chip OK — they each take a slice of rows.
//
//   TP>1 (ring_size>1) — MUX-based multi-worker ring AG:
//     - num_workers_per_chip worker cores per chip (default 2).
//     - 1 fabric MUX core per direction with a valid neighbor (so 0, 1, or 2
//       MUX cores per chip). Each MUX has num_workers_per_chip channels (one
//       per worker).
//     - Worker layout: [worker_0, ..., worker_N-1, fwd_mux, bwd_mux].
//     - Each worker connects to BOTH MUX cores (its channel_id == worker_id)
//       and uses `fabric_multicast_noc_fused_unicast_with_atomic_inc` to
//       multicast its stats tile to every other chip in the ring axis. The
//       remote chip's matching-position worker (same noc_x, noc_y) receives
//       the data + atomic_inc on its GlobalSemaphore.
//     - Termination: one designated worker per MUX acts as termination master
//       and graceful-terminates the MUX once all peer workers have signaled
//       completion. See `wan_rmsnorm_fused_writer_mux.cpp` for the kernel-side
//       handshake.
// =============================================================================

namespace {

uint32_t float_to_u32(float v) {
    uint32_t out;
    std::memcpy(&out, &v, sizeof(float));
    return out;
}

}  // namespace

// Upper cap on TP>1 worker cores per chip. The MUX channel count is uint8_t
// but in practice the fabric MUX rejects (or deadlocks) above ~64 full-size
// channels per core. For big shapes (Wan N=18944 with ~592 tile rows),
// more workers means fewer chunks-per-worker, which keeps each worker's
// chunk loop short and the per-chunk fabric overhead amortized.
constexpr uint32_t kMaxMuxWorkersPerChip = 64u;
// num_tile_rows below this falls back to the LEGACY whole-tile writer
// (single worker). The packed-page MUX writer has significant per-chunk
// fabric overhead that doesn't pay off until we have ≥4 tile-rows worth
// of compute per chip — at that point parallelism + packed bytes win.
constexpr uint32_t kMuxRowsThreshold = 4u;

// Pick num_workers for the MUX/packed path. Two regimes:
//
//  - SMALL (num_tile_rows ≤ kSmallShapeRowsLimit): per-worker compute is
//    tiny, so prioritize compute parallelism. Use max(num_tile_rows,
//    kMaxMuxWorkersPerChip). This wins for prodlike shapes (N=256 H=640pd).
//
//  - LARGE (num_tile_rows > kSmallShapeRowsLimit): per-worker compute is
//    substantial. Use rows/2 workers so chunks pack 2 rows per fabric
//    packet — halving the packet count per chip and reducing fabric
//    overhead. Cap at kMaxMuxWorkersPerChip. This wins for multichunk
//    shapes (N=512+ at H=64pd).
constexpr uint32_t kSmallShapeRowsLimit = 8u;
uint32_t pick_num_workers_tp_gt_1(uint32_t num_tile_rows) {
    if (num_tile_rows < kMuxRowsThreshold) {
        return 1u;
    }
    if (num_tile_rows <= kSmallShapeRowsLimit) {
        return std::min<uint32_t>(kMaxMuxWorkersPerChip, num_tile_rows);
    }
    const uint32_t target = std::max(1u, num_tile_rows / 2u);
    return std::min<uint32_t>(target, kMaxMuxWorkersPerChip);
}

// Sizing derivation used in both spec computation (to size the stats scratch
// tensor in `compute_output_specs`) and the program factory (to lay out
// kernels + CBs). Single source of truth so the two cannot drift.
WanFusedDistributedRmsnormSizing compute_sizing(const WanFusedDistributedRmsnormParams& args, const Tensor& input) {
    WanFusedDistributedRmsnormSizing s;
    const auto& padded = input.padded_shape();
    const uint32_t W = padded[-1];
    const uint32_t folded_H = input.physical_volume() / W;
    s.num_tile_rows = folded_H / TILE_HEIGHT;
    s.is_tp_1 = (args.ring_size == 1);
    s.num_workers = s.is_tp_1 ? 1u : pick_num_workers_tp_gt_1(s.num_tile_rows);
    s.use_mux = !s.is_tp_1 && (s.num_workers > 1);
    const uint32_t rows_per_worker = tt::div_up(s.num_tile_rows, s.num_workers);
    // Phase 9 packed-AG: one fabric mcast per chunk, so fewer chunks = fewer
    // fabric round-trips. Aim for 1 chunk per worker (= rows_per_worker rows
    // per packet) for the multichunk shape regime where each worker already
    // has few rows. Cap at kMaxChunkSizeRows for L1 budget (chunk-sized CBs:
    // input, stats_local, packed_gathered, stats_gathered all scale with
    // chunk_size).
    constexpr uint32_t kMaxChunkSizeRows = 8u;
    // L1 budget cap: input_cb is double-buffered 2 * chunk * num_tile_cols
    // bf16 tiles = chunk * num_tile_cols * 4 KB per worker. Other CBs add
    // ~150 KB. Keep input_cb ≤ 512 KB so total ≤ 750 KB (half of L1):
    //   chunk * num_tile_cols ≤ 128.
    const uint32_t num_tile_cols_for_chunk_cap = std::max(1u, W / TILE_WIDTH);
    const uint32_t chunk_h_cap = std::max(1u, 128u / num_tile_cols_for_chunk_cap);
    s.chunk_size_rows =
        std::min<uint32_t>(std::min<uint32_t>(std::max(1u, rows_per_worker), kMaxChunkSizeRows), chunk_h_cap);
    s.window_size = s.chunk_size_rows;
    // Pages are addressed across the whole chip (not per-worker) so the
    // buffer shape doesn't depend on num_workers. Each chunk a worker
    // produces lands at a page index derived only from (device_idx, chunk
    // index on this chip), which lets the caller spec the buffer without
    // knowing the worker count.
    s.num_chunks_per_device = s.use_mux ? tt::div_up(s.num_tile_rows, s.window_size) : 0u;
    s.total_pages = s.use_mux ? args.ring_size * s.num_chunks_per_device : 0u;
    s.page_size_bytes = s.use_mux ? TILE_HEIGHT * s.window_size * sizeof(float) : 0u;
    return s;
}

WanFusedDistributedRmsnormMeshWorkloadFactory::cached_program_t
WanFusedDistributedRmsnormMeshWorkloadFactory::create_at(
    const WanFusedDistributedRmsnormParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const WanFusedDistributedRmsnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    Tensor& output_tensor = tensor_return_value.at(0);
    // Stats DRAM scratch: only allocated for the MUX writer path (TP>1 with
    // multiple workers). When not allocated (TP=1 or num_workers=1) the
    // legacy writer doesn't reference it.
    Tensor* stats_dram_tensor = tensor_return_value.size() > 1 ? &tensor_return_value[1] : nullptr;
    const auto& input_tensor = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& bias = tensor_args.bias;
    const auto& trans_mat = tensor_args.transformation_mat;
    const auto& rope_cos = tensor_args.rope_cos;
    const auto& rope_sin = tensor_args.rope_sin;

    Program program = CreateProgram();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t W = input_shape[-1];                            // hidden dim per device
    const uint32_t folded_H = input_tensor.physical_volume() / W;  // total seq rows
    const uint32_t num_tile_cols = W / TILE_WIDTH;
    const uint32_t num_tile_rows = folded_H / TILE_HEIGHT;

    const uint32_t num_heads_per_device = args.num_heads_per_device;
    const uint32_t head_dim = W / num_heads_per_device;
    const uint32_t head_dim_tiles = head_dim / TILE_WIDTH;

    const bool has_weight = weight.has_value();
    const bool has_bias = bias.has_value();
    const bool fuse_rope = trans_mat.has_value() && rope_cos.has_value() && rope_sin.has_value();

    // Per-head RoPE: cos/sin shape[1] == num_heads_per_device gives each head
    // its own cos/sin block. shape[1] == 1 means broadcast (current default).
    // rope_seqlen_tiles = cos sequence length in tiles, used by reader as the
    // per-head stride when computing tile indices.
    const bool per_head_rope = fuse_rope && (rope_cos->logical_shape()[1] == num_heads_per_device);
    const uint32_t rope_seqlen_tiles = fuse_rope ? (rope_cos->logical_shape()[2] / TILE_HEIGHT) : 0u;
    // Per-row RoPE push count: num_tile_cols for per-head (all heads' cos for
    // this row, packed contiguously), head_dim_tiles for broadcast.
    const uint32_t rope_tiles_per_row = per_head_rope ? num_tile_cols : head_dim_tiles;

    // ------------------------------------------------------------------------
    // Ring topology: my position + forward/backward fabric neighbors.
    // ------------------------------------------------------------------------
    auto* mesh_device = input_tensor.device();

    std::optional<ttnn::MeshCoordinate> forward_coord = std::nullopt;
    std::optional<ttnn::MeshCoordinate> backward_coord = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> forward_fabric_node_id = std::nullopt;
    std::optional<tt::tt_fabric::FabricNodeId> backward_fabric_node_id = std::nullopt;
    uint32_t device_index = 0;
    if (args.ring_size > 1) {
        forward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, mesh_coordinate, /*offset=*/1, args.topology, args.cluster_axis);
        backward_coord = ttnn::ccl::get_physical_neighbor_from_physical_coord(
            input_tensor, mesh_coordinate, /*offset=*/-1, args.topology, args.cluster_axis);
        device_index =
            ttnn::ccl::get_linearized_index_from_physical_coord(input_tensor, mesh_coordinate, args.cluster_axis);
        if (forward_coord.has_value()) {
            forward_fabric_node_id = mesh_device->get_fabric_node_id(forward_coord.value());
        }
        if (backward_coord.has_value()) {
            backward_fabric_node_id = mesh_device->get_fabric_node_id(backward_coord.value());
        }
    }

    uint32_t num_targets_forward = 0;
    uint32_t num_targets_backward = 0;
    if (args.ring_size > 1) {
        if (args.topology == ttnn::ccl::Topology::Linear) {
            ttnn::ccl::LineTopology line_topology(args.ring_size, device_index);
            num_targets_forward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::FORWARD);
            num_targets_backward = line_topology.get_distance_to_end_of_line(ttnn::ccl::LineDirection::BACKWARD);
        } else if (args.topology == ttnn::ccl::Topology::Ring) {
            num_targets_forward = tt::div_up(args.ring_size - 1, 2);
            num_targets_backward = args.ring_size - 1 - num_targets_forward;
            if (device_index % 2 == 0) {
                std::swap(num_targets_forward, num_targets_backward);
            }
        }
    }

    const bool is_tp_1 = (args.ring_size == 1);
    // Will be finalized after we pick num_workers below — set initial estimate.
    bool use_mux = !is_tp_1;

    // ------------------------------------------------------------------------
    // Core allocation
    // ------------------------------------------------------------------------
    IDevice* device = input_tensor.device();
    const auto grid_size = device->compute_with_storage_grid_size();
    const auto core_grid = CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    const uint32_t max_cores = core_grid.size();

    uint32_t num_workers;
    if (is_tp_1) {
        num_workers = std::min<uint32_t>(max_cores, num_tile_rows);
    } else {
        // TP>1: heuristic picks 1 worker for small shapes (MUX overhead beats
        // parallelism), more workers for larger shapes. When num_workers=1 we
        // SKIP MUX entirely and use the legacy direct-fabric writer.
        num_workers = pick_num_workers_tp_gt_1(num_tile_rows);
    }
    use_mux = !is_tp_1 && (num_workers > 1);

    // Multi-link MUX: allocate one MUX core per (direction, link). Workers are
    // partitioned round-robin: worker i uses link (i % num_links_eff) for both
    // its fwd and bwd MUX. We round num_workers down to a multiple of
    // num_links_eff so each link's MUX has the same number of clients (the
    // num_mux_clients CT arg in the writer is a single value).
    const uint32_t num_links_requested = std::max<uint32_t>(1u, args.num_links);
    uint32_t num_links_eff = use_mux ? std::min<uint32_t>(num_links_requested, num_workers) : 1u;
    if (use_mux && num_links_eff > 1) {
        num_workers = (num_workers / num_links_eff) * num_links_eff;
        if (num_workers == 0) {
            num_workers = num_links_eff;
        }
    }

    const bool fwd_mux_valid = use_mux && forward_fabric_node_id.has_value();
    const bool bwd_mux_valid = use_mux && backward_fabric_node_id.has_value();
    const uint32_t num_mux_per_direction = use_mux ? num_links_eff : 0u;
    const uint32_t num_mux_cores =
        (fwd_mux_valid ? num_mux_per_direction : 0u) + (bwd_mux_valid ? num_mux_per_direction : 0u);
    const uint32_t total_cores_needed = num_workers + num_mux_cores;
    TT_FATAL(
        total_cores_needed <= max_cores,
        "wan_fused_distributed_rmsnorm needs {} cores ({} workers + {} mux) but only {} available",
        total_cores_needed,
        num_workers,
        num_mux_cores,
        max_cores);

    const uint32_t num_tile_rows_per_worker = tt::div_up(num_tile_rows, num_workers);

    // Layout: [worker_0..worker_N-1, fwd_mux_0..fwd_mux_L-1, bwd_mux_0..bwd_mux_L-1]
    // where L = num_links_eff (only if that direction is valid).
    const auto all_cores_vec = corerange_to_cores(core_grid, max_cores, /*row_major=*/true);
    std::vector<CoreCoord> worker_cores(all_cores_vec.begin(), all_cores_vec.begin() + num_workers);
    std::vector<CoreCoord> fwd_mux_cores;  // size = num_links_eff if valid
    std::vector<CoreCoord> bwd_mux_cores;
    {
        uint32_t next_core_idx = num_workers;
        if (fwd_mux_valid) {
            for (uint32_t lnk = 0; lnk < num_mux_per_direction; lnk++) {
                fwd_mux_cores.push_back(all_cores_vec[next_core_idx++]);
            }
        }
        if (bwd_mux_valid) {
            for (uint32_t lnk = 0; lnk < num_mux_per_direction; lnk++) {
                bwd_mux_cores.push_back(all_cores_vec[next_core_idx++]);
            }
        }
    }

    CoreRangeSet worker_core_set;
    for (const auto& c : worker_cores) {
        worker_core_set = worker_core_set.merge(CoreRangeSet({CoreRange(c, c)}));
    }

    // chunk_size_rows: aim for ≥2 chunks per worker so Phase 5's double-buffered
    // input_cb can overlap chunk N+1's reader fill + chunk N+1's AG with chunk
    // N's compute and chunk N's output drain. When the worker has ≥2 rows, pick
    // ceil(rows/2) as the chunk size (capped at kMaxChunkSizeRows); when only
    // 1 row, chunk=1 (no overlap possible at all).
    constexpr uint32_t kMaxChunkSizeRows = 8u;
    // L1 budget cap (matches compute_sizing): chunk * num_tile_cols ≤ 128
    // keeps input_cb under ~512 KB per worker.
    const uint32_t chunk_h_cap = std::max(1u, 128u / std::max(1u, num_tile_cols));
    const uint32_t chunk_size_rows =
        std::min<uint32_t>(std::min<uint32_t>(std::max(1u, num_tile_rows_per_worker), kMaxChunkSizeRows), chunk_h_cap);
    // Phase 9 packed-page AG: every chunk this chip processes maps to a
    // distinct DRAM page. Page index = my_device_index * num_chunks_per_device
    // + chunk_idx_on_device. Independent of num_workers per design.
    const uint32_t num_chunks_per_device = use_mux ? tt::div_up(num_tile_rows, chunk_size_rows) : 0u;

    // ------------------------------------------------------------------------
    // Compute kernel config + dtype/format setup
    // ------------------------------------------------------------------------
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), args.compute_kernel_config);
    const uint32_t dst_reg_count = get_dest_reg_count(args.compute_kernel_config);
    const uint32_t block_size = dst_reg_count;

    const tt::DataFormat input_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const DataType output_dtype = args.dtype.value_or(input_tensor.dtype());
    const tt::DataFormat output_format = datatype_to_dataformat_converter(output_dtype);
    const tt::DataFormat fp32_format = tt::DataFormat::Float32;
    const tt::DataFormat bf16_format = tt::DataFormat::Float16_b;

    const uint32_t input_tile_size = tt::tile_size(input_format);
    const uint32_t output_tile_size = tt::tile_size(output_format);
    const uint32_t fp32_tile_size = tt::tile_size(fp32_format);
    const uint32_t bf16_tile_size = tt::tile_size(bf16_format);

    // ------------------------------------------------------------------------
    // Persistent DRAM stats buffer (Phase 1, TP>1 only).
    //
    // The previous design had each worker fabric-mcast stats tiles directly
    // into the remote chip's `stats_gathered_cb` L1 region. That's unsafe
    // across ops: another op on the remote chip could be using that L1 range
    // by the time our packet arrives. With a persistent DRAM buffer, the
    // remote chip's DRAM is the destination — a fixed allocation that's only
    // touched by this op. After AG completes (sem hits ring_size-1 per row)
    // the writer reads the gathered stats from local DRAM into stats_gathered_cb
    // for compute consumption.
    //
    // Layout: [num_workers, chunk_size_rows, ring_size] tiles, fp32.
    //   Worker i, row r, device d → tile_idx = i * chunk × ring + r * ring + d.
    // The buffer is a regular device tensor (allocated by the framework via
    // `create_output_tensors` → `create_device_tensor`). On a MeshDevice that
    // gives a mesh-coherent MeshBuffer allocation: every chip sees the same
    // DRAM address — required so the fabric mcast's NoC address resolves to
    // the matching tile on every remote chip. (A per-chip `tt_metal::CreateBuffer`
    // does NOT give that guarantee on submeshes; that mismatch caused TP=2
    // LINE NaN bugs where fabric mcasts landed at the wrong remote page.)
    TT_FATAL(
        !use_mux || stats_dram_tensor != nullptr,
        "create_at requires a stats DRAM scratch tensor at tensor_return_value[1] when use_mux is true");
    tt::tt_metal::Buffer* stats_dram_buffer = use_mux ? stats_dram_tensor->buffer() : nullptr;

    // ------------------------------------------------------------------------
    // CB allocations (on worker cores)
    // ------------------------------------------------------------------------
    constexpr uint32_t input_cb_id = tt::CBIndex::c_0;
    constexpr uint32_t stats_local_cb_id = tt::CBIndex::c_1;
    constexpr uint32_t stats_gathered_cb_id = tt::CBIndex::c_2;
    constexpr uint32_t weight_cb_id = tt::CBIndex::c_3;
    constexpr uint32_t reduce_scalar_sum_cb_id = tt::CBIndex::c_4;
    constexpr uint32_t reduce_scalar_avg_cb_id = tt::CBIndex::c_5;
    constexpr uint32_t epsilon_cb_id = tt::CBIndex::c_6;
    constexpr uint32_t reduce_result_cb_id = tt::CBIndex::c_7;
    constexpr uint32_t intermediate_cb_id = tt::CBIndex::c_8;
    constexpr uint32_t pre_intermediate_cb_id = tt::CBIndex::c_9;
    constexpr uint32_t output_cb_id = tt::CBIndex::c_10;
    constexpr uint32_t transformation_mat_cb_id = tt::CBIndex::c_11;
    constexpr uint32_t rope_cos_cb_id = tt::CBIndex::c_12;
    constexpr uint32_t rope_sin_cb_id = tt::CBIndex::c_13;
    constexpr uint32_t rotated_input_cb_id = tt::CBIndex::c_14;
    constexpr uint32_t reserved_packet_header_cb_id = tt::CBIndex::c_15;
    // Phase 9 packed-page AG (use_mux only):
    //   stats_transposed_local_cb : window_size fp32 tiles, post-transpose.
    //       Real per-token sum-of-squares now lives in row 0 of each tile
    //       (2 contiguous 64-byte spans at face_00[0] and face_01[0]).
    //       Compute pushes, writer pops.
    //   stats_packed_local_cb     : 2 staging slots of page_size_bytes
    //       (= TILE_HEIGHT * window_size * sizeof(float)). Writer packs row 0
    //       of W tiles into one row-major span, then fabric-mcasts that
    //       single packet. Double-buffered so it can prep packet N+1 while
    //       packet N is in flight.
    //   stats_packed_gathered_cb  : ring_size slots of page_size_bytes.
    //       Writer NoC-reads the (ring_size-1) remote-device pages from DRAM
    //       here; the local-device page is L1-copied from stats_packed_local
    //       (Phase 1.1 skip-local-roundtrip pattern, applied at the page
    //       granularity).
    //   stats_transposed_gathered_cb : ring_size fp32 tiles. Compute
    //       transposes each row-0 gathered tile (stats_gathered_cb) back to
    //       col 0 here, so the existing post-reduce<AVG,REDUCE_ROW> chain
    //       runs unchanged on it.
    constexpr uint32_t stats_transposed_local_cb_id = tt::CBIndex::c_16;
    constexpr uint32_t stats_packed_local_cb_id = tt::CBIndex::c_17;
    constexpr uint32_t stats_packed_gathered_cb_id = tt::CBIndex::c_18;
    constexpr uint32_t stats_transposed_gathered_cb_id = tt::CBIndex::c_19;
    constexpr uint32_t bias_cb_id = tt::CBIndex::c_20;

    // Double-buffer input_cb (Phase 5): reader can fill chunk N+1 while compute
    // is in chunk N's post phase. Cumulative cb_wait_front in compute (Phase 4)
    // pairs naturally with this — compute uses absolute indices within the
    // current chunk, and cb_pop_front at end of chunk frees that chunk's slots
    // back to the reader.
    const uint32_t chunk_input_tiles = chunk_size_rows * num_tile_cols;
    const uint32_t input_cb_tiles = 2 * chunk_input_tiles;
    create_cb(input_cb_id, program, worker_core_set, input_tile_size, input_cb_tiles, input_format);

    const uint32_t stats_local_tiles = (args.ring_size > 1) ? chunk_size_rows : 1;
    create_cb(stats_local_cb_id, program, worker_core_set, fp32_tile_size, stats_local_tiles, fp32_format);
    const uint32_t stats_gathered_tiles = chunk_size_rows * args.ring_size;
    create_cb(stats_gathered_cb_id, program, worker_core_set, fp32_tile_size, stats_gathered_tiles, fp32_format);

    // Phase 9 packed-page AG: dedicated CBs only when the MUX writer runs.
    // The legacy single-worker writer still uses the old whole-tile AG path
    // so it doesn't need these; give it 1-slot stubs so compile-time
    // arguments stay valid across both paths.
    {
        const uint32_t window = use_mux ? chunk_size_rows : 1u;
        const uint32_t ring = use_mux ? args.ring_size : 1u;
        const uint32_t page_size_bytes = use_mux ? TILE_HEIGHT * window * sizeof(float) : fp32_tile_size;
        create_cb(stats_transposed_local_cb_id, program, worker_core_set, fp32_tile_size, window, fp32_format);
        // packed CBs use page_size_bytes per slot, not the fp32 tile size.
        tt::tt_metal::CircularBufferConfig packed_local_cfg =
            tt::tt_metal::CircularBufferConfig(2u * page_size_bytes, {{stats_packed_local_cb_id, fp32_format}})
                .set_page_size(stats_packed_local_cb_id, page_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, worker_core_set, packed_local_cfg);
        tt::tt_metal::CircularBufferConfig packed_gathered_cfg =
            tt::tt_metal::CircularBufferConfig(ring * page_size_bytes, {{stats_packed_gathered_cb_id, fp32_format}})
                .set_page_size(stats_packed_gathered_cb_id, page_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, worker_core_set, packed_gathered_cfg);
        create_cb(stats_transposed_gathered_cb_id, program, worker_core_set, fp32_tile_size, ring, fp32_format);
    }

    if (has_weight) {
        create_cb(weight_cb_id, program, worker_core_set, bf16_tile_size, num_tile_cols, bf16_format);
    } else {
        create_cb(weight_cb_id, program, worker_core_set, bf16_tile_size, 1, bf16_format);
    }
    if (has_bias) {
        create_cb(bias_cb_id, program, worker_core_set, bf16_tile_size, num_tile_cols, bf16_format);
    } else {
        create_cb(bias_cb_id, program, worker_core_set, bf16_tile_size, 1, bf16_format);
    }

    create_cb(reduce_scalar_sum_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    create_cb(reduce_scalar_avg_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    create_cb(epsilon_cb_id, program, worker_core_set, bf16_tile_size, 1, bf16_format);
    create_cb(reduce_result_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    create_cb(pre_intermediate_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    create_cb(transformation_mat_cb_id, program, worker_core_set, bf16_tile_size, 1, bf16_format);

    if (fuse_rope) {
        // Size rope CBs to hold a whole chunk's worth — reader pushes
        // rope_tiles_per_row per row eagerly, compute pops them only at end
        // of each row in post phase. Without chunk-sized buffering the reader
        // blocks at row 1 and chunk_size_rows>1 input never fully arrives.
        // For per-head RoPE, rope_tiles_per_row = num_tile_cols
        // (num_heads_per_device * head_dim_tiles); for the broadcast default
        // it's just head_dim_tiles.
        const uint32_t rope_cb_tiles = chunk_size_rows * rope_tiles_per_row;
        create_cb(rope_cos_cb_id, program, worker_core_set, fp32_tile_size, rope_cb_tiles, fp32_format);
        create_cb(rope_sin_cb_id, program, worker_core_set, fp32_tile_size, rope_cb_tiles, fp32_format);
    } else {
        create_cb(rope_cos_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
        create_cb(rope_sin_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    }

    // intermediate_cb and rotated_input_cb are compute-only (producer and
    // consumer are the same TRISC pipeline within the post phase). Phase 7
    // restructures the post phase to do each sub-phase (mul-rms, weight,
    // matmul, cos-mul, sin-mul, add) across ALL col-blocks before moving to
    // the next, requiring these CBs to hold a full row between sub-phases.
    // The pack always pushes block_size tiles (LLK packer requirement) even
    // when only `tiles_in_block = num_tile_cols % block_size` are valid, so
    // we round the CB size UP to a multiple of block_size.
    const uint32_t intermediate_cb_tiles = tt::div_up(num_tile_cols, block_size) * block_size;
    create_cb(intermediate_cb_id, program, worker_core_set, bf16_tile_size, intermediate_cb_tiles, bf16_format);
    create_cb(rotated_input_cb_id, program, worker_core_set, bf16_tile_size, intermediate_cb_tiles, bf16_format);
    // output_cb is double-buffered so the writer can drain block N while compute
    // produces block N+1.
    create_cb(output_cb_id, program, worker_core_set, output_tile_size, block_size * 2, output_format);

    // Packet header CB — needed by the legacy writer's TP>1 fabric forwarder
    // (it reserves 2 header slots from this CB). MUX path uses PacketHeaderPool
    // and doesn't touch this CB, but the CT arg slot must still be valid.
    // Size: 8 slots when fabric is in use, 1 slot otherwise.
    const uint32_t packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    const uint32_t packet_header_cb_tiles = (is_tp_1) ? 1u : 8u;
    tt::tt_metal::CircularBufferConfig packet_header_cb_config =
        tt::tt_metal::CircularBufferConfig(
            packet_header_cb_tiles * packet_header_size_bytes,
            {{reserved_packet_header_cb_id, tt::DataFormat::RawUInt32}})
            .set_page_size(reserved_packet_header_cb_id, packet_header_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, worker_core_set, packet_header_cb_config);

    // ------------------------------------------------------------------------
    // Reader kernel (on worker cores)
    // ------------------------------------------------------------------------
    const uint32_t H_full = W * args.ring_size;
    std::vector<uint32_t> reader_compile_args = {
        input_cb_id,
        weight_cb_id,
        reduce_scalar_sum_cb_id,
        reduce_scalar_avg_cb_id,
        epsilon_cb_id,
        transformation_mat_cb_id,
        rope_cos_cb_id,
        rope_sin_cb_id,
        num_tile_cols,
        block_size,
        /*reduce_factor=*/H_full,
        float_to_u32(args.epsilon),
        static_cast<uint32_t>(has_weight),
        static_cast<uint32_t>(fuse_rope),
        head_dim_tiles,
        chunk_size_rows,
        static_cast<uint32_t>(per_head_rope),
        rope_seqlen_tiles,
        bias_cb_id,
        static_cast<uint32_t>(has_bias),
    };
    TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    if (has_weight) {
        TensorAccessorArgs(weight.value().buffer()).append_to(reader_compile_args);
    } else {
        TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);  // dummy
    }
    if (has_bias) {
        TensorAccessorArgs(bias.value().buffer()).append_to(reader_compile_args);
    } else {
        TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);  // dummy
    }
    if (fuse_rope) {
        TensorAccessorArgs(trans_mat.value().buffer()).append_to(reader_compile_args);
        TensorAccessorArgs(rope_cos.value().buffer()).append_to(reader_compile_args);
        TensorAccessorArgs(rope_sin.value().buffer()).append_to(reader_compile_args);
    } else {
        TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
        TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
        TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/dataflow/"
        "wan_rmsnorm_fused_reader.cpp",
        worker_core_set,
        ReaderDataMovementConfig(reader_compile_args));

    // ------------------------------------------------------------------------
    // FabricMuxConfig (only when use_mux: TP>1 AND num_workers>1)
    // One MUX kernel per (direction, link); each has num_workers_per_link
    // channels (one per worker assigned to that link).
    // ------------------------------------------------------------------------
    const size_t mux_base_l1_address = mesh_device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    const size_t buffer_size_bytes_full_size_channel = tt::tt_fabric::get_tt_fabric_channel_buffer_size_bytes();
    const uint32_t num_workers_per_link = use_mux ? (num_workers / num_links_eff) : 0u;
    std::unique_ptr<tt::tt_fabric::FabricMuxConfig> mux_kernel_config;
    if (use_mux) {
        mux_kernel_config = std::make_unique<tt::tt_fabric::FabricMuxConfig>(
            /*num_full_size_channels=*/static_cast<uint8_t>(num_workers_per_link),
            /*num_header_only_channels=*/0,
            /*num_buffers_full_size_channel=*/1,
            /*num_buffers_header_only_channel=*/0,
            buffer_size_bytes_full_size_channel,
            mux_base_l1_address);
    }

    // ------------------------------------------------------------------------
    // Writer kernel (on worker cores). Three variants:
    //   is_tp_1            → legacy writer with is_tp_1=1 (no fabric).
    //   TP>1, num_workers=1 → legacy writer with is_tp_1=0 (direct fabric, single core).
    //   TP>1, num_workers>1 → MUX writer.
    // ------------------------------------------------------------------------
    KernelHandle writer_kernel_id;
    if (!use_mux) {
        // Legacy single-core writer (works for both TP=1 and TP>1 single-worker).
        std::vector<uint32_t> writer_compile_args = {
            output_cb_id,
            num_tile_cols,
            block_size,
            /*is_tp_1=*/static_cast<uint32_t>(is_tp_1 ? 1u : 0u),
            stats_local_cb_id,
            stats_gathered_cb_id,
            reserved_packet_header_cb_id,
            args.ring_size,
            device_index,
            num_targets_forward,
            num_targets_backward,
            head_dim_tiles,
            num_tile_rows,
        };
        TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);

        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/dataflow/"
            "wan_rmsnorm_fused_writer.cpp",
            worker_core_set,
            WriterDataMovementConfig(writer_compile_args));
    } else {
        // MUX writer for TP>1 (Phase 9 packed-page AG): first 13 CT args,
        // then 5 MUX CT args, then TensorAccessorArgs for output and stats
        // scratch. The two transposed/packed CBs replace the whole-tile
        // L1↔fabric path used in earlier phases.
        std::vector<uint32_t> writer_compile_args = {
            output_cb_id,
            num_tile_cols,
            block_size,
            // Compute pushes col-0 stat tiles here (reduce<SUM,REDUCE_ROW>
            // output: 32 sums in col 0, rest = 0 by LLK). The writer
            // extracts col 0 directly via strided L1 loads to pack the page.
            stats_local_cb_id,
            // The writer scatters the gathered packed pages into row 0 of
            // these tiles for compute to transpose back.
            stats_gathered_cb_id,
            // Packed-page staging + receive CBs.
            stats_packed_local_cb_id,
            stats_packed_gathered_cb_id,
            args.ring_size,
            device_index,
            num_targets_forward,
            num_targets_backward,
            chunk_size_rows,
            num_chunks_per_device,
            head_dim_tiles,
            num_tile_rows,
        };
        // Each link's MUX has num_workers_per_link clients; the writer kernel's
        // num_mux_clients CT arg uses this per-link count (termination master
        // waits for num_workers_per_link - 1 incs on its sem).
        ttnn::ccl::fabric_mux_connection_ct_args(
            num_workers_per_link,
            tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
            *mux_kernel_config,
            writer_compile_args);
        TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);
        // Persistent DRAM stats buffer accessor args (Phase 1).
        TensorAccessorArgs(stats_dram_buffer).append_to(writer_compile_args);

        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/dataflow/"
            "wan_rmsnorm_fused_writer_mux.cpp",
            worker_core_set,
            WriterDataMovementConfig(writer_compile_args));
    }

    // ------------------------------------------------------------------------
    // Compute kernel (on worker cores)
    // ------------------------------------------------------------------------
    std::vector<uint32_t> compute_compile_args = {
        input_cb_id,
        stats_local_cb_id,
        stats_gathered_cb_id,
        weight_cb_id,
        reduce_scalar_sum_cb_id,
        reduce_scalar_avg_cb_id,
        epsilon_cb_id,
        reduce_result_cb_id,
        intermediate_cb_id,
        pre_intermediate_cb_id,
        output_cb_id,
        transformation_mat_cb_id,
        rope_cos_cb_id,
        rope_sin_cb_id,
        rotated_input_cb_id,
        num_tile_cols,
        block_size,
        /*stats_tiles_cols=*/args.ring_size,
        chunk_size_rows,
        /*use_legacy_rsqrt=*/0u,
        static_cast<uint32_t>(has_weight),
        static_cast<uint32_t>(fuse_rope),
        head_dim_tiles,
        /*is_tp_1=*/(args.ring_size == 1) ? 1u : 0u,
        // Phase 9 packed AG CBs (used when is_tp_1 == 0 AND use_mux). For
        // is_tp_1 the compute kernel sidesteps the packed path entirely
        // (pushes col-0 stats straight into stats_gathered_cb).
        stats_transposed_local_cb_id,
        stats_transposed_gathered_cb_id,
        static_cast<uint32_t>(use_mux ? 1u : 0u),  // packed_ag_enabled
        static_cast<uint32_t>(per_head_rope),
        bias_cb_id,
        static_cast<uint32_t>(has_bias),
    };

    KernelHandle compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/compute/"
        "wan_rmsnorm_fused_compute.cpp",
        worker_core_set,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_args,
        });

    // ------------------------------------------------------------------------
    // MUX kernels (only TP>1; one per (direction, link))
    // ------------------------------------------------------------------------
    std::vector<KernelHandle> fwd_mux_kernel_ids(num_mux_per_direction, 0);
    std::vector<KernelHandle> bwd_mux_kernel_ids(num_mux_per_direction, 0);
    if (fwd_mux_valid || bwd_mux_valid) {
        const auto mux_ct_args = mux_kernel_config->get_fabric_mux_compile_time_args();
        for (uint32_t lnk = 0; lnk < num_mux_per_direction; lnk++) {
            if (fwd_mux_valid) {
                fwd_mux_kernel_ids[lnk] = tt::tt_metal::CreateKernel(
                    program,
                    "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    CoreRangeSet({CoreRange(fwd_mux_cores[lnk], fwd_mux_cores[lnk])}),
                    tt::tt_metal::DataMovementConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt::tt_metal::NOC::RISCV_0_default,
                        .compile_args = mux_ct_args,
                        .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
            }
            if (bwd_mux_valid) {
                bwd_mux_kernel_ids[lnk] = tt::tt_metal::CreateKernel(
                    program,
                    "tt_metal/fabric/impl/kernels/tt_fabric_mux.cpp",
                    CoreRangeSet({CoreRange(bwd_mux_cores[lnk], bwd_mux_cores[lnk])}),
                    tt::tt_metal::DataMovementConfig{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
                        .noc = tt::tt_metal::NOC::RISCV_0_default,
                        .compile_args = mux_ct_args,
                        .opt_level = tt::tt_metal::KernelBuildOptLevel::O3});
            }
        }
    }

    // ------------------------------------------------------------------------
    // Common runtime args
    // ------------------------------------------------------------------------
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();
    const uint32_t weight_addr = has_weight ? weight.value().buffer()->address() : 0;
    const uint32_t bias_addr = has_bias ? bias.value().buffer()->address() : 0;
    const uint32_t trans_mat_addr = fuse_rope ? trans_mat.value().buffer()->address() : 0;
    const uint32_t rope_cos_addr = fuse_rope ? rope_cos.value().buffer()->address() : 0;
    const uint32_t rope_sin_addr = fuse_rope ? rope_sin.value().buffer()->address() : 0;

    uint32_t out_ready_sem_bank_addr = 0;
    if (args.ring_size > 1) {
        TT_FATAL(
            !args.multi_device_global_semaphore.empty(),
            "TP>1 requires at least one GlobalSemaphore in multi_device_global_semaphore");
        out_ready_sem_bank_addr = args.multi_device_global_semaphore.at(0).address();
    }

    // ------------------------------------------------------------------------
    // MUX runtime args (TP>1 only). One termination master per (direction, link):
    // the first worker assigned to each link (worker i = link itself). Workers
    // are partitioned round-robin: worker i → link (i % num_links_eff).
    // ------------------------------------------------------------------------
    // Per-link termination master logical/virtual core.
    std::vector<CoreCoord> link_master_logical(use_mux ? num_links_eff : 0u);
    std::vector<CoreCoord> link_master_virtual(use_mux ? num_links_eff : 0u);
    if (use_mux) {
        for (uint32_t lnk = 0; lnk < num_links_eff; lnk++) {
            link_master_logical[lnk] = worker_cores[lnk];  // worker_id == lnk
            link_master_virtual[lnk] = mesh_device->worker_core_from_logical_core(link_master_logical[lnk]);
        }
    }

    std::vector<CoreCoord> fwd_mux_virtual(num_mux_per_direction, CoreCoord{0, 0});
    std::vector<CoreCoord> bwd_mux_virtual(num_mux_per_direction, CoreCoord{0, 0});
    for (uint32_t lnk = 0; lnk < num_mux_per_direction; lnk++) {
        if (fwd_mux_valid) {
            fwd_mux_virtual[lnk] = mesh_device->worker_core_from_logical_core(fwd_mux_cores[lnk]);
        }
        if (bwd_mux_valid) {
            bwd_mux_virtual[lnk] = mesh_device->worker_core_from_logical_core(bwd_mux_cores[lnk]);
        }
    }

    // Wire MUX kernel RT args: one set per (direction, link), each with the
    // correct link_idx into the fabric.
    if (fwd_mux_valid || bwd_mux_valid) {
        const auto src_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
        for (uint32_t lnk = 0; lnk < num_mux_per_direction; lnk++) {
            if (fwd_mux_valid) {
                auto mux_rt_args = mux_kernel_config->get_fabric_mux_run_time_args(
                    src_node_id, forward_fabric_node_id.value(), /*link_idx=*/lnk, program, {fwd_mux_cores[lnk]});
                tt::tt_metal::SetRuntimeArgs(program, fwd_mux_kernel_ids[lnk], {fwd_mux_cores[lnk]}, mux_rt_args);
            }
            if (bwd_mux_valid) {
                auto mux_rt_args = mux_kernel_config->get_fabric_mux_run_time_args(
                    src_node_id, backward_fabric_node_id.value(), /*link_idx=*/lnk, program, {bwd_mux_cores[lnk]});
                tt::tt_metal::SetRuntimeArgs(program, bwd_mux_kernel_ids[lnk], {bwd_mux_cores[lnk]}, mux_rt_args);
            }
        }
    }

    // ------------------------------------------------------------------------
    // Per-worker runtime args
    // ------------------------------------------------------------------------
    // Captured inside the worker loop (use_mux branch); stored in
    // shared_variables so override_runtime_arguments can refresh the stats
    // scratch address each launch.
    std::optional<size_t> stats_dram_addr_writer_arg_idx;
    // Worker assignment. For the packed-page MUX path, distribute chip-global
    // chunks across workers (each chunk = a contiguous block of chunk_size_rows
    // tile-rows). The DRAM page index for a chunk is `device * num_chunks_per_device
    // + chunk_idx`, so workers must align on chunk boundaries — they cannot
    // straddle a chunk. Even distribution: first (num_chunks % num_workers)
    // workers get (floor+1) chunks; the rest get floor.
    // For the legacy writer path (use_mux = false), keep the row-based split.
    const uint32_t base_chunks_per_worker = use_mux ? (num_chunks_per_device / num_workers) : 0u;
    const uint32_t extra_chunks = use_mux ? (num_chunks_per_device % num_workers) : 0u;
    for (uint32_t i = 0; i < num_workers; i++) {
        const auto& core = worker_cores[i];
        uint32_t tile_row_start;
        uint32_t tile_row_end;
        uint32_t worker_chunk_base = 0;
        if (use_mux) {
            // Even distribution by chunk.
            const uint32_t this_worker_chunks = base_chunks_per_worker + (i < extra_chunks ? 1u : 0u);
            worker_chunk_base = i * base_chunks_per_worker + std::min(i, extra_chunks);
            tile_row_start = std::min(worker_chunk_base * chunk_size_rows, num_tile_rows);
            tile_row_end = std::min(tile_row_start + this_worker_chunks * chunk_size_rows, num_tile_rows);
        } else {
            tile_row_start = std::min(i * num_tile_rows_per_worker, num_tile_rows);
            tile_row_end = std::min(tile_row_start + num_tile_rows_per_worker, num_tile_rows);
        }
        const uint32_t this_core_rows = tile_row_end - tile_row_start;

        std::vector<uint32_t> reader_rt_args = {
            input_addr,
            weight_addr,
            bias_addr,
            trans_mat_addr,
            rope_cos_addr,
            rope_sin_addr,
            tile_row_start,
            tile_row_end,
        };
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        std::vector<uint32_t> writer_rt_args = {
            output_addr,
            tile_row_start,
            tile_row_end,
        };
        if (use_mux) {
            // Round-robin link assignment: worker i uses link (i % num_links_eff).
            // channel_id within the assigned MUX is i / num_links_eff.
            const uint32_t link = i % num_links_eff;
            const uint32_t channel_id_in_link = i / num_links_eff;
            const bool is_term_master_of_link = (channel_id_in_link == 0);
            writer_rt_args.push_back(out_ready_sem_bank_addr);
            // Packed-page DRAM scratch base address + this worker's first
            // chunk index on the chip. DRAM page idx for a (device, chunk)
            // pair = device * num_chunks_per_device + chunk_idx. The buffer
            // is allocated fresh per launch (regular device tensor), so
            // override_runtime_arguments refreshes the address slot — record
            // its index in the per-program rt args vector.
            stats_dram_addr_writer_arg_idx = writer_rt_args.size();
            writer_rt_args.push_back(stats_dram_buffer->address());
            writer_rt_args.push_back(worker_chunk_base);
            ttnn::ccl::fabric_mux_connection_rt_args(
                /*mux_connection_valid=*/fwd_mux_valid,
                /*is_termination_master=*/is_term_master_of_link,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                fwd_mux_valid ? fwd_mux_virtual[link] : CoreCoord{0, 0},
                /*worker_id=*/channel_id_in_link,
                core,
                *mux_kernel_config,
                program,
                link_master_virtual[link],
                writer_rt_args);
            ttnn::ccl::fabric_mux_connection_rt_args(
                /*mux_connection_valid=*/bwd_mux_valid,
                /*is_termination_master=*/is_term_master_of_link,
                tt::tt_fabric::FabricMuxChannelType::FULL_SIZE_CHANNEL,
                bwd_mux_valid ? bwd_mux_virtual[link] : CoreCoord{0, 0},
                /*worker_id=*/channel_id_in_link,
                core,
                *mux_kernel_config,
                program,
                link_master_virtual[link],
                writer_rt_args);
        } else if (!is_tp_1) {
            // TP>1 single-worker path: append out_ready_sem + direct fabric
            // connection rt args (FabricConnectionManager layout).
            writer_rt_args.push_back(out_ready_sem_bank_addr);
            writer_rt_args.push_back(forward_fabric_node_id.has_value() ? 1u : 0u);
            if (forward_fabric_node_id.has_value()) {
                const auto local_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    local_node_id, forward_fabric_node_id.value(), /*link_idx=*/0, program, {core}, writer_rt_args);
            }
            writer_rt_args.push_back(backward_fabric_node_id.has_value() ? 1u : 0u);
            if (backward_fabric_node_id.has_value()) {
                const auto local_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
                tt::tt_fabric::append_fabric_connection_rt_args(
                    local_node_id, backward_fabric_node_id.value(), /*link_idx=*/0, program, {core}, writer_rt_args);
            }
        }
        SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);

        std::vector<uint32_t> compute_rt_args = {this_core_rows};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_rt_args);
    }

    return {
        std::move(program),
        WanFusedDistributedRmsnormSharedVariables{
            .reader_kernel_ids = {reader_kernel_id},
            .writer_kernel_ids = {writer_kernel_id},
            .compute_kernel_ids = {compute_kernel_id},
            .cores = worker_cores,
            .stats_dram_addr_writer_arg_idx = stats_dram_addr_writer_arg_idx,
        }};
}

WanFusedDistributedRmsnormMeshWorkloadFactory::cached_mesh_workload_t
WanFusedDistributedRmsnormMeshWorkloadFactory::create_mesh_workload(
    const WanFusedDistributedRmsnormParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const WanFusedDistributedRmsnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& range : tensor_coords.ranges()) {
        for (const auto& coord : range) {
            auto cached = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
            workload.add_program(ttnn::MeshCoordinateRange(coord), std::move(cached.program));
            shared_variables.emplace(ttnn::MeshCoordinateRange(coord), std::move(cached.shared_variables));
        }
    }

    return cached_mesh_workload_t{std::move(workload), std::move(shared_variables)};
}

void WanFusedDistributedRmsnormMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const WanFusedDistributedRmsnormParams& /*operation_attributes*/,
    const WanFusedDistributedRmsnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    const uint32_t input_addr = tensor_args.input.buffer()->address();
    const uint32_t output_addr = tensor_return_value.at(0).buffer()->address();
    const uint32_t weight_addr = tensor_args.weight.has_value() ? tensor_args.weight.value().buffer()->address() : 0;
    const uint32_t bias_addr = tensor_args.bias.has_value() ? tensor_args.bias.value().buffer()->address() : 0;
    const uint32_t trans_mat_addr =
        tensor_args.transformation_mat.has_value() ? tensor_args.transformation_mat.value().buffer()->address() : 0;
    const uint32_t rope_cos_addr =
        tensor_args.rope_cos.has_value() ? tensor_args.rope_cos.value().buffer()->address() : 0;
    const uint32_t rope_sin_addr =
        tensor_args.rope_sin.has_value() ? tensor_args.rope_sin.value().buffer()->address() : 0;
    // Stats DRAM scratch is reallocated per launch (it's a regular device
    // tensor), so its address changes and must be refreshed on cache hits.
    // The MUX writer reads it from a fixed runtime-arg slot whose host-side
    // index is captured in shared.stats_dram_addr_writer_arg_idx (set at
    // create_at time, only when use_mux).
    const uint32_t stats_dram_addr = tensor_return_value.size() > 1 ? tensor_return_value[1].buffer()->address() : 0u;

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& shared = cached_workload.shared_variables.at(range);
        const auto& reader_kernel_id = shared.reader_kernel_ids[0];
        const auto& writer_kernel_id = shared.writer_kernel_ids[0];

        auto& reader_runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
        auto& writer_runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);

        for (const auto& core : shared.cores) {
            auto& reader_args = reader_runtime_args_by_core.at(core.x).at(core.y);
            reader_args[0] = input_addr;
            reader_args[1] = weight_addr;
            reader_args[2] = bias_addr;
            reader_args[3] = trans_mat_addr;
            reader_args[4] = rope_cos_addr;
            reader_args[5] = rope_sin_addr;

            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
            if (shared.stats_dram_addr_writer_arg_idx.has_value()) {
                writer_args[shared.stats_dram_addr_writer_arg_idx.value()] = stats_dram_addr;
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
