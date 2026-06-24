// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "wan_fused_distributed_rmsnorm_program_factory.hpp"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
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
// num_tile_rows below this uses a single worker — the per-row forwarder overhead
// doesn't pay off with <4 tile-rows of compute per chip.
constexpr uint32_t kMuxRowsThreshold = 4u;

// Worker-count ceiling, derived from the device core grid (no hardcoded constant).
// The fabric forwarder removed the shared-MUX contention that previously capped this
// at 32 (a Wormhole-galaxy-specific number); a worker sweep then showed parallelism
// keeps helping with more workers — but only up to WHOLE GRID ROWS. Workers are placed
// row-major, so a count that is a multiple of grid.x tiles complete rows; pushing past
// that into a ragged final row (and leaving zero idle cores) costs 3–9% to NoC/dispatch
// contention (the 8x9 galaxy peaks at 64 = 8 full rows, regresses at the full-grid 68).
// So: budget = grid − one forwarder core per link, rounded DOWN to whole rows. The few
// idle cores that fall out (budget mod grid.x) are the slack that avoids the all-cores-
// busy contention — derived from geometry, not a magic margin. Adapts to any grid.
// `WAN_RMSNORM_WORKER_CAP` overrides for sweeps. Read inside the single-source-of-truth
// sizing path so the op + create_stats_buffer agree on num_workers / buffer geometry.
uint32_t derive_worker_cap(const CoreCoord& grid_size, uint32_t num_links) {
    const char* env = std::getenv("WAN_RMSNORM_WORKER_CAP");
    if (env != nullptr) {
        const long v = std::strtol(env, nullptr, 10);
        if (v > 0) {
            return static_cast<uint32_t>(v);
        }
    }
    const uint32_t max_cores = grid_size.x * grid_size.y;
    const uint32_t num_forwarders = std::max<uint32_t>(1u, num_links);  // one forwarder per link
    const uint32_t budget = max_cores > num_forwarders ? max_cores - num_forwarders : 1u;
    // Round down to whole grid rows (grid.x cores each); fall back to the raw budget if
    // even a single row doesn't fit.
    const uint32_t whole_rows = (grid_size.x > 0) ? (budget / grid_size.x) * grid_size.x : 0u;
    return whole_rows > 0u ? whole_rows : budget;
}
// Diagnostic override: WAN_RMSNORM_INPUT_CB_CHUNKS dials how many chunks deep
// the input_cb is buffered (default 2 = Phase-5 double-buffer). Deeper buffering
// lets the reader run further ahead of compute, keeping more DRAM input reads
// outstanding (the read path runs well below peak because the reader stalls on
// input_cb space once it is ~2 chunks ahead). Read once, clamped to [2, 8].
uint32_t input_cb_chunks() {
    static const uint32_t d = [] {
        const char* env = std::getenv("WAN_RMSNORM_INPUT_CB_CHUNKS");
        if (env != nullptr) {
            const long v = std::strtol(env, nullptr, 10);
            if (v >= 2 && v <= 8) {
                return static_cast<uint32_t>(v);
            }
        }
        return 2u;
    }();
    return d;
}
// Diagnostic override: WAN_RMSNORM_FORCE_WORKERS pins the exact worker count
// (still capped by num_tile_rows so we never over-provision). Lets a perf sweep
// push small shapes PAST the rows/2 heuristic to test whether more parallelism
// keeps shrinking the latency-bound wall. Read once.
uint32_t force_num_workers() {
    static const uint32_t v = [] {
        const char* env = std::getenv("WAN_RMSNORM_FORCE_WORKERS");
        if (env != nullptr) {
            const long n = std::strtol(env, nullptr, 10);
            if (n > 0) {
                return static_cast<uint32_t>(n);
            }
        }
        return 0u;
    }();
    return v;
}
// Diagnostic sweep knob: WAN_RMSNORM_FORCE_CHUNK=N forces chunk_size_rows=N
// (applied last in both compute_sizing and the program factory so the stats
// buffer stays consistent with the kernel). Timing-only. Note the per-head /
// streaming clamps still pin chunk=1 afterward for those paths.
uint32_t force_chunk_size() {
    static const uint32_t v = [] {
        const char* env = std::getenv("WAN_RMSNORM_FORCE_CHUNK");
        if (env != nullptr) {
            const long n = std::strtol(env, nullptr, 10);
            if (n > 0) {
                return static_cast<uint32_t>(n);
            }
        }
        return 0u;
    }();
    return v;
}
// Depth (in block_size groups) of the STREAMED per-head cos/sin CBs. Per-head
// RoPE has a distinct cos/sin per head, so the whole-row resident footprint is
// O(num_heads*head_dim) and overflows L1 at wide shards (TP=2 feat-2048). The
// reader already pushes block_size groups and the compute pops per block, so the
// CB only needs a couple of blocks of look-ahead. Smaller = less L1; larger =
// more reader prefetch. Default 2 (one block in flight + one filling). Broadcast
// RoPE is unaffected (its tiny head_dim_tiles buffer stays fully resident).
uint32_t rope_stream_blocks() {
    static const uint32_t v = [] {
        const char* env = std::getenv("WAN_RMSNORM_ROPE_STREAM_BLOCKS");
        if (env != nullptr) {
            const long n = std::strtol(env, nullptr, 10);
            if (n > 0) {
                return static_cast<uint32_t>(n);
            }
        }
        return 2u;
    }();
    return v;
}
// DIAGNOSTIC ABLATIONS (WAN_ABLATION env): inject a per-ablation -D into the
// reader/writer kernels so we can selectively skip NoC traffic and measure
// where kernel time goes. These BREAK correctness — perf attribution only.
//   1 = skip rope cos/sin reads   2 = skip input read   3 = skip output write
//   4 = skip fabric mcast+sem inc+sem wait   5 = skip writer gather/scatter
//   6 = skip weight/bias reads
std::map<std::string, std::string> ablation_defines() {
    std::map<std::string, std::string> d;
    const char* env = std::getenv("WAN_ABLATION");
    if (env != nullptr) {
        switch (std::strtol(env, nullptr, 10)) {
            case 1: d["WAN_ABL_SKIP_ROPE_READ"] = "1"; break;
            case 2: d["WAN_ABL_SKIP_INPUT_READ"] = "1"; break;
            case 3: d["WAN_ABL_SKIP_OUTPUT_WRITE"] = "1"; break;
            case 4: d["WAN_ABL_SKIP_FABRIC"] = "1"; break;
            case 5: d["WAN_ABL_SKIP_GATHER_SCATTER"] = "1"; break;
            case 6: d["WAN_ABL_SKIP_WEIGHT_READ"] = "1"; break;
            case 7: d["WAN_ABL_SKIP_COMPUTE"] = "1"; break;
            // 8: pure-compute — stub ALL DRAM reads/writes + fabric/AG while
            // keeping every CB reserve/push/wait/pop (they live outside the
            // per-skip #ifndef guards), so compute runs full-speed LLKs on
            // garbage. Inverse of case 7: isolates compute from all I/O.
            case 8:
                d["WAN_ABL_SKIP_INPUT_READ"] = "1";
                d["WAN_ABL_SKIP_WEIGHT_READ"] = "1";
                d["WAN_ABL_SKIP_ROPE_READ"] = "1";
                d["WAN_ABL_SKIP_OUTPUT_WRITE"] = "1";
                d["WAN_ABL_SKIP_FABRIC"] = "1";
                d["WAN_ABL_SKIP_GATHER_SCATTER"] = "1";
                break;
            default: break;
        }
    }
    return d;
}
// Streaming low-L1 fallback decision. The fast path keeps a whole tile-row of
// `input_cb` resident from PRE through POST (one DRAM read). On very wide
// shards (e.g. Wan TP=2: num_tile_cols=80) the resident input_cb plus the
// row-sized intermediate/rotated/output CBs overflow L1 (the static-CB cap is
// 1,572,864 B per core). When that happens we fall back to streaming `input_cb`
// in block_size chunks for BOTH the PRE sum-of-squares and a POST re-read pass
// — input_cb then costs O(block_size) instead of O(num_tile_cols), at the price
// of reading the row twice (≈ composite bandwidth). intermediate/rotated/output
// stay resident (they fit once input_cb shrinks). The PRE accumulation order is
// unchanged, so the streamed path is bit-exact with the resident path.
//
// WAN_RMSNORM_FORCE_STREAMING overrides the heuristic: "1" forces streaming on
// (for testing the path on small shapes), "0" forces it off. Read once.
//
// The byte estimate below mirrors the big-CB sizes allocated further down and
// is calibrated against the observed overflow (resident TP=2 allocates
// ~1,666,432 B). `kFixedOverheadBytes` covers the remaining small CBs (rope,
// stats, scalars, packed-AG, packet headers).
uint32_t streaming_force_mode() {
    static const uint32_t mode = [] {
        const char* env = std::getenv("WAN_RMSNORM_FORCE_STREAMING");
        if (env != nullptr) {
            if (std::strcmp(env, "1") == 0) {
                return 1u;  // force on
            }
            if (std::strcmp(env, "0") == 0) {
                return 2u;  // force off
            }
        }
        return 0u;  // auto
    }();
    return mode;
}
bool decide_streaming_low_l1(
    uint32_t num_tile_cols,
    uint32_t block_size,
    uint32_t chunk_size_rows,
    uint32_t input_tile_bytes,
    uint32_t intermediate_tile_bytes,
    uint32_t output_tile_bytes,
    bool has_weight,
    bool per_head_norm) {
    const uint32_t mode = streaming_force_mode();
    if (mode == 1u) {
        return true;
    }
    if (mode == 2u) {
        return false;
    }
    // per_head_norm uses head-block reduces over small head_dim shards; its L1
    // profile never overflows and the streamed compute path only handles the
    // whole-row reduce, so never auto-enable streaming for it.
    if (per_head_norm) {
        return false;
    }
    const uint32_t padded_row = ((num_tile_cols + block_size - 1u) / block_size) * block_size;
    const uint64_t input_bytes =
        static_cast<uint64_t>(input_cb_chunks()) * chunk_size_rows * num_tile_cols * input_tile_bytes;
    // intermediate_cb + rotated_input_cb (both row-sized, intermediate_tile_bytes).
    const uint64_t intermediate_bytes = 2ull * padded_row * intermediate_tile_bytes;
    // output_cb is 2 padded rows.
    const uint64_t output_bytes = 2ull * padded_row * output_tile_bytes;
    // Broadcast weight is num_tile_cols bf16 tiles (2048 B). Per-token is larger
    // but those shapes have small num_tile_cols and don't trigger streaming.
    const uint64_t weight_bytes = has_weight ? static_cast<uint64_t>(num_tile_cols) * 2048ull : 0ull;
    constexpr uint64_t kFixedOverheadBytes = 196608ull;      // ~192 KB of small CBs
    constexpr uint64_t kResidentL1BudgetBytes = 1572864ull;  // static-CB cap per core
    const uint64_t total = input_bytes + intermediate_bytes + output_bytes + weight_bytes + kFixedOverheadBytes;
    return total > kResidentL1BudgetBytes;
}
// The POST phase is sub-phase-major (mul-rms -> weight -> matmul -> rope, each
// across the whole row) — the fast layout, but it keeps intermediate_cb AND
// rotated_input_cb resident for a full row. At wide PER-HEAD shards (LTX TP=2
// feat-2048: 64 tile-cols) the resident rotated_input_cb (whole row, fp32)
// pushes total static CBs past L1. When that happens we fuse the matmul + RoPE
// finalize per block (block-major for those two sub-phases) so rotated_input_cb
// is block-local — fits L1 at the cost of per-block reconfigs. Estimate mirrors
// the big-CB allocations below; calibrated against the measured feat-2048
// overflow (1,593,632 B) vs the feat-1024 TP=4 case (fits). Trigger with margin
// below the real L1 cap so a borderline shard fuses rather than OOM-ing.
bool post_rotated_overflows_l1(
    uint32_t num_tile_cols,
    uint32_t block_size,
    uint32_t input_tile_bytes,
    uint32_t intermediate_tile_bytes,
    uint32_t output_tile_bytes,
    uint32_t rope_cb_tiles,
    uint32_t rope_tile_bytes,
    bool has_weight) {
    const uint32_t padded = ((num_tile_cols + block_size - 1u) / block_size) * block_size;
    uint64_t total = static_cast<uint64_t>(num_tile_cols) * input_tile_bytes;  // input_cb (chunk=1)
    total += 2ull * padded * intermediate_tile_bytes;                          // intermediate + rotated (whole row)
    total += 2ull * padded * output_tile_bytes;                                // output_cb (2 rows)
    total += has_weight ? static_cast<uint64_t>(num_tile_cols) * 2048ull : 0ull;  // weight_cb
    total += 2ull * rope_cb_tiles * rope_tile_bytes;                           // streamed cos + sin
    constexpr uint64_t kPostFixedOverheadBytes = 491520ull;                    // stats/packed-AG/pre-interm/scalars/trans/headers
    constexpr uint64_t kFuseTriggerBytes = 1400000ull;                         // margin below the ~1.43 MB L1 cap
    return total + kPostFixedOverheadBytes > kFuseTriggerBytes;
}
uint32_t pick_num_workers_tp_gt_1(uint32_t num_tile_rows, uint32_t cap) {
    if (num_tile_rows < kMuxRowsThreshold) {
        return 1u;
    }
    const uint32_t forced = force_num_workers();
    if (forced > 0u) {
        return std::min<uint32_t>(forced, num_tile_rows);
    }
    // One worker per tile-row, capped at the core budget (grid − forwarders). A worker
    // sweep on the forwarder showed parallelism keeps helping up to the grid limit, so
    // we provision as many workers as fit; `cap` already excludes the forwarder cores.
    return std::min<uint32_t>(num_tile_rows, cap);
}

// Sizing derivation used in both spec computation (to size the stats scratch
// tensor in `compute_output_specs`) and the program factory (to lay out
// kernels + CBs). Single source of truth so the two cannot drift.
WanFusedDistributedRmsnormSizing compute_sizing(
    const WanFusedDistributedRmsnormParams& args,
    const Tensor& input,
    const WanFusedDistributedRmsnormInputs& tensor_args) {
    (void)tensor_args;  // page geometry no longer depends on rope/streaming detection
    WanFusedDistributedRmsnormSizing s;
    const auto& padded = input.padded_shape();
    const uint32_t W = padded[-1];
    const uint32_t folded_H = input.physical_volume() / W;
    s.num_tile_rows = folded_H / TILE_HEIGHT;
    // per_head_norm reduces locally over head_dim per head — no AG needed even
    // when ring_size > 1. From the kernel's perspective, this is "is_tp_1" =
    // no fabric, no MUX, legacy writer path.
    s.is_tp_1 = (args.ring_size == 1) || args.per_head_norm;
    // Worker cap = device compute grid − forwarder cores. Derived from the input's
    // device so create_stats_buffer / validate / compute_output_specs / create_at all
    // agree on num_workers (they share this single-source-of-truth path).
    const uint32_t worker_cap = derive_worker_cap(input.device()->compute_with_storage_grid_size(), args.num_links);
    s.num_workers = s.is_tp_1 ? 1u : pick_num_workers_tp_gt_1(s.num_tile_rows, worker_cap);
    // `use_mux` now means "uses the fabric-forwarder all-gather (+ DRAM scratch)".
    // The MUX and legacy single-worker writers are gone — one fabric path.
    s.use_mux = !s.is_tp_1;
    // The forwarder round == one tile-row; sticks are coalesced across the
    // forwarder's worker group, so chunk/window are not row-batching knobs here.
    s.chunk_size_rows = 1u;
    s.window_size = 1u;
    if (s.use_mux) {
        // num_forwarders = min(num_links, num_workers): one coalescing forwarder
        // per independent routing plane. Each forwarder mcasts up to
        // sticks_per_packet 128 B sticks per row-round; the DRAM scratch page IS
        // one packet. Pages per device = num_forwarders * max_rounds, where a
        // round is one tile-row of the worker group.
        const uint32_t num_links_requested = std::max<uint32_t>(1u, args.num_links);
        const uint32_t num_forwarders = std::min<uint32_t>(num_links_requested, s.num_workers);
        const uint32_t max_rounds = tt::div_up(s.num_tile_rows, s.num_workers);
        // Pack as many 128 B fp32 sticks as fit one fabric packet. Reuse the
        // existing page formula by setting window_size = sticks_per_packet:
        // page_size_bytes = TILE_HEIGHT(=32) * window_size * 4 = sticks * 128.
        const uint32_t sticks_per_packet =
            std::max<uint32_t>(1u, tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / 128u);
        s.window_size = sticks_per_packet;
        s.num_chunks_per_device = num_forwarders * max_rounds;
        s.total_pages = args.ring_size * s.num_chunks_per_device;
        s.page_size_bytes = TILE_HEIGHT * s.window_size * sizeof(float);
    }
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
    // Per-token weight/bias: shape [N, H] (or [B,1,N,H]) instead of broadcast
    // [1, H]. Detected by checking the second-to-last logical dim — broadcast
    // weight has 1 there. When set, the reader reads per-row tiles using
    // noc_async_read_tile (full 4 KB/tile) and compute uses mul_tiles instead
    // of mul_tiles_bcast_rows.
    const bool per_token_weight = has_weight && (weight->logical_shape()[-2] > 1);
    const bool per_token_bias = has_bias && (bias->logical_shape()[-2] > 1);
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

    // per_head_norm reduces over head_dim locally per head — no AG, no fabric,
    // legacy writer path even with ring_size > 1.
    const bool is_tp_1 = (args.ring_size == 1) || args.per_head_norm;
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
        // TP>1 (forwarder AG): one worker per tile-row, capped at the core budget
        // (grid − forwarders). Same derivation as compute_sizing so the stats-buffer
        // geometry matches. Tiny shapes (<kMuxRowsThreshold) collapse to 1 worker.
        num_workers = pick_num_workers_tp_gt_1(num_tile_rows, derive_worker_cap(grid_size, args.num_links));
    }
    use_mux = !is_tp_1;  // "uses the fabric-forwarder all-gather"

    // Forwarder model: one coalescing forwarder core per independent routing
    // plane (num_forwarders = min(num_links, num_workers)). Each forwarder owns a
    // contiguous worker group and holds fwd+bwd fabric. No MUX cores, no legacy
    // single-worker path.
    const uint32_t num_links_requested = std::max<uint32_t>(1u, args.num_links);
    const uint32_t num_forwarders = use_mux ? std::min<uint32_t>(num_links_requested, num_workers) : 0u;
    const uint32_t total_cores_needed = num_workers + num_forwarders;
    TT_FATAL(
        total_cores_needed <= max_cores,
        "wan_fused_distributed_rmsnorm needs {} cores ({} workers + {} forwarders) but only {} available",
        total_cores_needed,
        num_workers,
        num_forwarders,
        max_cores);

    const uint32_t num_tile_rows_per_worker = tt::div_up(num_tile_rows, num_workers);
    const uint32_t workers_per_forwarder = use_mux ? tt::div_up(num_workers, num_forwarders) : num_workers;
    const uint32_t max_rounds = num_tile_rows_per_worker;  // forwarder round == tile-row

    // [worker_0..N-1, forwarder_0..F-1] on the device grid (row-major).
    const auto all_cores_vec = corerange_to_cores(core_grid, max_cores, /*row_major=*/true);
    std::vector<CoreCoord> worker_cores(all_cores_vec.begin(), all_cores_vec.begin() + num_workers);
    std::vector<CoreCoord> forwarder_cores;
    for (uint32_t f = 0; f < num_forwarders; f++) {
        forwarder_cores.push_back(all_cores_vec[num_workers + f]);
    }

    CoreRangeSet worker_core_set;
    for (const auto& c : worker_cores) {
        worker_core_set = worker_core_set.merge(CoreRangeSet({CoreRange(c, c)}));
    }
    CoreRangeSet forwarder_core_set;
    for (const auto& c : forwarder_cores) {
        forwarder_core_set = forwarder_core_set.merge(CoreRangeSet({CoreRange(c, c)}));
    }
    // Grid-wide set: the packet CB + sync semaphores are created here so their
    // L1 address is identical on every worker and forwarder core — a worker
    // learns its forwarder's packet base / sem addr from its OWN
    // get_write_ptr(packet_cb) / get_semaphore(id), no cross-core address args.
    const CoreRangeSet all_core_set({core_grid});

    // Partition helpers: worker w -> forwarder (w / wpf), slot (w % wpf);
    // contiguous tile-row split (worker w owns [w*rpw, min((w+1)*rpw, N))).
    auto worker_forwarder = [&](uint32_t w) -> uint32_t { return use_mux ? (w / workers_per_forwarder) : 0u; };
    auto worker_slot = [&](uint32_t w) -> uint32_t { return use_mux ? (w % workers_per_forwarder) : 0u; };
    auto worker_num_rows = [&](uint32_t w) -> uint32_t {
        const uint32_t s = std::min(w * num_tile_rows_per_worker, num_tile_rows);
        const uint32_t e = std::min(s + num_tile_rows_per_worker, num_tile_rows);
        return e - s;
    };

    // chunk_size_rows: aim for ≥2 chunks per worker so Phase 5's double-buffered
    // input_cb can overlap chunk N+1's reader fill + chunk N+1's AG with chunk
    // N's compute and chunk N's output drain. When the worker has ≥2 rows, pick
    // ceil(rows/2) as the chunk size (capped at kMaxChunkSizeRows); when only
    // 1 row, chunk=1 (no overlap possible at all).
    // Chunk cap = 1. A fuller sweep (rope + non-rope, real 38/152-tile-row sizes,
    // chunk 1-4 at W up to 64) showed chunk=1 is best or tied EVERYWHERE: fabric/AG
    // is only ~2us exposed so bigger chunks buy no amortization, and chunk>1 is ~10%
    // SLOWER on the large (152-row) shapes (the prefetch-overlap win never
    // materialized). So pin chunk=1; WAN_RMSNORM_FORCE_CHUNK still overrides for sweeps.
    constexpr uint32_t kMaxChunkSizeRows = 1u;
    // L1 budget cap (matches compute_sizing): chunk * num_tile_cols ≤ 128
    // keeps input_cb under ~512 KB per worker.
    const uint32_t chunk_h_cap = std::max(1u, 128u / std::max(1u, num_tile_cols));
    uint32_t chunk_size_rows =
        std::min<uint32_t>(std::min<uint32_t>(std::max(1u, num_tile_rows_per_worker), kMaxChunkSizeRows), chunk_h_cap);
    // num_chunks_per_device is computed below, AFTER the per-head-RoPE /
    // streaming chunk clamp, so the packed-page AG count matches the final chunk.

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

    // intermediate/rotated CBs are fp32 when fp32_dest_acc_en (full precision
    // across post sub-phases), bf16 otherwise. Needed both for the streaming
    // decision (below) and the CB allocation (further down).
    const tt::DataFormat intermediate_format = fp32_dest_acc_en ? fp32_format : bf16_format;
    const uint32_t intermediate_tile_size = tt::tile_size(intermediate_format);

    // Streaming low-L1 fallback: when the resident input_cb + row-sized
    // intermediate/rotated/output CBs would overflow L1, stream input_cb in
    // block_size chunks for PRE + a POST re-read pass instead. See
    // decide_streaming_low_l1() for the budget heuristic.
    const bool streaming_low_l1 = decide_streaming_low_l1(
        num_tile_cols,
        block_size,
        chunk_size_rows,
        input_tile_size,
        intermediate_tile_size,
        output_tile_size,
        has_weight,
        args.per_head_norm);
    // Clamp to a single resident row for (a) per-head RoPE — its cos/sin CBs are
    // chunk*num_tile_cols fp32 tiles (overflow L1 at feat>=1024) and the compute
    // deadlocks at chunk>=2 with many rows; and (b) the streaming-low-L1 path,
    // which only supports one resident row. compute_sizing applies the SAME clamp
    // so the caller's stats buffer (window/pages) matches. feat-2048 per-head RoPE
    // still can't fit even one row -> clean compile-time CB-alloc OOM (needs
    // cos/sin streaming, a separate change), NOT a hang.
    // Per-head RoPE forces chunk_size_rows=1 (the chunk>=2 per-head path deadlocked
    // under the watcher; chunk=1 is the safe + optimal layout anyway). Streaming
    // low-L1 also requires chunk=1.
    if (per_head_rope || streaming_low_l1) {
        chunk_size_rows = 1u;
    }
    if (use_mux && force_chunk_size() > 0u) {
        chunk_size_rows = force_chunk_size();
    }
    // Forwarder AG: DRAM pages per device = num_forwarders * max_rounds (one page
    // per forwarder per row-round). Page idx = my_device*num_chunks_per_device +
    // forwarder*max_rounds + round.
    const uint32_t num_chunks_per_device = use_mux ? (num_forwarders * max_rounds) : 0u;
    // The streamed compute path handles only the whole-row reduce with one row
    // resident at a time (chunk_size_rows==1, enforced by the clamp above).
    TT_FATAL(
        !streaming_low_l1 || chunk_size_rows == 1,
        "wan_fused_distributed_rmsnorm streaming low-L1 path requires chunk_size_rows==1 (got {})",
        chunk_size_rows);
    TT_FATAL(
        !streaming_low_l1 || (num_tile_cols % block_size == 0),
        "wan_fused_distributed_rmsnorm streaming low-L1 path requires num_tile_cols ({}) divisible by block_size "
        "({})",
        num_tile_cols,
        block_size);

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
    constexpr uint32_t packet_cb_id = tt::CBIndex::c_17;  // forwarder coalesced packet (grid-wide, depth 2)
    constexpr uint32_t stats_transposed_gathered_cb_id = tt::CBIndex::c_19;
    constexpr uint32_t bias_cb_id = tt::CBIndex::c_20;

    // Double-buffer input_cb (Phase 5): reader can fill chunk N+1 while compute
    // is in chunk N's post phase. Cumulative cb_wait_front in compute (Phase 4)
    // pairs naturally with this — compute uses absolute indices within the
    // current chunk, and cb_pop_front at end of chunk frees that chunk's slots
    // back to the reader.
    //
    // Streaming low-L1: input_cb holds only a handful of block_size-tile blocks
    // (constant in num_tile_cols). The reader streams the row twice — once for
    // PRE, once for the POST re-read — and compute pops each block as it goes,
    // so the CB just needs enough depth for the reader to run a few blocks
    // ahead of compute within each pass.
    const uint32_t chunk_input_tiles = chunk_size_rows * num_tile_cols;
    constexpr uint32_t kStreamingInputBlocks = 4u;
    const uint32_t input_cb_tiles =
        streaming_low_l1 ? (kStreamingInputBlocks * block_size) : (input_cb_chunks() * chunk_input_tiles);
    create_cb(input_cb_id, program, worker_core_set, input_tile_size, input_cb_tiles, input_format);

    // per_head_norm produces num_heads_per_device stat tiles per row instead
    // of one. is_tp_1 already captures the "no AG needed" path for both
    // ring_size==1 and per_head_norm, so the compute kernel pushes directly
    // into stats_gathered_cb (skipping stats_local). When per_head_norm is on
    // but is_tp_1 path is used we still need stats_gathered_cb sized for the
    // per-head fan-out.
    const uint32_t per_row_stats_count = args.per_head_norm ? args.num_heads_per_device : args.ring_size;
    const uint32_t stats_local_tiles = (args.ring_size > 1 && !args.per_head_norm) ? chunk_size_rows : 1;
    create_cb(stats_local_cb_id, program, worker_core_set, fp32_tile_size, stats_local_tiles, fp32_format);
    // The MUX writer scatters one chunk's gathered stats at a time; the legacy
    // single-worker writer (!is_tp_1 && !use_mux) instead all-gathers the worker's
    // WHOLE row range, addressing slots as base + (r*ring_size + device). Size
    // stats_gathered for num_tile_rows rows on that path — otherwise its
    // cb_reserve_back(num_tile_rows*ring_size) blocks forever on a chunk-sized CB
    // (the TP=2 small-shape hang). is_tp_1 / MUX push per chunk, so chunk-sized is fine.
    const uint32_t stats_gathered_rows = (!is_tp_1 && !use_mux) ? num_tile_rows : chunk_size_rows;
    const uint32_t stats_gathered_tiles = stats_gathered_rows * per_row_stats_count;
    create_cb(stats_gathered_cb_id, program, worker_core_set, fp32_tile_size, stats_gathered_tiles, fp32_format);

    // Transposed stat CBs on the worker cores. For is_tp_1 these are unused stubs
    // (that path keeps stats in col 0 and reduces locally). chunk==1 so local is 1.
    create_cb(stats_transposed_local_cb_id, program, worker_core_set, fp32_tile_size, 1u, fp32_format);
    create_cb(
        stats_transposed_gathered_cb_id,
        program,
        worker_core_set,
        fp32_tile_size,
        use_mux ? args.ring_size : 1u,
        fp32_format);
    uint32_t unit_packet_bytes = 0u;
    if (use_mux) {
        // Coalesced fabric packet, allocated on the WHOLE grid so its L1 address
        // is identical on every worker + forwarder core (a worker writes its 128 B
        // stick into its forwarder's copy at this same address; the forwarder reads
        // its own). page == one fabric packet (sticks_per_packet * 128 B), depth 2.
        const uint32_t sticks_per_packet =
            std::max<uint32_t>(1u, tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / 128u);
        unit_packet_bytes = sticks_per_packet * 128u;
        TT_FATAL(
            sticks_per_packet >= workers_per_forwarder,
            "wan_fused_distributed_rmsnorm: fabric packet holds {} sticks but a forwarder group has {} workers",
            sticks_per_packet,
            workers_per_forwarder);
        tt::tt_metal::CircularBufferConfig packet_cfg =
            tt::tt_metal::CircularBufferConfig(2u * unit_packet_bytes, {{packet_cb_id, fp32_format}})
                .set_page_size(packet_cb_id, unit_packet_bytes);
        tt::tt_metal::CreateCircularBuffer(program, all_core_set, packet_cfg);
    }

    // Per-token weight/bias is per-row: weight_cb holds chunk_size_rows
    // worth of weight tiles (one row's slice at a time), popped per row by
    // compute. Broadcast weight/bias holds a single row's worth, retained
    // across the whole worker. has_bias is implied by has_weight.
    const uint32_t weight_cb_tiles =
        has_weight ? (per_token_weight ? chunk_size_rows * num_tile_cols : num_tile_cols) : 1;
    create_cb(weight_cb_id, program, worker_core_set, bf16_tile_size, weight_cb_tiles, bf16_format);
    const uint32_t bias_cb_tiles = has_bias ? (per_token_bias ? chunk_size_rows * num_tile_cols : num_tile_cols) : 1;
    create_cb(bias_cb_id, program, worker_core_set, bf16_tile_size, bias_cb_tiles, bf16_format);

    create_cb(reduce_scalar_sum_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    create_cb(reduce_scalar_avg_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    create_cb(epsilon_cb_id, program, worker_core_set, bf16_tile_size, 1, bf16_format);
    create_cb(reduce_result_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    create_cb(pre_intermediate_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    create_cb(transformation_mat_cb_id, program, worker_core_set, bf16_tile_size, 1, bf16_format);

    // Block-major POST: fuse the matmul + RoPE finalize per block so
    // rotated_input_cb is block-local instead of whole-row (set below once the
    // streamed rope CB size is known). Off unless the resident POST overflows L1.
    bool fuse_mm_rope = false;
    if (fuse_rope) {
        // Size rope CBs to hold a whole chunk's worth — reader pushes
        // rope_tiles_per_row per row eagerly, compute pops them only at end
        // of each row in post phase. Without chunk-sized buffering the reader
        // blocks at row 1 and chunk_size_rows>1 input never fully arrives.
        // For per-head RoPE, rope_tiles_per_row = num_tile_cols
        // (num_heads_per_device * head_dim_tiles); for the broadcast default
        // it's just head_dim_tiles.
        //
        // Size the cos/sin CBs to match the actual cos/sin tensor dtype rather
        // than assuming fp32. LTX feeds bf16 cos/sin (its standalone RoPE op is
        // all-bf16); accepting bf16 here halves the rope-table DRAM reads and L1
        // footprint. The reader derives tile bytes from get_tile_size(rope_cos_cb)
        // and the compute reconfigs the unpacker via reconfig_data_format(), so
        // both follow this format automatically.
        const tt::DataFormat rope_format = datatype_to_dataformat_converter(rope_cos->dtype());
        const uint32_t rope_tile_size = tt::tile_size(rope_format);
        const uint32_t rope_resident_tiles = chunk_size_rows * rope_tiles_per_row;
        // Decide block-major POST FIRST, using the RESIDENT (whole-row) cos/sin + rotated
        // footprint — so we only leave the fast resident layout when it would actually
        // overflow L1 (TP=2 feat-2048). All TP=4 per-head shards fit, so they stay resident.
        fuse_mm_rope = per_head_rope && post_rotated_overflows_l1(
                                            num_tile_cols,
                                            block_size,
                                            input_tile_size,
                                            intermediate_tile_size,
                                            output_tile_size,
                                            rope_resident_tiles,
                                            rope_tile_size,
                                            has_weight);
        // Stream per-head cos/sin (cap the CB at a few block_size groups, compute pops
        // per block) ONLY in the block-major path — that's where we need the L1 back.
        // The resident path keeps the WHOLE-ROW cos/sin so it isn't slowed (a shrunk
        // cos/sin CB regressed TP=4 video self-attn ~10-27% by starving prefetch).
        const uint32_t rope_cb_tiles = (per_head_rope && fuse_mm_rope)
                                           ? std::min(rope_resident_tiles, rope_stream_blocks() * block_size)
                                           : rope_resident_tiles;
        create_cb(rope_cos_cb_id, program, worker_core_set, rope_tile_size, rope_cb_tiles, rope_format);
        create_cb(rope_sin_cb_id, program, worker_core_set, rope_tile_size, rope_cb_tiles, rope_format);
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
    //
    // Format mirrors composite's rmsnorm_post_allgather: FP32 when
    // fp32_dest_acc_en is on (preserves full precision across sub-phases),
    // bf16 otherwise. With bf16 here, every sub-phase output gets rounded
    // to 8 mantissa bits — composite-vs-fused diverges noticeably when the
    // input distribution has high variance (e.g. matmul-output values).
    // (intermediate_format / intermediate_tile_size are computed earlier for
    // the streaming-low-L1 decision.)
    const uint32_t intermediate_cb_tiles = tt::div_up(num_tile_cols, block_size) * block_size;
    create_cb(
        intermediate_cb_id,
        program,
        worker_core_set,
        intermediate_tile_size,
        intermediate_cb_tiles,
        intermediate_format);
    // Block-major POST fuses matmul+rope per block, so rotated_input_cb only needs
    // to hold a block (double-buffered for matmul->rope pipelining) instead of a
    // whole row — this is what reclaims the L1 for wide per-head shards. The
    // resident sub-phase-major path (everything else) keeps the whole-row buffer.
    const uint32_t rotated_cb_tiles = fuse_mm_rope ? (2u * block_size) : intermediate_cb_tiles;
    create_cb(
        rotated_input_cb_id,
        program,
        worker_core_set,
        intermediate_tile_size,
        rotated_cb_tiles,
        intermediate_format);
    // output_cb sized to 2 full padded rows so the writer can deep-drain a
    // whole row under ONE noc_async_writes_flushed() (instead of flushing every
    // block_size=2 tiles) while compute produces the next row. The shallow
    // 4-tile CB capped write concurrency at block_size, leaving the output
    // drain at ~22% of POST as back-pressure (P_ADD's 5.9× core-to-core spread
    // = shared DRAM-write contention). A row-deep CB lets writes pipeline at
    // DRAM depth, mirroring the deep-read reader change. The MUX writer drains
    // per-block (overlap preserved) but flushes+pops once per row.
    const uint32_t output_cb_tiles = 2u * intermediate_cb_tiles;
    create_cb(output_cb_id, program, worker_core_set, output_tile_size, output_cb_tiles, output_format);

    // Packet header CB. The forwarder reserves 2 header slots (fwd+bwd) from it,
    // so on the AG path it lives on the FORWARDER cores. is_tp_1's drain-only
    // writer never touches it (1-slot stub on the worker cores).
    const uint32_t packet_header_size_bytes = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    {
        const uint32_t header_tiles = use_mux ? 4u : 1u;
        const CoreRangeSet& header_cores = use_mux ? forwarder_core_set : worker_core_set;
        tt::tt_metal::CircularBufferConfig packet_header_cb_config =
            tt::tt_metal::CircularBufferConfig(
                header_tiles * packet_header_size_bytes, {{reserved_packet_header_cb_id, tt::DataFormat::RawUInt32}})
                .set_page_size(reserved_packet_header_cb_id, packet_header_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, header_cores, packet_header_cb_config);
    }

    // Sync semaphores for the worker<->forwarder handshake. arrival (workers inc,
    // forwarder waits) + go (forwarder incs, workers wait) are on-chip; created on
    // the WHOLE grid so their L1 address is identical across workers + forwarders
    // (kernels resolve via get_semaphore(id)). out_ready is the caller's
    // GlobalSemaphore, fabric-inc'd by peer forwarders (resolved later).
    const uint32_t arrival_sem_id = use_mux ? tt::tt_metal::CreateSemaphore(program, all_core_set, 0u) : 0u;
    const uint32_t go_sem_id = use_mux ? tt::tt_metal::CreateSemaphore(program, all_core_set, 0u) : 0u;

    // ------------------------------------------------------------------------
    // Reader kernel (on worker cores)
    // ------------------------------------------------------------------------
    const uint32_t H_full = W * args.ring_size;
    // Per-head norm reduces over head_dim only, so the AVG scalar divides by
    // head_dim instead of H_full.
    const uint32_t reduce_factor = args.per_head_norm ? (W / args.num_heads_per_device) : H_full;
    std::vector<uint32_t> reader_compile_args = {
        input_cb_id,
        weight_cb_id,
        rope_cos_cb_id,
        rope_sin_cb_id,
        num_tile_cols,
        block_size,
        static_cast<uint32_t>(has_weight),
        static_cast<uint32_t>(fuse_rope),
        head_dim_tiles,
        chunk_size_rows,
        static_cast<uint32_t>(per_head_rope),
        rope_seqlen_tiles,
        bias_cb_id,
        static_cast<uint32_t>(has_bias),
        static_cast<uint32_t>(per_token_weight),
        static_cast<uint32_t>(per_token_bias),
        static_cast<uint32_t>(streaming_low_l1),
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
        TensorAccessorArgs(rope_cos.value().buffer()).append_to(reader_compile_args);
        TensorAccessorArgs(rope_sin.value().buffer()).append_to(reader_compile_args);
    } else {
        TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
        TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/dataflow/"
        "wan_rmsnorm_fused_reader.cpp",
        worker_core_set,
        ReaderDataMovementConfig(reader_compile_args, ablation_defines()));

    // ------------------------------------------------------------------------
    // Writer kernel (on worker cores). Two variants:
    //   is_tp_1 (ring==1 / per_head_norm)  → drain-only writer (no fabric; stats
    //                                         stay local in compute).
    //   AG path (ring>1, !per_head_norm)   → forwarder-model worker writer; the
    //                                         per-link forwarder cores hold the fabric.
    // ------------------------------------------------------------------------
    KernelHandle writer_kernel_id;
    if (!use_mux) {
        // Drain-only writer for the is_tp_1 (no-AG) path.
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
        // Scalar/eps/trans_mat population args (the writer always populates these
        // CBs so the reader starts the input read ASAP). Appended after the output
        // accessor; the kernel reads them at output_args.next_compile_time_args_offset().
        writer_compile_args.push_back(reduce_scalar_sum_cb_id);
        writer_compile_args.push_back(reduce_scalar_avg_cb_id);
        writer_compile_args.push_back(epsilon_cb_id);
        writer_compile_args.push_back(transformation_mat_cb_id);
        writer_compile_args.push_back(reduce_factor);
        writer_compile_args.push_back(float_to_u32(args.epsilon));
        writer_compile_args.push_back(static_cast<uint32_t>(fuse_rope));
        if (fuse_rope) {
            TensorAccessorArgs(trans_mat.value().buffer()).append_to(writer_compile_args);
        } else {
            TensorAccessorArgs(input_tensor.buffer()).append_to(writer_compile_args);  // dummy
        }

        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/dataflow/"
            "wan_rmsnorm_fused_writer.cpp",
            worker_core_set,
            WriterDataMovementConfig(writer_compile_args, ablation_defines()));
    } else {
        // Forwarder-model worker writer (AG path): no fabric. Per row it pushes
        // its 128 B stick into the forwarder's grid-uniform packet CB, waits the
        // go-sem, reads the coalesced ring gather from DRAM into row-0 of the
        // transposed-gathered tiles, and drains output. my_forwarder_index /
        // my_slot are per-core runtime args (set in the rt loop).
        std::vector<uint32_t> writer_compile_args = {
            output_cb_id,
            num_tile_cols,
            block_size,
            stats_transposed_local_cb_id,
            stats_transposed_gathered_cb_id,
            args.ring_size,
            head_dim_tiles,
            num_tile_rows,
            max_rounds,
            128u,  // stick_bytes (32 fp32)
            num_chunks_per_device,
            packet_cb_id,
            arrival_sem_id,
            go_sem_id,
        };
        TensorAccessorArgs(output_tensor.buffer()).append_to(writer_compile_args);
        TensorAccessorArgs(stats_dram_buffer).append_to(writer_compile_args);
        writer_compile_args.push_back(reduce_scalar_sum_cb_id);
        writer_compile_args.push_back(reduce_scalar_avg_cb_id);
        writer_compile_args.push_back(epsilon_cb_id);
        writer_compile_args.push_back(transformation_mat_cb_id);
        writer_compile_args.push_back(reduce_factor);
        writer_compile_args.push_back(float_to_u32(args.epsilon));
        writer_compile_args.push_back(static_cast<uint32_t>(fuse_rope));
        if (fuse_rope) {
            TensorAccessorArgs(trans_mat.value().buffer()).append_to(writer_compile_args);
        } else {
            TensorAccessorArgs(input_tensor.buffer()).append_to(writer_compile_args);  // dummy
        }

        writer_kernel_id = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/dataflow/"
            "wan_rmsnorm_fused_worker_writer.cpp",
            worker_core_set,
            WriterDataMovementConfig(writer_compile_args, ablation_defines()));
    }

    // ------------------------------------------------------------------------
    // Forwarder kernels (AG path): one per forwarder core (per-core CT args).
    // ------------------------------------------------------------------------
    std::vector<KernelHandle> forwarder_kernel_ids(num_forwarders, 0);
    for (uint32_t f = 0; f < num_forwarders; f++) {
        const uint32_t group_begin = f * workers_per_forwarder;
        const uint32_t group_end = std::min(group_begin + workers_per_forwarder, num_workers);
        const uint32_t group_size = group_end - group_begin;
        std::vector<uint32_t> fwd_ct = {
            packet_cb_id,
            reserved_packet_header_cb_id,
            args.ring_size,
            device_index,
            num_targets_forward,
            num_targets_backward,
            f,  // forwarder_index
            num_forwarders,
            group_size,
            max_rounds,
            128u,  // stick_bytes
            num_chunks_per_device,
            arrival_sem_id,
            go_sem_id,
        };
        TensorAccessorArgs(stats_dram_buffer).append_to(fwd_ct);
        forwarder_kernel_ids[f] = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/wan_fused_distributed_rmsnorm/device/kernels/dataflow/"
            "wan_rmsnorm_fused_forwarder.cpp",
            CoreRangeSet({CoreRange(forwarder_cores[f], forwarder_cores[f])}),
            WriterDataMovementConfig(fwd_ct, ablation_defines()));
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
        // is_tp_1 must match the factory-level is_tp_1 (== ring_size==1 ||
        // per_head_norm). per_head_norm reduces locally per head (no AG), so the
        // compute must take the is_tp_1 path too: push the num_heads per-row stat
        // tiles straight into stats_gathered_cb (sized for num_heads) and consume
        // them locally in POST. With the old (ring_size==1)-only form, per_head
        // ring>1 routed PRE to stats_local_cb (sized 1 tile, no consumer) and
        // wedged on the 2nd head's cb_reserve_back.
        /*is_tp_1=*/static_cast<uint32_t>(is_tp_1 ? 1u : 0u),
        // Phase 9 packed AG CBs (used when is_tp_1 == 0 AND use_mux). For
        // is_tp_1 the compute kernel sidesteps the packed path entirely
        // (pushes col-0 stats straight into stats_gathered_cb).
        stats_transposed_local_cb_id,
        stats_transposed_gathered_cb_id,
        static_cast<uint32_t>(use_mux ? 1u : 0u),  // packed_ag_enabled
        static_cast<uint32_t>(per_head_rope),
        bias_cb_id,
        static_cast<uint32_t>(has_bias),
        static_cast<uint32_t>(args.per_head_norm ? 1u : 0u),
        args.num_heads_per_device,
        static_cast<uint32_t>(per_token_weight),
        static_cast<uint32_t>(per_token_bias),
        float_to_u32(args.epsilon),  // eps_bits: fp32 scalar for fused +eps in reduce post-op
        static_cast<uint32_t>(streaming_low_l1),
        static_cast<uint32_t>(fuse_mm_rope),  // block-major POST: fuse matmul+rope per block (rotated block-local)
    };

    // fp32 dest accumulation is REQUIRED, unconditionally — not just for fp32
    // input. It is what keeps every internal CB (stats, reduce, intermediate,
    // rotated) at fp32 (intermediate_format above) and the reduce/eps/rsqrt
    // accumulating in fp32 DST. Without it the intermediates silently drop to
    // bf16 (8 mantissa bits) and the sum(x**2) / normalize lose precision (worse
    // for bf16 input than fp32 input, since the unpacker also downcasts through
    // SrcA to TF32). The op's invariant is "inputs/outputs may be bf16 or fp32,
    // internals are always fp32", so we enforce it here regardless of input dtype.
    // (Note: the FPU eltwise path — mul_tiles/add_tiles — still truncates its
    // operands to TF32 ~10 mantissa bits; that floor is inherent to SrcA/SrcB and
    // is NOT lifted by fp32_dest_acc_en or UnpackToDestFp32.)
    TT_FATAL(
        fp32_dest_acc_en,
        "wan_fused_distributed_rmsnorm requires fp32_dest_acc_en=true in the compute kernel config "
        "(internals are always fp32); got fp32_dest_acc_en=false.");

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
            .defines = ablation_defines(),
        });

    // ------------------------------------------------------------------------
    // MUX kernels (only TP>1; one per (direction, link))
    // ------------------------------------------------------------------------
    // ------------------------------------------------------------------------
    // Common runtime args
    // ------------------------------------------------------------------------
    const uint32_t input_addr = input_tensor.buffer()->address();
    const uint32_t output_addr = output_tensor.buffer()->address();
    const uint32_t weight_addr = has_weight ? weight.value().buffer()->address() : 0;
    const uint32_t bias_addr = has_bias ? bias.value().buffer()->address() : 0;
    const uint32_t rope_cos_addr = fuse_rope ? rope_cos.value().buffer()->address() : 0;
    const uint32_t rope_sin_addr = fuse_rope ? rope_sin.value().buffer()->address() : 0;
    const uint32_t trans_mat_addr_rt = fuse_rope ? trans_mat.value().buffer()->address() : 0u;
    const uint32_t stats_dram_addr = use_mux ? stats_dram_buffer->address() : 0u;

    uint32_t out_ready_sem_bank_addr = 0;
    if (args.ring_size > 1) {
        TT_FATAL(
            !args.multi_device_global_semaphore.empty(),
            "TP>1 requires at least one GlobalSemaphore in multi_device_global_semaphore");
        out_ready_sem_bank_addr = args.multi_device_global_semaphore.at(0).address();
    }

    // Virtual (NoC) coords: workers write sticks / inc arrival on their forwarder;
    // forwarders inc the go-sem on their workers.
    std::vector<CoreCoord> forwarder_virtual(num_forwarders);
    for (uint32_t f = 0; f < num_forwarders; f++) {
        forwarder_virtual[f] = mesh_device->worker_core_from_logical_core(forwarder_cores[f]);
    }
    std::vector<CoreCoord> worker_virtual(num_workers);
    for (uint32_t i = 0; i < num_workers; i++) {
        worker_virtual[i] = mesh_device->worker_core_from_logical_core(worker_cores[i]);
    }

    // ------------------------------------------------------------------------
    // Per-worker runtime args (contiguous tile-row split).
    // ------------------------------------------------------------------------
    std::optional<size_t> stats_dram_addr_writer_arg_idx;  // worker-writer stats_dram slot (override refresh)
    for (uint32_t i = 0; i < num_workers; i++) {
        const auto& core = worker_cores[i];
        const uint32_t tile_row_start = std::min(i * num_tile_rows_per_worker, num_tile_rows);
        const uint32_t tile_row_end = std::min(tile_row_start + num_tile_rows_per_worker, num_tile_rows);
        const uint32_t this_core_rows = tile_row_end - tile_row_start;

        std::vector<uint32_t> reader_rt_args = {
            input_addr, weight_addr, bias_addr, rope_cos_addr, rope_sin_addr, tile_row_start, tile_row_end};
        SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);

        std::vector<uint32_t> writer_rt_args;
        if (!use_mux) {
            // is_tp_1 drain-only writer: output_addr, start, end, trans_mat (rt[3]).
            writer_rt_args = {output_addr, tile_row_start, tile_row_end, trans_mat_addr_rt};
        } else {
            // worker-writer: output, start, end, trans_mat, stats_dram (rt[4]),
            // forwarder NoC x/y, my_forwarder_index, my_slot.
            const uint32_t f = worker_forwarder(i);
            writer_rt_args = {output_addr, tile_row_start, tile_row_end, trans_mat_addr_rt};
            stats_dram_addr_writer_arg_idx = writer_rt_args.size();
            writer_rt_args.push_back(stats_dram_addr);
            writer_rt_args.push_back(static_cast<uint32_t>(forwarder_virtual[f].x));
            writer_rt_args.push_back(static_cast<uint32_t>(forwarder_virtual[f].y));
            writer_rt_args.push_back(f);
            writer_rt_args.push_back(worker_slot(i));
        }
        SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);

        std::vector<uint32_t> compute_rt_args = {this_core_rows};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_rt_args);
    }

    // ------------------------------------------------------------------------
    // Per-forwarder runtime args: stats_dram, out_ready GlobalSemaphore, the
    // group's worker NoC coords, present_count[r], then fwd+bwd fabric-connection
    // args on this forwarder's routing plane (link_idx = f).
    // ------------------------------------------------------------------------
    const auto local_node_id = mesh_device->get_fabric_node_id(mesh_coordinate);
    for (uint32_t f = 0; f < num_forwarders; f++) {
        const auto& core = forwarder_cores[f];
        const uint32_t group_begin = f * workers_per_forwarder;
        const uint32_t group_end = std::min(group_begin + workers_per_forwarder, num_workers);
        std::vector<uint32_t> fwd_rt = {stats_dram_addr, out_ready_sem_bank_addr};
        for (uint32_t w = group_begin; w < group_end; w++) {
            fwd_rt.push_back(static_cast<uint32_t>(worker_virtual[w].x));
            fwd_rt.push_back(static_cast<uint32_t>(worker_virtual[w].y));
        }
        for (uint32_t r = 0; r < max_rounds; r++) {
            uint32_t pc = 0;
            for (uint32_t w = group_begin; w < group_end; w++) {
                if (worker_num_rows(w) > r) {
                    pc++;
                }
            }
            fwd_rt.push_back(pc);
        }
        fwd_rt.push_back(forward_fabric_node_id.has_value() ? 1u : 0u);
        if (forward_fabric_node_id.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                local_node_id, forward_fabric_node_id.value(), /*link_idx=*/f, program, {core}, fwd_rt);
        }
        fwd_rt.push_back(backward_fabric_node_id.has_value() ? 1u : 0u);
        if (backward_fabric_node_id.has_value()) {
            tt::tt_fabric::append_fabric_connection_rt_args(
                local_node_id, backward_fabric_node_id.value(), /*link_idx=*/f, program, {core}, fwd_rt);
        }
        SetRuntimeArgs(program, forwarder_kernel_ids[f], core, fwd_rt);
    }

    return {
        std::move(program),
        WanFusedDistributedRmsnormSharedVariables{
            .reader_kernel_ids = {reader_kernel_id},
            .writer_kernel_ids = {writer_kernel_id},
            .compute_kernel_ids = {compute_kernel_id},
            .forwarder_kernel_ids = forwarder_kernel_ids,
            .forwarder_cores = forwarder_cores,
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
            reader_args[3] = rope_cos_addr;
            reader_args[4] = rope_sin_addr;

            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
            writer_args[3] = trans_mat_addr;  // worker-writer + drain-only writer: trans_mat at rt[3]
            if (shared.stats_dram_addr_writer_arg_idx.has_value()) {
                // worker-writer (AG path): stats_dram scratch at rt[4].
                writer_args[shared.stats_dram_addr_writer_arg_idx.value()] = stats_dram_addr;
            }
        }
        // Forwarders read the stats DRAM scratch base at rt[0].
        for (size_t f = 0; f < shared.forwarder_kernel_ids.size(); f++) {
            auto& fwd_args_by_core = GetRuntimeArgs(program, shared.forwarder_kernel_ids[f]);
            const auto& fc = shared.forwarder_cores[f];
            fwd_args_by_core.at(fc.x).at(fc.y)[0] = stats_dram_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
