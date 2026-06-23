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
// Worker-count contention ceiling. A feat=1024 RoPE sweep (powers-of-2 tile-rows
// x workers x chunk) showed parallel scaling is near-linear up to 32 workers, then
// MUX-link + simultaneous-DRAM-read contention makes MORE workers SLOWER (64 was
// 4-20% worse than 32 at seq 2048/4096). So 32 is a perf ceiling, not a core budget.
// WAN_RMSNORM_WORKER_CAP overrides it for tuning/sweeps.
constexpr uint32_t kMaxMuxWorkersPerChip = 32u;
// num_tile_rows below this falls back to the LEGACY whole-tile writer
// (single worker). The packed-page MUX writer has significant per-chunk
// fabric overhead that doesn't pay off until we have ≥4 tile-rows worth
// of compute per chip — at that point parallelism + packed bytes win.
constexpr uint32_t kMuxRowsThreshold = 4u;

// Pick num_workers for the MUX/packed path: one worker per tile-row, capped at
// kMaxMuxWorkersPerChip (the contention ceiling). A powers-of-2 sweep (rope +
// non-rope) showed min(num_tile_rows, 32) is optimal at every sequence length —
// superseding the old SMALL/LARGE two-regime rule (which under-parallelized
// medium shapes via rows/2 and over-parallelized large ones). With chunk pinned
// to 1, fabric/AG is negligible (~2us), so there's no packet-count reason to use
// fewer workers. See the commit history / RMSNORM_FUSION_FINDINGS.md.
// Diagnostic override: WAN_RMSNORM_WORKER_CAP lets a perf sweep dial the
// per-chip worker cap without rebuilding. Read once. Defaults to
// kMaxMuxWorkersPerChip. Read inside the single-source-of-truth sizing path so
// the op and create_stats_buffer agree on num_workers/buffer geometry.
uint32_t mux_worker_cap() {
    static const uint32_t cap = [] {
        const char* env = std::getenv("WAN_RMSNORM_WORKER_CAP");
        if (env != nullptr) {
            const long v = std::strtol(env, nullptr, 10);
            if (v > 0) {
                return static_cast<uint32_t>(v);
            }
        }
        return kMaxMuxWorkersPerChip;
    }();
    return cap;
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
// buffer stays consistent with the kernel). Timing-only; combine with
// WAN_RMSNORM_NO_PERHEAD_CLAMP=1 to defeat the per-head/streaming clamps.
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
// Diagnostic: WAN_RMSNORM_NO_PERHEAD_CLAMP=1 skips forcing chunk_size_rows=1 for
// per-head RoPE, so the natural (chunk>=2) path runs — used to REPRODUCE the
// per-head-RoPE chunk>=2 compute deadlock under the watcher. Applied in BOTH
// compute_sizing and the program factory so the buffer stays consistent.
bool perhead_chunk_clamp_disabled() {
    static const bool v = (std::getenv("WAN_RMSNORM_NO_PERHEAD_CLAMP") != nullptr);
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
// WAN_RMSNORM_FORCE_FUSE_MM_ROPE=1: force the block-major (fused matmul+rope) POST
// for per-head RoPE even when the resident path would fit. Used to validate the
// fused path's correctness/perf on configs (e.g. TP=4) that have a clean reference.
bool force_fuse_mm_rope() {
    static const bool v = (std::getenv("WAN_RMSNORM_FORCE_FUSE_MM_ROPE") != nullptr);
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
uint32_t pick_num_workers_tp_gt_1(uint32_t num_tile_rows) {
    const uint32_t cap = mux_worker_cap();
    if (num_tile_rows < kMuxRowsThreshold) {
        return 1u;
    }
    const uint32_t forced = force_num_workers();
    if (forced > 0u) {
        return std::min<uint32_t>(forced, num_tile_rows);
    }
    // One worker per tile-row, capped at the contention ceiling (kMaxMuxWorkersPerChip).
    // The sweep showed min(rows, 32) is optimal at every sequence length: below 32 rows
    // more workers always help (compute-bound, ~linear scaling); above it the cap avoids
    // the >32-worker contention regression. Supersedes the old rows/2 large-shape rule,
    // which under-parallelized at 32 tile-rows and over-parallelized past 64.
    return std::min<uint32_t>(num_tile_rows, cap);
}

// Sizing derivation used in both spec computation (to size the stats scratch
// tensor in `compute_output_specs`) and the program factory (to lay out
// kernels + CBs). Single source of truth so the two cannot drift.
WanFusedDistributedRmsnormSizing compute_sizing(
    const WanFusedDistributedRmsnormParams& args,
    const Tensor& input,
    const WanFusedDistributedRmsnormInputs& tensor_args) {
    WanFusedDistributedRmsnormSizing s;
    const auto& padded = input.padded_shape();
    const uint32_t W = padded[-1];
    const uint32_t folded_H = input.physical_volume() / W;
    s.num_tile_rows = folded_H / TILE_HEIGHT;
    // per_head_norm reduces locally over head_dim per head — no AG needed even
    // when ring_size > 1. From the kernel's perspective, this is "is_tp_1" =
    // no fabric, no MUX, legacy writer path.
    s.is_tp_1 = (args.ring_size == 1) || args.per_head_norm;
    s.num_workers = s.is_tp_1 ? 1u : pick_num_workers_tp_gt_1(s.num_tile_rows);
    s.use_mux = !s.is_tp_1 && (s.num_workers > 1);
    // CRITICAL: mirror create_at's num_links rounding so the stats-buffer geometry
    // sized here (chunk_size_rows / num_chunks_per_device / page_size_bytes) agrees
    // with the program factory's actual kernel layout. create_at rounds num_workers
    // DOWN to a multiple of num_links_eff; if we skip that here, the two disagree
    // whenever pick() isn't already link-aligned. e.g. num_tile_rows=38 picks 19
    // workers -> rows_per_worker=2 -> chunk=2 here, but the factory rounds to 16 ->
    // rows_per_worker=3 -> chunk=3. The buffer is then laid out for 2-row pages while
    // the writer emits 3-row pages, so the AG scatters garbage into each chunk's last
    // row (uniform 2x output on those rows). Shapes where pick() is already a multiple
    // of num_links (e.g. 152 rows -> 64 workers) happen to agree, which is why only
    // some shapes were corrupted.
    {
        const uint32_t num_links_requested = std::max<uint32_t>(1u, args.num_links);
        const uint32_t num_links_eff = s.use_mux ? std::min<uint32_t>(num_links_requested, s.num_workers) : 1u;
        if (s.use_mux && num_links_eff > 1) {
            s.num_workers = (s.num_workers / num_links_eff) * num_links_eff;
            if (s.num_workers == 0) {
                s.num_workers = num_links_eff;
            }
        }
    }
    const uint32_t rows_per_worker = tt::div_up(s.num_tile_rows, s.num_workers);
    // Phase 9 packed-AG: one fabric mcast per chunk, so fewer chunks = fewer
    // fabric round-trips. Aim for 1 chunk per worker (= rows_per_worker rows
    // per packet) for the multichunk shape regime where each worker already
    // has few rows. Cap at kMaxChunkSizeRows for L1 budget (chunk-sized CBs:
    // input, stats_local, packed_gathered, stats_gathered all scale with
    // chunk_size).
    // Chunk cap = 1. A fuller sweep (rope + non-rope, real 38/152-tile-row sizes,
    // chunk 1-4 at W up to 64) showed chunk=1 is best or tied EVERYWHERE: fabric/AG
    // is only ~2us exposed so bigger chunks buy no amortization, and chunk>1 is ~10%
    // SLOWER on the large (152-row) shapes (the prefetch-overlap win never
    // materialized). So pin chunk=1; WAN_RMSNORM_FORCE_CHUNK still overrides for sweeps.
    constexpr uint32_t kMaxChunkSizeRows = 1u;
    // L1 budget cap: input_cb is double-buffered 2 * chunk * num_tile_cols
    // bf16 tiles = chunk * num_tile_cols * 4 KB per worker. Other CBs add
    // ~150 KB. Keep input_cb ≤ 512 KB so total ≤ 750 KB (half of L1):
    //   chunk * num_tile_cols ≤ 128.
    const uint32_t num_tile_cols_for_chunk_cap = std::max(1u, W / TILE_WIDTH);
    const uint32_t chunk_h_cap = std::max(1u, 128u / num_tile_cols_for_chunk_cap);
    s.chunk_size_rows =
        std::min<uint32_t>(std::min<uint32_t>(std::max(1u, rows_per_worker), kMaxChunkSizeRows), chunk_h_cap);

    // Clamp to a single resident row for per-head RoPE and for the streaming-low-L1
    // fallback — MUST match the program factory so the stats buffer's window/pages
    // agree. Non-clamped shapes keep the original (unrounded) chunk: the program
    // uses its own num_links-rounded chunk for compute, and the AG tolerates that
    // pre-existing window/chunk difference (only the CLAMPED cases need to match,
    // and there both sides are trivially 1). Detect per-head RoPE / streaming from
    // the same inputs the program factory uses.
    const uint32_t num_tile_cols = std::max(1u, W / TILE_WIDTH);
    const bool fuse_rope = tensor_args.transformation_mat.has_value() && tensor_args.rope_cos.has_value() &&
                           tensor_args.rope_sin.has_value();
    const bool per_head_rope =
        fuse_rope && (tensor_args.rope_cos->logical_shape()[1] == args.num_heads_per_device);
    bool streaming = false;
    if (s.use_mux && !per_head_rope) {
        const auto [c_fid, c_apx, c_fp32, c_pl1, c_dfs] =
            get_compute_kernel_config_args(input.device()->arch(), args.compute_kernel_config);
        const uint32_t block_size = get_dest_reg_count(args.compute_kernel_config);
        const uint32_t in_b = tt::tile_size(datatype_to_dataformat_converter(input.dtype()));
        const uint32_t out_b = tt::tile_size(datatype_to_dataformat_converter(args.dtype.value_or(input.dtype())));
        const uint32_t interm_b =
            c_fp32 ? tt::tile_size(tt::DataFormat::Float32) : tt::tile_size(tt::DataFormat::Float16_b);
        streaming = decide_streaming_low_l1(
            num_tile_cols, block_size, s.chunk_size_rows, in_b, interm_b, out_b, tensor_args.weight.has_value(),
            args.per_head_norm);
    }
    if ((per_head_rope && !perhead_chunk_clamp_disabled()) || streaming) {
        s.chunk_size_rows = 1u;
    }
    if (s.use_mux && force_chunk_size() > 0u) {
        s.chunk_size_rows = force_chunk_size();
    }
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
    if ((per_head_rope && !perhead_chunk_clamp_disabled()) || streaming_low_l1) {
        chunk_size_rows = 1u;
    }
    if (use_mux && force_chunk_size() > 0u) {
        chunk_size_rows = force_chunk_size();
    }
    // Phase 9 packed-page AG: every chunk this chip processes maps to a distinct
    // DRAM page. Page index = my_device_index * num_chunks_per_device + chunk_idx.
    const uint32_t num_chunks_per_device = use_mux ? tt::div_up(num_tile_rows, chunk_size_rows) : 0u;
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
    constexpr uint32_t stats_packed_local_cb_id = tt::CBIndex::c_17;
    constexpr uint32_t stats_packed_gathered_cb_id = tt::CBIndex::c_18;
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
        fuse_mm_rope = per_head_rope && (force_fuse_mm_rope() ||
                                         post_rotated_overflows_l1(
                                             num_tile_cols, block_size, input_tile_size, intermediate_tile_size,
                                             output_tile_size, rope_resident_tiles, rope_tile_size, has_weight));
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
        // Scalar/eps/trans_mat population args (writer populates these CBs so the
        // reader starts the input read ASAP). Appended AFTER the accessors so the
        // fixed/MUX/accessor CT indices above are unchanged; the kernel reads them
        // at stats_dram_args.next_compile_time_args_offset().
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
            "wan_rmsnorm_fused_writer_mux.cpp",
            worker_core_set,
            WriterDataMovementConfig(writer_compile_args, ablation_defines()));
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

    // Float32 input requires fp32 dest accumulation; otherwise the unpacker
    // would silently downcast through SrcA to TF32 / Float16_b (~10 mantissa
    // bits) and the pre-phase sum(x**2) loses precision when |x| is large.
    // Our compute kernel uses mul_tiles (SrcA/SrcB FPU path), which still
    // truncates SrcA to TF32 even when fp32_dest_acc_en is true — TF32's 10
    // mantissa bits is comparable to bf16, so we accept that precision floor
    // and don't set UnpackToDestFp32 (that mode only takes effect on the
    // unpack-to-dest paths like transpose_dest, used by Welford kernels).
    TT_FATAL(
        !(input_format == tt::DataFormat::Float32 && !fp32_dest_acc_en),
        "wan_fused_distributed_rmsnorm with Float32 input requires fp32_dest_acc_en=true in the "
        "compute kernel config.");

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
            // trans_mat addr for the writer-side scalar/trans_mat population.
            // Sits at stats_dram_addr_writer_arg_idx + 2; refreshed there in
            // override_runtime_arguments. 0 when no RoPE (writer won't read it).
            writer_rt_args.push_back(fuse_rope ? trans_mat.value().buffer()->address() : 0u);
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
        } else {
            // Legacy writer (is_tp_1 OR TP>1 single-worker). trans_mat addr at rt
            // index 3 for the writer-side scalar/trans_mat population (0 = no RoPE,
            // never read); refreshed in override_runtime_arguments.
            writer_rt_args.push_back(fuse_rope ? trans_mat.value().buffer()->address() : 0u);
            if (!is_tp_1) {
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
                        local_node_id,
                        backward_fabric_node_id.value(),
                        /*link_idx=*/0,
                        program,
                        {core},
                        writer_rt_args);
                }
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
            reader_args[3] = rope_cos_addr;
            reader_args[4] = rope_sin_addr;

            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
            if (shared.stats_dram_addr_writer_arg_idx.has_value()) {
                // MUX writer: stats_dram + (worker_chunk_base) + trans_mat.
                const size_t idx = shared.stats_dram_addr_writer_arg_idx.value();
                writer_args[idx] = stats_dram_addr;
                // trans_mat addr for the writer-side population sits at idx + 2
                // (stats_dram, worker_chunk_base, trans_mat — see create_at).
                writer_args[idx + 2] = trans_mat_addr;
            } else {
                // Legacy writer: trans_mat addr for the writer-side scalar/
                // trans_mat population sits at rt index 3.
                writer_args[3] = trans_mat_addr;
            }
        }
    }
}

}  // namespace ttnn::experimental::prim
