// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dit_fused_distributed_rmsnorm_program_factory.hpp"

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
//   is_tp_1 (ring_size==1, or per_head_norm): no all-gather — the compute kernel
//     reduces stats locally and the writer just drains output_cb to DRAM. No
//     fabric. Multiple worker cores per chip each take a slice of tile-rows.
//
//   All-gather path (ring_size>1, whole-row norm): per chip, num_workers worker
//     cores + num_forwarders (= num_links) forwarder cores. Each worker computes
//     its tile-rows' partial sum-of-squares (PRE), transposes the stat tile, and
//     NoC-writes a 128 B "stick" into its forwarder's coalesced packet. The
//     forwarder ring-multicasts the whole group's packet over fabric (one fused
//     write+atomic per round), so the gathered partial stats land in a DRAM
//     scratch page on every chip; workers read them back, finish the norm (POST,
//     + optional RoPE), and drain output. Kernels: reader (input/weight/rope),
//     compute (PRE+POST), worker_writer (stick push + gather read + output
//     drain), forwarder (coalesce + fabric mcast + cross-chip sync).
// =============================================================================

namespace {

uint32_t float_to_u32(float v) {
    uint32_t out;
    std::memcpy(&out, &v, sizeof(float));
    return out;
}

}  // namespace

// num_tile_rows below this uses a single worker — spinning up forwarders + the
// per-round AG handshake doesn't pay off with <4 tile-rows of compute per chip.
constexpr uint32_t kMuxRowsThreshold = 4u;
// input_cb double-buffer depth (chunks): reader fills chunk N+1 while compute is in
// chunk N's post phase. 2 = one in flight + one filling.
constexpr uint32_t kInputCbChunks = 2u;
// Depth (in block_size groups) of the STREAMED per-head cos/sin CBs — a couple of
// blocks of reader look-ahead (one in flight + one filling).
constexpr uint32_t kRopeStreamBlocks = 2u;

// Worker-count ceiling. Three bounds, smallest wins (details in the body):
//   1. grid budget: cores − one forwarder per link, rounded DOWN to whole grid rows
//      (workers are placed row-major; a ragged final row costs NoC/dispatch contention,
//      and the leftover idle cores are useful slack).
//   2. fabric-packet validity: workers_per_forwarder must fit one coalesced packet
//      (sticks_per_packet * num_forwarders).
//   3. measured DRAM/compute knee (arch-specific; Blackhole = 48).
// Read inside the single-source-of-truth sizing path so the op + create_stats_buffer
// agree on num_workers / buffer geometry.
uint32_t derive_worker_cap(
    const CoreCoord& grid_size,
    uint32_t num_links,
    tt::ARCH arch,
    uint32_t stick_bytes = 128u,
    uint32_t ring_size = 1u,
    uint32_t num_tile_rows = 0u) {
    const uint32_t max_cores = grid_size.x * grid_size.y;
    const uint32_t num_forwarders = std::max<uint32_t>(1u, num_links);  // one forwarder per link
    const uint32_t budget = max_cores > num_forwarders ? max_cores - num_forwarders : 1u;
    // Round down to whole grid rows (grid.x cores each); fall back to the raw budget if
    // even a single row doesn't fit.
    const uint32_t whole_rows = (grid_size.x > 0) ? (budget / grid_size.x) * grid_size.x : 0u;
    uint32_t cap = whole_rows > 0u ? whole_rows : budget;
    // DEV worker-count sweep knob: WAN_RMSNORM_WORKER_CAP overrides the grid-derived cap
    // (still validity-clamped below, and it bypasses the arch knee). Used to re-sweep the
    // per-shape optimum before finalizing the heuristic.
    const char* worker_cap_env = std::getenv("WAN_RMSNORM_WORKER_CAP");
    if (worker_cap_env != nullptr) {
        const int forced = std::atoi(worker_cap_env);
        if (forced > 0) {
            cap = static_cast<uint32_t>(forced);
        }
    }
    // Validity clamp: a forwarder coalesces at most sticks_per_packet 128 B sticks
    // into one fabric packet, so workers_per_forwarder (= ceil(cap/num_forwarders))
    // must not exceed it. Bound cap by sticks_per_packet * num_forwarders. On BH the
    // raw grid budget is 108 (12x10) -> 54 workers/forwarder > 32 -> would TT_FATAL;
    // this clamps it to the valid 64 (RMS, 128 B sticks) or 32 (LayerNorm, 256 B).
    const uint32_t sticks_per_packet =
        std::max<uint32_t>(1u, tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / stick_bytes);
    cap = std::min(cap, sticks_per_packet * num_forwarders);
    // Perf knee is ARCH- AND WORKLOAD-specific. The BH worker sweep (2026-07-01, in
    // REBENCH_baseline_vs_fused.md) shows the optimum is set by two competing effects:
    //   * per-round DRAM/NoC + fabric contention (favours FEWER workers), which scales
    //     with fabric pressure (ring_size = TP degree) and total row count; and
    //   * round count = ceil(rows/workers) (favours MORE workers).
    // The knee where contention wins:
    //   - ring_size >= 8 (8-hop AG, heavy fabric): 48 — e.g. FLUX tp8 N16384 (512 rows)
    //     48=275µs vs 64=335µs, N4096 (128) 48=83 vs 64=90.
    //   - ring_size <= 4, very large row counts: 48 — Wan self/cross_sp4 (592 rows)
    //     48=410/356µs vs 64=477/434µs (the two most expensive shapes; protect them).
    //   - ring_size <= 4, moderate rows: 64 — round-bound, more workers win. Biggest
    //     wins are the 152-row LTX s2 shapes (videoQ_s2 64=76µs vs 48=143µs = 1.9x) and
    //     Wan sp8 (296) / FLUX tp4 N2048,N8192 (64,256). (Balancing to fewer, evenly-
    //     loaded workers was tried and did NOT help — it is a round-count win, not a
    //     remainder-balance one.)
    // WH's grid-derived budget (64) is already its optimum, so no BH-style knee there.
    // Tuned to the DiT (Wan/LTX/FLUX) shape suite; 48 is the conservative fallback.
    // The WAN_RMSNORM_WORKER_CAP override bypasses this knee so it can sweep BH freely.
    if (arch == tt::ARCH::BLACKHOLE && worker_cap_env == nullptr) {
        constexpr uint32_t kBhContentionKnee = 48u;
        constexpr uint32_t kBhRoundBoundCap = 64u;
        constexpr uint32_t kBhRing4RowThreshold = 448u;  // between Wan sp8 (296, want 64) and sp4 (592, want 48)
        const uint32_t bh_knee =
            (ring_size <= 4u && num_tile_rows <= kBhRing4RowThreshold) ? kBhRoundBoundCap : kBhContentionKnee;
        cap = std::min(cap, bh_knee);
    }
    return cap;
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
// The byte estimate below mirrors the big-CB sizes allocated further down and
// is calibrated against the observed overflow (resident TP=2 allocates
// ~1,666,432 B). `kFixedOverheadBytes` covers the remaining small CBs (rope,
// stats, scalars, packed-AG, packet headers).
bool decide_streaming_low_l1(
    uint32_t num_tile_cols,
    uint32_t block_size,
    uint32_t chunk_size_rows,
    uint32_t input_tile_bytes,
    uint32_t intermediate_tile_bytes,
    uint32_t output_tile_bytes,
    bool has_weight,
    uint32_t weight_tile_bytes,
    bool per_head_norm,
    uint32_t extra_resident_bytes = 0u) {
    // per_head_norm uses head-block reduces over small head_dim shards; its L1
    // profile never overflows and the streamed compute path only handles the
    // whole-row reduce, so never auto-enable streaming for it.
    if (per_head_norm) {
        return false;
    }
    const uint32_t padded_row = ((num_tile_cols + block_size - 1u) / block_size) * block_size;
    const uint64_t input_bytes =
        static_cast<uint64_t>(kInputCbChunks) * chunk_size_rows * num_tile_cols * input_tile_bytes;
    // intermediate_cb + rotated_input_cb (both row-sized, intermediate_tile_bytes).
    const uint64_t intermediate_bytes = 2ull * padded_row * intermediate_tile_bytes;
    // output_cb is 2 padded rows.
    const uint64_t output_bytes = 2ull * padded_row * output_tile_bytes;
    // Broadcast weight is num_tile_cols tiles of weight_tile_bytes (2048 bf16 / 4096 fp32).
    // Per-token is larger but those shapes have small num_tile_cols and don't trigger streaming.
    const uint64_t weight_bytes = has_weight ? static_cast<uint64_t>(num_tile_cols) * weight_tile_bytes : 0ull;
    constexpr uint64_t kFixedOverheadBytes = 196608ull;      // ~192 KB of small CBs
    constexpr uint64_t kResidentL1BudgetBytes = 1572864ull;  // static-CB cap per core
    // extra_resident_bytes: CBs the resident layout adds beyond the above (e.g. the
    // Welford recip LUT, reduce_width*4 B). Counting them here lets a borderline-resident
    // shard (e.g. LTX TP2 at 64 tiles) correctly fall to the streaming/block-major layout
    // instead of overflowing L1 with the extra CB.
    const uint64_t total =
        input_bytes + intermediate_bytes + output_bytes + weight_bytes + kFixedOverheadBytes + extra_resident_bytes;
    return total > kResidentL1BudgetBytes;
}
// Block-major POST fallback. Streaming input_cb alone (decide_streaming_low_l1)
// only shrinks input_cb — intermediate_cb + rotated_input_cb + output_cb stay
// WHOLE-ROW resident, so the streamed-input layout still tops out around
// num_tile_cols ~= 80 on Wormhole. At wider low-TP shards (TP=1 WAN feat-5120 =
// 160 tile-cols, LTX-video feat-4096 = 128, FLUX feat-6144 = 192; TP=2 FLUX
// feat-3072 = 96) it overflows L1. When that happens we ALSO make intermediate/
// rotated/output block-local (O(block_size)) and the compute fuses every POST
// sub-phase into one per-block loop (block-major POST). This estimate sums the
// big CBs that block-major removes (intermediate/rotated/output whole-row) plus
// the whole-row weight/bias (exact: weight/bias_tile_bytes is the real bf16/fp32 CB
// size), against the real CB-usable L1 cap with margin; a calibrated constant covers
// the streamed input_cb + cos/sin + the ~dozen small stat/scalar CBs. block_major implies
// streaming_low_l1 (input is streamed too) and per_head_norm==0 (streaming is
// never auto-enabled for per_head_norm — that path is a separate L1 question).
bool decide_block_major_post(
    uint32_t num_tile_cols,
    uint32_t block_size,
    uint32_t intermediate_tile_bytes,
    uint32_t output_tile_bytes,
    bool has_weight,
    bool has_bias,
    uint32_t weight_tile_bytes,
    uint32_t bias_tile_bytes,
    uint32_t weight_cb_tiles,
    uint32_t bias_cb_tiles,
    bool fuse_rope,
    bool is_layernorm,
    bool per_head_norm,
    uint32_t input_tile_bytes,
    uint64_t l1_cap_bytes) {
    const uint32_t padded = ((num_tile_cols + block_size - 1u) / block_size) * block_size;
    // Whole-row CBs that the block-major layout collapses to O(block_size):
    uint64_t whole_row = static_cast<uint64_t>(padded) * intermediate_tile_bytes;  // intermediate_cb
    // rotated_input_cb sizing. With per_head_norm, create_cb keeps both rotated_input_cb and
    // input_cb resident whole-row even for RMS no-rope, so count both. Otherwise use the original
    // gate (unchanged for callers that don't set per_head_norm).
    if (per_head_norm) {
        whole_row += static_cast<uint64_t>(padded) * intermediate_tile_bytes;  // rotated_input_cb (whole-row)
        whole_row += 2ull * num_tile_cols * input_tile_bytes;                  // resident input_cb (2 rows)
    } else {
        whole_row += (fuse_rope || is_layernorm) ? static_cast<uint64_t>(padded) * intermediate_tile_bytes : 0ull;
    }
    whole_row += 2ull * padded * output_tile_bytes;  // output_cb (2 rows)
    // weight_cb / bias_cb tile counts (passed in): every affine mode holds ONE row (num_tile_cols)
    // — broadcast resident, per-token / per-batch streamed per row. block-major does NOT shrink
    // these (only the intermediate/output whole-row CBs), so the decision counts them here at
    // their real size (num_tile_cols) rather than assuming they collapse.
    whole_row += has_weight ? static_cast<uint64_t>(weight_cb_tiles) * weight_tile_bytes : 0ull;  // weight_cb
    whole_row += has_bias ? static_cast<uint64_t>(bias_cb_tiles) * bias_tile_bytes : 0ull;        // bias_cb
    // Streamed input_cb + resident cos/sin + the dozen small fp32 stat/scalar CBs +
    // forwarder packet/header. Calibrated against the observed FLUX TP=2 feat-3072
    // streamed-input allocation (1,601,824 B at the same big-CB sum).
    constexpr uint64_t kSmallCbOverheadBytes = 225000ull;
    return whole_row + kSmallCbOverheadBytes > l1_cap_bytes;
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
    uint64_t total = static_cast<uint64_t>(num_tile_cols) * input_tile_bytes;     // input_cb (chunk=1)
    total += 2ull * padded * intermediate_tile_bytes;                             // intermediate + rotated (whole row)
    total += 2ull * padded * output_tile_bytes;                                   // output_cb (2 rows)
    total += has_weight ? static_cast<uint64_t>(num_tile_cols) * 2048ull : 0ull;  // weight_cb
    total += 2ull * rope_cb_tiles * rope_tile_bytes;                              // streamed cos + sin
    constexpr uint64_t kPostFixedOverheadBytes = 491520ull;  // stats/packed-AG/pre-interm/scalars/trans/headers
    constexpr uint64_t kFuseTriggerBytes = 1400000ull;       // margin below the ~1.43 MB L1 cap
    return total + kPostFixedOverheadBytes > kFuseTriggerBytes;
}
uint32_t pick_num_workers_tp_gt_1(uint32_t num_tile_rows, uint32_t cap) {
    if (num_tile_rows < kMuxRowsThreshold) {
        return 1u;
    }
    // One worker per tile-row, clamped to the workload-aware cap (derive_worker_cap:
    // grid budget, fabric-packet validity, and the BH perf knee).
    return std::min<uint32_t>(num_tile_rows, cap);
}

// Sizing derivation used in both spec computation (to size the stats scratch
// tensor in `compute_output_specs`) and the program factory (to lay out
// kernels + CBs). Single source of truth so the two cannot drift.
DitFusedDistributedRmsnormSizing compute_sizing(
    const DitFusedDistributedRmsnormParams& args, const Tensor& input) {
    // Page geometry depends only on the input shape, ring size, links, and norm_type — NOT on
    // weight/bias/RoPE or the streaming decision (window_size is fixed: 1 here, sticks_per_packet *
    // stats_per_token on the mux path). So there is no tensor_args to consult.
    DitFusedDistributedRmsnormSizing s;
    const auto& padded = input.padded_shape();
    const uint32_t W = padded[-1];
    const uint32_t folded_H = input.physical_volume() / W;
    s.num_tile_rows = folded_H / TILE_HEIGHT;
    // per_head_norm reduces locally over head_dim per head — no AG needed even
    // when ring_size > 1. From the kernel's perspective this is "is_tp_1" =
    // no fabric, no all-gather, drain-only writer.
    s.is_tp_1 = (args.ring_size == 1) || args.per_head_norm;
    // LayerNorm transports 2 stats/token (mean, M2) -> 256 B sticks; RMS 1 (128 B).
    // Set before derive_worker_cap so its packet-capacity clamp uses the right width.
    s.stats_per_token = (args.norm_type == DitFusedNormType::LAYERNORM) ? 2u : 1u;
    s.stick_bytes = s.stats_per_token * 128u;
    // Worker cap = device compute grid − forwarder cores. Derived from the input's
    // device so create_stats_buffer / validate / compute_output_specs / create_at all
    // agree on num_workers (they share this single-source-of-truth path). Only evaluated
    // on the !is_tp_1 (all-gather) branch: derive_worker_cap calls into fabric
    // (get_tt_fabric_max_payload_size_bytes), which reaches the fabric context — null when
    // the op runs on a single device with fabric uninitialized (TP=1). is_tp_1 forces
    // num_workers=1 and never uses the cap, so short-circuit before touching fabric.
    s.num_workers = s.is_tp_1 ? 1u
                              : pick_num_workers_tp_gt_1(
                                    s.num_tile_rows,
                                    derive_worker_cap(
                                        input.device()->compute_with_storage_grid_size(),
                                        args.num_links,
                                        input.device()->arch(),
                                        s.stick_bytes,
                                        args.ring_size,
                                        s.num_tile_rows));
    // use_mux selects the fabric-forwarder all-gather path (+ DRAM scratch);
    // !use_mux (is_tp_1) reduces locally with no fabric.
    s.use_mux = !s.is_tp_1;
    // The forwarder round == one tile-row; sticks are coalesced across the
    // forwarder's worker group, so the window is not a row-batching knob here.
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
        // Token-tiles (each stick_bytes wide) that fit one fabric packet.
        const uint32_t sticks_per_packet =
            std::max<uint32_t>(1u, tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / s.stick_bytes);
        // window_size is the logical fp32 page width / TILE_HEIGHT; keep the page
        // formula page_size = TILE_HEIGHT * window_size * 4 == sticks_per_packet * stick_bytes.
        s.window_size = sticks_per_packet * s.stats_per_token;
        s.num_chunks_per_device = num_forwarders * max_rounds;
        s.total_pages = args.ring_size * s.num_chunks_per_device;
        s.page_size_bytes = TILE_HEIGHT * s.window_size * sizeof(float);
    }
    return s;
}

tt::tt_metal::TensorSpec make_stats_tensor_spec(const DitFusedDistributedRmsnormSizing& sizing) {
    // Row-major fp32 DRAM-interleaved scratch: one accessor page per packed stats page.
    // Kept in one place so the pre-alloc helper, compute_output_specs, and validate agree.
    ttnn::Shape stats_shape({1u, 1u, sizing.total_pages, TILE_HEIGHT * sizing.window_size});
    MemoryConfig stats_mem{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM};
    return tt::tt_metal::TensorSpec(
        stats_shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::ROW_MAJOR), stats_mem));
}

DitFusedDistributedRmsnormMeshWorkloadFactory::cached_program_t
DitFusedDistributedRmsnormMeshWorkloadFactory::create_at(
    const DitFusedDistributedRmsnormParams& args,
    const ttnn::MeshCoordinate& mesh_coordinate,
    const DitFusedDistributedRmsnormInputs& tensor_args,
    std::vector<Tensor>& tensor_return_value) {
    Tensor& output_tensor = tensor_return_value.at(0);
    // Stats DRAM scratch: only allocated for the all-gather path (TP>1, whole-row
    // norm). When not allocated (is_tp_1) the drain-only writer doesn't reference it.
    Tensor* stats_dram_tensor = tensor_return_value.size() > 1 ? &tensor_return_value[1] : nullptr;
    const auto& input_tensor = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& bias = tensor_args.bias;
    const auto& trans_mat = tensor_args.transformation_mat;
    const auto& rope_cos = tensor_args.rope_cos;
    const auto& rope_sin = tensor_args.rope_sin;
    const auto& reciprocals = tensor_args.reciprocals;

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
    // Per-batch (adaLN) weight/bias: shape [batch, 1, H] — broadcast over seq (logical[-2]==1,
    // so NOT per-token) but distinct per batch. Detected by the affine tensor spanning `batch`
    // padded tile-rows (broadcast [1,1,H] spans exactly 1). weight_cb / bias_cb hold ONE row
    // (num_tile_cols); the reader STREAMS each output row's batch slice into them per row (batch
    // index wbatch = global_tile_row / rows_per_batch_tiles), so compute consumes per row with no
    // batch offset of its own — same as per-token. batch>1 is only reachable with
    // num_heads_per_device==1 (validated in the device op).
    const uint32_t batch = input_tensor.logical_shape()[1];
    const uint32_t rows_per_batch_tiles = (batch > 0) ? (num_tile_rows / batch) : num_tile_rows;
    auto affine_tile_rows = [](const Tensor& t) -> uint32_t {
        return (t.physical_volume() / t.padded_shape()[-1]) / TILE_HEIGHT;  // padded tile-rows spanned
    };
    const bool per_batch_weight =
        has_weight && !per_token_weight && batch > 1 && affine_tile_rows(*weight) == batch;
    const bool per_batch_bias = has_bias && !per_token_bias && batch > 1 && affine_tile_rows(*bias) == batch;
    // Broadcast reader read count: 1 row for true-broadcast, `batch` rows for per-batch adaLN.
    // Broadcast bulk-read count is one row (num_tile_cols): only TRUE broadcast [1,1,H] uses the
    // one-shot resident read. per-batch adaLN now streams its per-row batch slice (like per-token),
    // so it does NOT bulk-read all batches here.
    const uint32_t weight_bcast_tiles = num_tile_cols;
    const uint32_t bias_bcast_tiles = num_tile_cols;
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
    // Batched RoPE: cos/sin dim0 is 1 (broadcast the same RoPE to every input batch) or `batch`
    // (per-batch cos/sin). The reader indexes cos/sin by the within-batch seq row plus a per-batch
    // offset of rope_batch_stride_tiles tiles (== one batch's whole cos/sin: num_heads_dim *
    // rope_seqlen_tiles * head_dim_tiles). 0 for the broadcast case -> the offset collapses to 0.
    const bool rope_per_batch = fuse_rope && batch > 1 && (rope_cos->logical_shape()[0] == batch);
    const uint32_t rope_num_heads_dim = per_head_rope ? num_heads_per_device : 1u;
    const uint32_t rope_batch_stride_tiles =
        rope_per_batch ? (rope_num_heads_dim * rope_seqlen_tiles * head_dim_tiles) : 0u;

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
    // drain-only writer even with ring_size > 1.
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

    // LayerNorm gathers 2 stats/token (mean, M2) -> 256 B sticks; RMS 1 (128 B).
    // Must match compute_sizing so the stats-buffer geometry and the program agree.
    const uint32_t stick_bytes = (args.norm_type == DitFusedNormType::LAYERNORM) ? 256u : 128u;
    const uint32_t stats_per_token = stick_bytes / 128u;  // 1 RMS, 2 LayerNorm (mean, var)

    uint32_t num_workers;
    if (is_tp_1) {
        num_workers = std::min<uint32_t>(max_cores, num_tile_rows);
    } else {
        // TP>1 (forwarder AG): one worker per tile-row, capped at the core budget
        // (grid − forwarders). Same derivation as compute_sizing so the stats-buffer
        // geometry matches. Tiny shapes (<kMuxRowsThreshold) collapse to 1 worker.
        num_workers = pick_num_workers_tp_gt_1(
            num_tile_rows,
            derive_worker_cap(grid_size, args.num_links, device->arch(), stick_bytes, args.ring_size, num_tile_rows));
    }
    use_mux = !is_tp_1;  // "uses the fabric-forwarder all-gather"

    // Forwarder model: one coalescing forwarder core per independent routing
    // plane (num_forwarders = min(num_links, num_workers)). Each forwarder owns a
    // contiguous worker group and holds the fwd+bwd fabric connections for its link.
    const uint32_t num_links_requested = std::max<uint32_t>(1u, args.num_links);
    const uint32_t num_forwarders = use_mux ? std::min<uint32_t>(num_links_requested, num_workers) : 0u;
    const uint32_t total_cores_needed = num_workers + num_forwarders;
    TT_FATAL(
        total_cores_needed <= max_cores,
        "dit_fused_distributed_rmsnorm needs {} cores ({} workers + {} forwarders) but only {} available",
        total_cores_needed,
        num_workers,
        num_forwarders,
        max_cores);

    const uint32_t num_tile_rows_per_worker = tt::div_up(num_tile_rows, num_workers);
    const uint32_t workers_per_forwarder = use_mux ? tt::div_up(num_workers, num_forwarders) : num_workers;
    const uint32_t max_rounds = num_tile_rows_per_worker;  // forwarder round == tile-row

    // [worker_0..N-1, forwarder_0..F-1] on the device grid (row-major).
    const auto all_cores_vec = corerange_to_cores(core_grid, max_cores, /*row_wise=*/true);
    std::vector<CoreCoord> worker_cores(all_cores_vec.begin(), all_cores_vec.begin() + num_workers);
    std::vector<CoreCoord> forwarder_cores;
    forwarder_cores.reserve(num_forwarders);
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

    // The compute kernel processes one tile-row at a time (a sweep over chunk 1-4 showed
    // chunk=1 best-or-tied everywhere: fabric/AG is only ~2us exposed so bigger chunks buy
    // no amortization, and chunk>1 was ~10% slower on the large shapes). So chunk is fixed
    // at 1 and is no longer a compute-kernel arg; kept as a local for the CB/AG-page sizing
    // below (which is trivially per-row). num_chunks_per_device (below) = the AG page count.
    const uint32_t chunk_size_rows = 1u;

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

    // Welford reciprocal LUT (LayerNorm only): a caller-provided fp32 [.., reduce_width]
    // DRAM tensor of [1/1..1/reduce_width]. The reader NoC-reads it once into a CB so the
    // Welford LLK does an array load instead of a soft-float 1/(N+1) per sample (PRE is
    // ~90% of LN compute; the division is ~15-30% of PRE). Absent -> runtime-division
    // fallback. reduce_width == feat_local == num_tile_cols * 32; the LUT is one row-major
    // page of reduce_width * sizeof(float) bytes (== num_tile_cols * 128). Defined here
    // (before the streaming decision) so its resident CB cost is accounted for.
    const bool is_layernorm = (args.norm_type == DitFusedNormType::LAYERNORM);
    const bool use_recip_lut = is_layernorm && reciprocals.has_value();
    const uint32_t recip_lut_bytes = num_tile_cols * 128u;
    // Per-batch adaLN's tile-offset indexing is only wired into the LayerNorm compute kernel
    // (the RMS kernel ignores CT 41/42/43 and would silently apply batch-0's weight to all
    // batches). Reject a per-batch affine tensor on the RMS path up front.
    TT_FATAL(
        !(per_batch_weight || per_batch_bias) || is_layernorm,
        "Per-batch (adaLN) weight/bias ([batch,1,H]) is only supported on the LayerNorm path");

    // weight/bias CB formats follow the tensor dtype (bf16 or fp32). Computed here (before the
    // L1 decisions) so the resident-budget estimates count the true CB byte size for fp32 affine.
    const tt::DataFormat weight_format = has_weight ? datatype_to_dataformat_converter(weight->dtype()) : bf16_format;
    const tt::DataFormat bias_format = has_bias ? datatype_to_dataformat_converter(bias->dtype()) : bf16_format;
    const uint32_t weight_tile_sz = tt::tile_size(weight_format);
    const uint32_t bias_tile_sz = tt::tile_size(bias_format);
    // welford_zero_cb (LayerNorm warm-row accumulator reset): 2 resident fp32 tiles, always present
    // for LN. Counted in the resident budget so wide LN shards correctly choose block-major.
    const uint32_t welford_zero_bytes = is_layernorm ? 2u * fp32_tile_size : 0u;

    // Streaming low-L1 fallback: when the resident input_cb + row-sized
    // intermediate/rotated/output CBs would overflow L1, stream input_cb in
    // block_size chunks for PRE + a POST re-read pass instead. See
    // decide_streaming_low_l1() for the budget heuristic.
    const bool streaming_input = decide_streaming_low_l1(
        num_tile_cols,
        block_size,
        chunk_size_rows,
        input_tile_size,
        intermediate_tile_size,
        output_tile_size,
        has_weight,
        weight_tile_sz,
        args.per_head_norm,
        (use_recip_lut ? recip_lut_bytes : 0u) + welford_zero_bytes);
    // Block-major POST: even input-streaming leaves intermediate/rotated/output
    // whole-row, which overflows L1 on wide low-TP shards. When so, shrink those
    // CBs to block-local + run the fused per-block POST. Margin below l1_size_per_core
    // covers the ~72 KB firmware/kernel-config reserve plus slack.
    //
    // Two flavors:
    //  - whole-row norm: needs input STREAMING (the POST re-reads input pass 1), so it
    //    only engages when decide_streaming_low_l1 already chose to stream.
    //  - per_head_norm: keeps input RESIDENT and runs a HEAD-MAJOR POST (each head's
    //    head_dim_tiles cols fully processed: per-head 1/rms -> affine -> rope ->
    //    output, head-local CBs). No streaming needed (input stays resident through
    //    PRE+POST), so it engages on per_head_norm directly. per_head_norm forces
    //    is_tp_1, so there's no fabric/AG ordering to worry about.
    // Welford LayerNorm runs a dedicated compute kernel. Wide shards use the same
    // streaming-input (Welford PRE re-reads the row) + block-major POST machinery
    // as RMS; the reader's defer_input (block_major && is_tp_1 && streaming) keeps
    // the no-AG path from deadlocking. per_head_norm LayerNorm is not supported.
    if (is_layernorm) {
        TT_FATAL(!args.per_head_norm, "LayerNorm does not support per_head_norm (block-major head-major path)");
    }
    // is_layernorm / use_recip_lut / recip_lut_bytes are defined above (before the
    // streaming decision, so the recip CB's resident cost is accounted for there).
    uint64_t l1_cap_bytes = static_cast<uint64_t>(device->l1_size_per_core()) - 112640ull;
    if (use_recip_lut) {
        // Account for the resident recip CB so the block-major decision still fits L1.
        l1_cap_bytes -= recip_lut_bytes;
    }
    // welford_zero_cb is resident for LN regardless of layout; reserve it from the cap too.
    l1_cap_bytes -= welford_zero_bytes;
    // weight/bias CB tile counts — MUST match the create_cb sizing below. All modes hold ONE
    // row (num_tile_cols): broadcast resident, per-token / per-batch streamed per row.
    const uint32_t weight_cb_tiles_est = has_weight ? num_tile_cols : 0u;
    const uint32_t bias_cb_tiles_est = has_bias ? num_tile_cols : 0u;
    const bool overflows_resident_post = decide_block_major_post(
        num_tile_cols,
        block_size,
        intermediate_tile_size,
        output_tile_size,
        has_weight,
        has_bias,
        weight_tile_sz,
        bias_tile_sz,
        weight_cb_tiles_est,
        bias_cb_tiles_est,
        fuse_rope,
        is_layernorm,
        args.per_head_norm,
        input_tile_size,
        l1_cap_bytes);
    // When the resident POST overflows, the whole-row path MUST stream (block-major
    // re-reads input pass 1), so force streaming even if the input alone would fit
    // (decide_streaming_low_l1 false). Then block-major engages on the overflow.
    // The recip CB (reduce_width*4 B) pushes a borderline-resident AG LayerNorm shard
    // (e.g. LTX TP2 at 64 tiles, only ~6.7 KB resident headroom) over L1. The resident-fit
    // heuristics account exactly for the big CBs (input/intermediate/output/weight/bias/recip/
    // welford_zero) but still lump the small AG stats/packet CBs into a calibrated constant
    // (~100 KB), so neither catches the ~1.5 KB recip overflow. Force the streaming +
    // block-major layout for wide-ish AG LN-with-recip
    // shards: block-major makes the whole-row POST CBs block-local, freeing far more than
    // the recip CB costs. The 56-tile threshold keeps the smaller AG configs resident
    // (WAN TP4=40, FLUX TP2=48 fit the recip CB) while catching LTX TP2 (64).
    const bool force_recip_stream = use_recip_lut && use_mux && (num_tile_cols >= 56u);
    const bool streaming_low_l1 =
        streaming_input || force_recip_stream || (overflows_resident_post && !args.per_head_norm);
    // INVARIANT: streamed input REQUIRES a block-major POST. When streaming_low_l1 is set the
    // input_cb is block-sized and the reader pushes the row block-by-block; the resident POST
    // instead reads input_cb[col] across the WHOLE row, which for a block-sized CB wraps the ring
    // and reads stale tiles -> garbage (PCC~0). So for the non-per_head path block_major_post must
    // track streaming_low_l1 exactly. The old form `(overflows || force_recip) && streaming_low_l1`
    // dropped block-major when streaming was triggered by decide_streaming_low_l1 alone
    // (streaming_input) while decide_block_major_post said the POST fit resident — the two use
    // different L1 budgets (hardcoded 1.5 MB vs the arch L1 cap), so they disagree on Blackhole
    // (larger L1) for wide no-affine shards (dim=3072 -> 96 tile-cols, no weight/bias to tip
    // decide_block_major_post over), landing in the broken streamed-input + resident-POST combo.
    // streaming_low_l1 already implies num_tile_cols % block_size == 0 (the guard below), so
    // block-major's divisibility requirement is satisfied. per_head_norm keeps its own layout.
    const bool block_major_post =
        args.per_head_norm ? (overflows_resident_post || force_recip_stream) : streaming_low_l1;
    // TP>1 wide (streaming input 2-pass + block-major POST + fabric AG) uses the
    // reader's SPLIT input schedule: the PRE pass is read first so the local stats /
    // ring gather start ASAP, then the side inputs, then the POST re-read pass (weight
    // now resident before the block-major POST consumes it). This avoids the
    // reader<->compute deadlock without delaying the AG. Applies to RMS and LayerNorm.
    // Clamp to a single resident row for (a) per-head RoPE — its cos/sin CBs are
    // chunk*num_tile_cols fp32 tiles (overflow L1 at feat>=1024) and the compute
    // deadlocks at chunk>=2 with many rows; and (b) the streaming-low-L1 path,
    // which only supports one resident row. compute_sizing applies the SAME clamp
    // so the caller's stats buffer (window/pages) matches. feat-2048 per-head RoPE
    // still can't fit even one row -> clean compile-time CB-alloc OOM (needs
    // cos/sin streaming, a separate change), NOT a hang.
    // Forwarder AG: DRAM pages per device = num_forwarders * max_rounds (one page
    // per forwarder per row-round). Page idx = my_device*num_chunks_per_device +
    // forwarder*max_rounds + round.
    const uint32_t num_chunks_per_device = use_mux ? (num_forwarders * max_rounds) : 0u;
    TT_FATAL(
        !streaming_low_l1 || (num_tile_cols % block_size == 0),
        "dit_fused_distributed_rmsnorm streaming low-L1 path requires num_tile_cols ({}) divisible by block_size "
        "({})",
        num_tile_cols,
        block_size);
    // The resident LayerNorm POST now clamps its sub-phases to the per-block tail count,
    // so non-divisible num_tile_cols is supported there (e.g. SD3.5: dim 2432 -> 38/19
    // tile-cols at TP2/TP4). The block-major LayerNorm POST, however, streams input and
    // its PRE waits cb_wait_front(input_cb, block_size) per block — the shared reader
    // pushes only the tail count on the last block, so a non-divisible width would hang
    // that path. block-major LN needs num_tile_cols wide enough to stream (>=56) and is
    // only reached by divisible shapes today; keep a fail-fast guard for it until it gets
    // tail handling too.
    TT_FATAL(
        !(is_layernorm && block_major_post) || (num_tile_cols % block_size == 0),
        "dit_fused_distributed block-major LayerNorm requires num_tile_cols ({}) divisible by block_size ({}) "
        "(the block-major streaming POST path has no tail handling yet)",
        num_tile_cols,
        block_size);

    // ------------------------------------------------------------------------
    // Persistent DRAM stats scratch (all-gather path only).
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
    // Forwarder all-gather CBs (use_mux path):
    //   stats_transposed_local_cb : 1 fp32 tile, post-transpose. The per-token
    //       sum-of-squares lives in row 0 (2 contiguous 64-byte spans at
    //       face_00[0] and face_01[0]). Compute pushes; the worker writer pops it
    //       to emit its 128 B stick.
    //   packet_cb : the forwarder's coalesced fabric packet (grid-uniform L1
    //       address, depth 2). Workers NoC-write their sticks into it; the
    //       forwarder ring-mcasts it.
    //   stats_transposed_gathered_cb : ring_size fp32 tiles. The worker writer
    //       lands the ring gather in row 0 of these; compute transposes them back
    //       to col 0 so the post-reduce<AVG,REDUCE_ROW> chain runs unchanged.
    constexpr uint32_t stats_transposed_local_cb_id = tt::CBIndex::c_16;
    constexpr uint32_t packet_cb_id = tt::CBIndex::c_17;  // forwarder coalesced packet (grid-wide, depth 2)
    constexpr uint32_t stats_transposed_gathered_cb_id = tt::CBIndex::c_19;
    constexpr uint32_t bias_cb_id = tt::CBIndex::c_20;
    // Welford reciprocal LUT CB (LayerNorm; c_18 is otherwise free). One contiguous page
    // of reduce_width fp32 reciprocals; the reader fills it once, compute reads it as a
    // std::array<uint32_t, reduce_width>. A tiny stub when the LUT is unused.
    constexpr uint32_t recip_lut_cb_id = tt::CBIndex::c_18;
    // Holds a zeroed Welford state (mean=0 tile, M2=0 tile) captured ONCE at cold start while the
    // SFPU is clean, then reloaded each row via copy_tile + welford_restore_state to reset the
    // welford accumulator (mirrors the standard layernorm_large_tensor_welford fuse_pre_add path).
    // copy_tile is an unpredicated L1->DST read, so it resets every token lane — unlike
    // welford_init's SFPLOADI clear, which a prior row's combine can leave CC-predicated (ISSUE 3A).
    constexpr uint32_t welford_zero_cb_id = tt::CBIndex::c_21;

    // Double-buffer input_cb: reader can fill chunk N+1 while compute is in
    // chunk N's post phase. The cumulative cb_wait_front in compute pairs
    // naturally with this — compute uses absolute indices within the
    // current chunk, and cb_pop_front at end of chunk frees that chunk's slots
    // back to the reader.
    //
    // Streaming low-L1: input_cb holds only a handful of block_size-tile blocks
    // (constant in num_tile_cols). The reader streams the row twice — once for
    // PRE, once for the POST re-read — and compute pops each block as it goes,
    // so the CB just needs enough depth for the reader to run a few blocks
    // ahead of compute within each pass.
    const uint32_t chunk_input_tiles = chunk_size_rows * num_tile_cols;
    // Shallower streaming depth on the AG path (use_mux): the per-shard stats +
    // ring-gather + combine CBs consume L1 that the is_tp_1 path doesn't, so a
    // streamed wide TP>1 shard (LayerNorm) would overflow at depth 4. Depth 2 is
    // still double-buffered (reader one block ahead). RMS TP>1 shapes fit resident
    // (never stream), so this only affects wide TP>1 LayerNorm.
    const uint32_t kStreamingInputBlocks = use_mux ? 2u : 4u;
    const uint32_t input_cb_tiles =
        streaming_low_l1 ? (kStreamingInputBlocks * block_size) : (kInputCbChunks * chunk_input_tiles);
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
    // Both paths (forwarder AG and is_tp_1) push gathered stats one chunk at a time,
    // so a chunk-sized CB suffices.
    const uint32_t stats_gathered_rows = chunk_size_rows;
    const uint32_t stats_gathered_tiles = stats_gathered_rows * per_row_stats_count;
    create_cb(stats_gathered_cb_id, program, worker_core_set, fp32_tile_size, stats_gathered_tiles, fp32_format);

    // Transposed stat CBs on the worker cores. For is_tp_1 these are unused stubs
    // (that path keeps stats in col 0 and reduces locally). chunk==1 so local is
    // stats_per_token (1 RMS sum-sq, 2 LayerNorm mean+var); gathered is
    // stats_per_token * ring_size (the per-device partials to merge).
    create_cb(stats_transposed_local_cb_id, program, worker_core_set, fp32_tile_size, stats_per_token, fp32_format);
    create_cb(
        stats_transposed_gathered_cb_id,
        program,
        worker_core_set,
        fp32_tile_size,
        use_mux ? args.ring_size * stats_per_token : 1u,
        fp32_format);
    uint32_t unit_packet_bytes = 0u;
    if (use_mux) {
        // Coalesced fabric packet, allocated on the WHOLE grid so its L1 address
        // is identical on every worker + forwarder core (a worker writes its 128 B
        // stick into its forwarder's copy at this same address; the forwarder reads
        // its own). page == one fabric packet (sticks_per_packet * 128 B), depth 2.
        const uint32_t sticks_per_packet =
            std::max<uint32_t>(1u, tt::tt_fabric::get_tt_fabric_max_payload_size_bytes() / stick_bytes);
        unit_packet_bytes = sticks_per_packet * stick_bytes;
        TT_FATAL(
            sticks_per_packet >= workers_per_forwarder,
            "dit_fused_distributed_rmsnorm: fabric packet holds {} sticks but a forwarder group has {} workers",
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
    // weight/bias CBs use the tensor's own dtype (bf16 or fp32) — fp32 affine keeps the
    // modulation precision adaLN needs; the reader derives its face-row stride from the
    // CB tile size, the compute reconfigs the FPU operand format, so no kernel change needed.
    // (weight_format / bias_format / weight_tile_sz / bias_tile_sz computed earlier, by the
    // L1-decision block, so the resident estimates see the true fp32 sizes.)
    // weight_cb / bias_cb hold ONE row's worth (num_tile_cols) for every mode:
    //  - broadcast [1,1,H]: the single row, resident (never popped).
    //  - per-token [.,N,H]: this row's slice, re-pushed + popped per row.
    //  - per-batch [batch,1,H] adaLN: this row's batch slice (face-row broadcast), re-pushed +
    //    popped per row (streamed, not all-batches-resident) — so a wide per-batch shard fits L1
    //    at TP=1 (the full unsharded width) instead of holding batch*num_tile_cols tiles.
    // chunk_size_rows==1, so per-token's chunk_size_rows*num_tile_cols is also just num_tile_cols.
    const uint32_t weight_cb_tiles = has_weight ? num_tile_cols : 1;
    create_cb(weight_cb_id, program, worker_core_set, weight_tile_sz, weight_cb_tiles, weight_format);
    const uint32_t bias_cb_tiles = has_bias ? num_tile_cols : 1;
    create_cb(bias_cb_id, program, worker_core_set, bias_tile_sz, bias_cb_tiles, bias_format);
    // Recip LUT CB: one contiguous page of reduce_width fp32 (== recip_lut_bytes). A 4 B
    // stub when unused (the reader/compute gate on use_recip and never touch it).
    create_cb(recip_lut_cb_id, program, worker_core_set, use_recip_lut ? recip_lut_bytes : 4u, 1u, fp32_format);

    // Zeroed welford-state scratch (LayerNorm only): 2 fp32 tiles (mean, M2). 1-tile stub otherwise.
    create_cb(welford_zero_cb_id, program, worker_core_set, fp32_tile_size, is_layernorm ? 2u : 1u, fp32_format);
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
        // Block-major POST consumes cos/sin RESIDENT (the reader pushes the whole
        // row's cos/sin before the deferred POST input re-read pass), so never
        // stream them when block_major_post — keep the whole-row resident size.
        const uint32_t rope_cb_tiles = (per_head_rope && fuse_mm_rope && !block_major_post)
                                           ? std::min(rope_resident_tiles, kRopeStreamBlocks * block_size)
                                           : rope_resident_tiles;
        create_cb(rope_cos_cb_id, program, worker_core_set, rope_tile_size, rope_cb_tiles, rope_format);
        create_cb(rope_sin_cb_id, program, worker_core_set, rope_tile_size, rope_cb_tiles, rope_format);
    } else {
        create_cb(rope_cos_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
        create_cb(rope_sin_cb_id, program, worker_core_set, fp32_tile_size, 1, fp32_format);
    }

    // intermediate_cb and rotated_input_cb are compute-only (producer and
    // consumer are the same TRISC pipeline within the post phase). The post phase
    // is sub-phase-major — it runs each sub-phase (mul-rms, weight, matmul,
    // cos-mul, sin-mul, add) across ALL col-blocks before moving to the next, so
    // these CBs must hold a full row between sub-phases.
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
    // Block-major POST collapses intermediate_cb to O(block_size) (double-buffered)
    // so wide low-TP shards fit L1; the resident/input-streaming paths keep the
    // whole (padded) row for sub-phase-major handoff.
    const uint32_t intermediate_cb_tiles =
        block_major_post ? (2u * block_size) : (tt::div_up(num_tile_cols, block_size) * block_size);
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
        rotated_input_cb_id, program, worker_core_set, intermediate_tile_size, rotated_cb_tiles, intermediate_format);
    // output_cb sized to 2 full padded rows so the writer can deep-drain a
    // whole row under ONE noc_async_writes_flushed() (instead of flushing every
    // block_size=2 tiles) while compute produces the next row. The shallow
    // 4-tile CB capped write concurrency at block_size, leaving the output
    // drain at ~22% of POST as back-pressure (P_ADD's 5.9× core-to-core spread
    // = shared DRAM-write contention). A row-deep CB lets writes pipeline at
    // DRAM depth (the writer drains the whole row under one flush).
    const uint32_t output_cb_tiles = 2u * intermediate_cb_tiles;
    create_cb(output_cb_id, program, worker_core_set, output_tile_size, output_cb_tiles, output_format);

    // Packet header CB. The forwarder reserves 2 header slots (fwd+bwd) from it,
    // so on the AG path it lives on the FORWARDER cores. is_tp_1's drain-only
    // writer never touches it (1-slot stub on the worker cores).
    // get_tt_fabric_packet_header_size_bytes() reaches the fabric context (null at TP=1 on a
    // single device with fabric uninitialized). Only the AG path (use_mux) uses this CB, so on
    // the is_tp_1 path use a fixed stub page size instead of querying fabric — the stub is
    // allocated for CB-index consistency but never read.
    const uint32_t packet_header_size_bytes =
        use_mux ? tt::tt_fabric::get_tt_fabric_packet_header_size_bytes() : sizeof(uint32_t);
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
        static_cast<uint32_t>(per_head_rope),
        rope_seqlen_tiles,
        bias_cb_id,
        static_cast<uint32_t>(has_bias),
        static_cast<uint32_t>(per_token_weight),
        static_cast<uint32_t>(per_token_bias),
        static_cast<uint32_t>(streaming_low_l1),
        // reader input_schedule (CT arg 17): WHERE the streaming input passes are read
        // relative to the resident weight/bias/cos pushes that the block-major POST
        // consumes mid-pass. 0=INPUT_FIRST (all input at top: resident, or streaming
        // with a resident POST), 1=DEFER_ALL (side inputs then both passes: is_tp_1
        // block-major — no AG, so delaying PRE is free), 2=SPLIT (PRE pass at top so the
        // local stats / ring gather start ASAP, then side inputs, then the POST pass:
        // the AG (ring>1) block-major path — fixes the deadlock without delaying the AG).
        static_cast<uint32_t>(
            (block_major_post && streaming_low_l1) ? (is_tp_1 ? 1u /*DEFER_ALL*/ : 2u /*SPLIT*/) : 0u /*INPUT_FIRST*/),
        // CT 17/18: recip LUT (LayerNorm). use_recip gates a one-time DRAM read of the
        // reciprocals tensor into recip_lut_cb at the top of the reader; compute then
        // reads the CB as the Welford reciprocal_lut. recip accessor is appended last.
        static_cast<uint32_t>(use_recip_lut),
        recip_lut_cb_id,
        // CT 19/20: broadcast affine read counts (always num_tile_cols: only TRUE broadcast
        // [1,1,H] uses the one-shot resident read).
        weight_bcast_tiles,
        bias_bcast_tiles,
        // CT 21: per-batch RoPE stride (tiles). 0 -> broadcast cos/sin across the input batch
        // (reader reindexes by within-batch seq row); >0 -> each batch offsets by this many tiles.
        rope_batch_stride_tiles,
        // CT 22/23/24: per-batch adaLN weight/bias ([batch,1,H]) — streamed per row (face-row
        // broadcast read at wbatch*num_tile_cols, wbatch = tile_row / rows_per_batch_tiles),
        // consumed + popped per row like per-token. rows_per_batch_tiles = num_tile_rows / batch.
        static_cast<uint32_t>(per_batch_weight),
        static_cast<uint32_t>(per_batch_bias),
        rows_per_batch_tiles,
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
    // Recip LUT accessor (last). Dummy=input when the LUT is unused.
    if (use_recip_lut) {
        TensorAccessorArgs(reciprocals.value().buffer()).append_to(reader_compile_args);
    } else {
        TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_args);  // dummy
    }

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/kernels/dataflow/"
        "dit_rmsnorm_fused_reader.cpp",
        worker_core_set,
        ReaderDataMovementConfig(reader_compile_args));

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
            "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/kernels/dataflow/"
            "dit_rmsnorm_fused_writer.cpp",
            worker_core_set,
            WriterDataMovementConfig(writer_compile_args));
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
            stick_bytes,  // 128 (RMS, 1 stat) or 256 (LayerNorm, mean+M2)
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
            "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/kernels/dataflow/"
            "dit_rmsnorm_fused_worker_writer.cpp",
            worker_core_set,
            WriterDataMovementConfig(writer_compile_args));
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
            stick_bytes,  // 128 (RMS) or 256 (LayerNorm: mean+M2)
            num_chunks_per_device,
            arrival_sem_id,
            go_sem_id,
        };
        TensorAccessorArgs(stats_dram_buffer).append_to(fwd_ct);
        forwarder_kernel_ids[f] = CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/kernels/dataflow/"
            "dit_rmsnorm_fused_forwarder.cpp",
            CoreRangeSet({CoreRange(forwarder_cores[f], forwarder_cores[f])}),
            WriterDataMovementConfig(fwd_ct));
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
        // Packed AG CBs (all-gather path). For is_tp_1 the compute kernel
        // sidesteps the packed path entirely (pushes col-0 stats straight into
        // stats_gathered_cb).
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
        static_cast<uint32_t>(fuse_mm_rope),      // block-major POST: fuse matmul+rope per block (rotated block-local)
        static_cast<uint32_t>(block_major_post),  // full block-major POST (all sub-phases per block; wide low-TP)
        static_cast<uint32_t>(args.norm_type),    // 0=RMS (sum-of-squares), 1=Welford LayerNorm (mean/variance)
        // CT 38/39: recip LUT (LayerNorm). When use_recip the LN kernel reads recip_lut_cb
        // as a std::array<uint32_t, reduce_width> and passes it to welford_update/finalize
        // (array load vs soft-float 1/(N+1)); else it uses the runtime-division fallback.
        recip_lut_cb_id,
        static_cast<uint32_t>(use_recip_lut),
        // CT 40: zeroed welford-state CB (LayerNorm warm-row accumulator reset).
        welford_zero_cb_id,
        // CT 41/42/43: per-batch adaLN. per_batch_weight/bias tell the compute to consume + POP
        // weight_cb/bias_cb per row (the reader streams each row's batch slice, face-row broadcast)
        // using mul_bcast_rows — same per-row consumption as per-token, broadcast op.
        // rows_per_batch_tiles is unused by compute now (the reader owns batch indexing); kept for
        // arg stability.
        static_cast<uint32_t>(per_batch_weight),
        static_cast<uint32_t>(per_batch_bias),
        rows_per_batch_tiles,
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
        "dit_fused_distributed_rmsnorm requires fp32_dest_acc_en=true in the compute kernel config "
        "(internals are always fp32); got fp32_dest_acc_en=false.");

    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/experimental/ccl/dit_fused_distributed_rmsnorm/device/kernels/compute/" +
        std::string(is_layernorm ? "dit_layernorm_fused_compute.cpp" : "dit_rmsnorm_fused_compute.cpp");
    KernelHandle compute_kernel_id = CreateKernel(
        program,
        compute_kernel_path,
        worker_core_set,
        ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_compile_args,
        });

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
    const uint32_t recip_addr_rt = use_recip_lut ? reciprocals.value().buffer()->address() : 0u;
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
            input_addr,
            weight_addr,
            bias_addr,
            rope_cos_addr,
            rope_sin_addr,
            tile_row_start,
            tile_row_end,
            recip_addr_rt};  // RT 7: recip LUT DRAM addr (0 when unused; refreshed on cache hit)
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

        // RT arg 1 (tile_row_start): the worker's first GLOBAL tile-row, so LN compute can map
        // each local row to its batch (global_row / rows_per_batch_tiles) for per-batch adaLN.
        std::vector<uint32_t> compute_rt_args = {this_core_rows, tile_row_start};
        SetRuntimeArgs(program, compute_kernel_id, core, compute_rt_args);
    }

    // ------------------------------------------------------------------------
    // Per-forwarder runtime args: stats_dram, out_ready GlobalSemaphore, the
    // group's worker NoC coords, present_count[r], then fwd+bwd fabric-connection
    // args on this forwarder's routing plane (link_idx = f).
    //
    // Guarded on num_forwarders>0 (== use_mux == !is_tp_1): the TP=1 / per_head_norm
    // path has no forwarders and no all-gather, and may run on a single device with
    // fabric NOT initialized. get_fabric_node_id() reaches into the fabric context
    // (null when fabric is down), so it must not be called on the no-fabric path.
    // ------------------------------------------------------------------------
    if (num_forwarders > 0) {
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
    }

    return {
        std::move(program),
        DitFusedDistributedRmsnormSharedVariables{
            .reader_kernel_ids = {reader_kernel_id},
            .writer_kernel_ids = {writer_kernel_id},
            .compute_kernel_ids = {compute_kernel_id},
            .forwarder_kernel_ids = forwarder_kernel_ids,
            .forwarder_cores = forwarder_cores,
            .cores = worker_cores,
            .stats_dram_addr_writer_arg_idx = stats_dram_addr_writer_arg_idx,
        }};
}

DitFusedDistributedRmsnormMeshWorkloadFactory::cached_mesh_workload_t
DitFusedDistributedRmsnormMeshWorkloadFactory::create_mesh_workload(
    const DitFusedDistributedRmsnormParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const DitFusedDistributedRmsnormInputs& tensor_args,
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

void DitFusedDistributedRmsnormMeshWorkloadFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    const DitFusedDistributedRmsnormParams& operation_attributes,
    const DitFusedDistributedRmsnormInputs& tensor_args,
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
    // The worker writer reads it from a fixed runtime-arg slot whose host-side
    // index is captured in shared.stats_dram_addr_writer_arg_idx (set at
    // create_at time, only on the all-gather path).
    const uint32_t stats_dram_addr = tensor_return_value.size() > 1 ? tensor_return_value[1].buffer()->address() : 0u;
    // Recip LUT tensor is also a regular (caller-owned) device tensor; refresh its addr.
    const uint32_t recip_addr =
        tensor_args.reciprocals.has_value() ? tensor_args.reciprocals.value().buffer()->address() : 0u;

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
            reader_args[7] = recip_addr;  // RT 7: recip LUT DRAM addr

            auto& writer_args = writer_runtime_args_by_core.at(core.x).at(core.y);
            writer_args[0] = output_addr;
            writer_args[3] = trans_mat_addr;  // worker-writer + drain-only writer: trans_mat at rt[3]
            if (shared.stats_dram_addr_writer_arg_idx.has_value()) {
                // worker-writer (AG path): stats_dram scratch at rt[4].
                writer_args[shared.stats_dram_addr_writer_arg_idx.value()] = stats_dram_addr;
            }
        }
        // Forwarders read the stats DRAM scratch base at rt[0] and the out_ready
        // GlobalSemaphore address at rt[1]. BOTH must be refreshed on cache hits:
        // the caller ping-pongs a distinct semaphore per launch and the semaphore
        // is NOT part of compute_program_hash, so a cache hit that skipped this
        // refresh would silently keep using the semaphore baked at first compile —
        // defeating the ping-pong isolation and racing the in-kernel sem reset
        // against peers' in-flight fabric atomic-incs (AG desync / hang). Mirrors
        // rms_allgather_program_factory's runtime_args[7] = semaphore.address().
        const uint32_t out_ready_sem_addr = operation_attributes.multi_device_global_semaphore.empty()
                                                ? 0u
                                                : operation_attributes.multi_device_global_semaphore.at(0).address();
        for (size_t f = 0; f < shared.forwarder_kernel_ids.size(); f++) {
            auto& fwd_args_by_core = GetRuntimeArgs(program, shared.forwarder_kernel_ids[f]);
            const auto& fc = shared.forwarder_cores[f];
            fwd_args_by_core.at(fc.x).at(fc.y)[0] = stats_dram_addr;
            fwd_args_by_core.at(fc.x).at(fc.y)[1] = out_ready_sem_addr;
        }
    }
}

}  // namespace ttnn::experimental::prim
