// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_device_operation.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

#include "kernels/indexer_score_work_split.hpp"  // banded grid mapping (rows_for_groups / cols_for_bands)
#include "ttnn/operations/ccl/ccl_common.hpp"    // get_linearized_index_from_physical_coord

namespace ttnn::operations::experimental::indexer_score {

namespace {
// Largest linearized index of q's devices along cluster_axis (0 on a single device). Single source for the
// worst-case window check (max_chunk_start) and the host-side chunk_start deduction, so a future change to
// the coord/linearization semantics can't desync the deduced base from the validated window.
uint32_t max_linearized_rank(const Tensor& q, std::optional<uint32_t> cluster_axis) {
    uint32_t max_rank = 0;
    if (q.device_storage().get_coords().size() > 1) {
        for (const auto& coord : q.device_storage().get_coords()) {
            max_rank = std::max(max_rank, ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, cluster_axis));
        }
    }
    return max_rank;
}

// Largest per-device chunk_start = base + max_rank*Sq. Used by the worst-case window check.
uint32_t max_chunk_start(const operation_attributes_t& attrs, const Tensor& q, uint32_t Sq) {
    return attrs.chunk_start_idx + max_linearized_rank(q, attrs.cluster_axis) * Sq;
}

// Miss-only checks: hash-pinned (placement, non-indexed k batch shape) so they can't differ on a hit. The
// slot/kv_len values that do differ on a hit live in validate_runtime_values.
void validate_static(const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& q = t.q;
    const auto& k = t.k;
    const auto& w = t.weights;
    // On device + allocated. q/weights interleaved; k may be interleaved OR ND-sharded DRAM (a sharded k
    // pins its bank mapping via the hashed spec).
    for (const auto& [tp, name, allow_sharded] :
         {std::tuple{&q, "q", false}, std::tuple{&k, "k", true}, std::tuple{&w, "weights", false}}) {
        TT_FATAL(tp->storage_type() == StorageType::DEVICE, "indexer_score {} must be on device", name);
        TT_FATAL(tp->buffer() != nullptr, "indexer_score {} must have an allocated buffer", name);
        if (!allow_sharded) {
            TT_FATAL(!tp->is_sharded(), "indexer_score {} must use interleaved memory", name);
        }
    }
    TT_FATAL(
        q.device() == k.device() && q.device() == w.device(), "indexer_score q, k, weights must be on the same device");

    // Non-indexed k must be single-slot [1,1,T,D] (the indexed slot < B check is runtime -> below).
    if (!attrs.cache_batch_idx.has_value()) {
        const uint32_t kB = k.logical_shape()[0];
        TT_FATAL(kB == 1, "indexer_score k batch must be 1 unless cache_batch_idx is set (got {})", kB);
    }
}

// The slot/kv_len values: hashed only by has_value(), so they can differ on a hit. Run on miss AND hit.
void validate_runtime_values(const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& k = t.k;

    // Indexed KV cache: cache_batch_idx picks a slot of [B,1,T,D] k; an out-of-range slot offsets pages OOB.
    if (attrs.cache_batch_idx.has_value()) {
        const uint32_t kB = k.logical_shape()[0];
        TT_FATAL(
            attrs.cache_batch_idx.value() < kB,
            "indexer_score cache_batch_idx ({}) must be < k batch slots ({})",
            attrs.cache_batch_idx.value(),
            kB);
    }

    // Runtime KV length: kv_len <= T is the valid prefix this dispatch (not hashed -> re-checked here). The
    // chunk-window-vs-kv_len bound lives in validate_chunk_start.
    if (attrs.kv_len.has_value()) {
        const uint32_t T = k.logical_shape()[2];
        const uint32_t kv_len = attrs.kv_len.value();
        // kv_len need NOT be tile-aligned: a sub-tile valid length (real ISL not a multiple of 32) is masked
        // per-column in the last valid tile (cols [kv_len%32, 32) -> -inf), mirroring ring_mla's
        // global_n_partial_col. Only (0, T] is required.
        TT_FATAL(
            kv_len > 0 && kv_len <= T,
            "indexer_score kv_len {} must be in (0, T={}] (the allocated k length)",
            kv_len,
            T);
    }
}

// Runs on miss AND hit (chunk_start is hash-excluded). Base alignment + the fullest device's window vs T.
// NOTE: the window is intentionally NOT bounded by kv_len. With a partially-filled (padded) chunk -- the real
// ISL is smaller than the fixed chunk, so kv_len < the chunk extent -- the padded-query devices legitimately
// have causal windows that reach past the valid prefix. That is correct, not an error: keys at positions
// >= kv_len are masked (kv_len caps each cell's valid columns) and the padded queries' output is discarded.
// So kv_len purely masks keys; only the allocated-buffer bound (window <= T) is enforced here. This mirrors
// ring_mla, whose valid region is the contiguous prefix [0, kv_len) and whose pad is dropped by masking.
void validate_chunk_start(const operation_attributes_t& attrs, const tensor_args_t& t) {
    const uint32_t Sq = t.q.logical_shape()[2];
    const uint32_t T = t.k.logical_shape()[2];
    TT_FATAL(
        attrs.chunk_start_idx % tt::constants::TILE_WIDTH == 0,
        "chunk_start_idx {} must be tile-aligned",
        attrs.chunk_start_idx);
    const uint32_t max_cs = max_chunk_start(attrs, t.q, Sq);
    TT_FATAL(
        max_cs + Sq <= T,
        "fullest-device chunk window [{}, {}+{}) exceeds T={} (base={}, per-rank stride Sq={})",
        max_cs,
        max_cs,
        Sq,
        T,
        attrs.chunk_start_idx,
        Sq);
}
// Slab layout (slab_sp / slab_chunk_size, passed by the caller): ring_size/chunk are hashed and bake invP's
// divisors into the reader, so the per-shard slab must be tile-aligned and tile the cache evenly -- miss-only
// (a hit can't differ). Sq <= cl bounds a device's queries to at most one cache-slab crossing (one straddle).
void validate_slab(const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (!attrs.slab.has_value()) {
        return;
    }
    const uint32_t ring = attrs.slab->ring_size;
    const uint32_t chunk = attrs.slab->chunk_size;
    const uint32_t T = t.k.logical_shape()[2];
    const uint32_t Sq = t.q.logical_shape()[2];
    TT_FATAL(ring >= 1, "slab ring_size (slab_sp) must be >= 1 (got {})", ring);
    TT_FATAL(chunk > 0 && chunk % ring == 0, "slab chunk_size {} must be > 0 and divisible by slab_sp {}", chunk, ring);
    const uint32_t chunk_local = chunk / ring;
    TT_FATAL(
        chunk_local % tt::constants::TILE_WIDTH == 0,
        "slab per-shard width chunk_size/slab_sp ({}) must be tile-aligned",
        chunk_local);
    TT_FATAL(
        T % chunk == 0,
        "slab chunk_size {} must divide T {} (the cache must be a whole number of global chunks)",
        chunk,
        T);
    TT_FATAL(
        Sq <= chunk_local,
        "Sq ({}) must be <= the slab per-shard width chunk_size/slab_sp ({}); a device's queries may cross at "
        "most one cache-slab boundary (one straddle). A coarser indexer SP (Sq > cl) is not yet supported.",
        Sq,
        chunk_local);
}
}  // namespace

IndexerScoreDeviceOperation::program_factory_t IndexerScoreDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::IndexerScoreProgramFactory{};
}

ttsl::hash::hash_t IndexerScoreDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Hash what shapes the binary, NOT the runtime values: chunk_start_idx is EXCLUDED, cache_batch_idx /
    // kv_len contribute only has_value() (so distinct slot / kv_len / chunk_start reuse one program).
    // cluster_axis IS hashed; tensor_args cover dtype + shape. apply_relu / num_groups / block_size pick the
    // compile-time kernel path, so they MUST be hashed (else DSA vs MSA, or pooled vs unpooled, would collide).
    return tt::tt_metal::operation::hash_operation<IndexerScoreDeviceOperation>(
        attrs.apply_relu,
        attrs.num_groups,
        attrs.block_size,
        attrs.synthesize_gate,  // gate read from DRAM vs filled in-kernel -> different reader binary
        attrs.gate_scale,       // the in-kernel fill value; distinct scales get distinct programs
        attrs.program_config,
        attrs.compute_kernel_config,
        attrs.cluster_axis.has_value(),
        attrs.cluster_axis.value_or(0u),
        attrs.has_indexed_kv_cache(),
        attrs.has_runtime_kv_len(),
        // The slab layout bakes invP divisors into the reader as compile-time defines, so ring_size/chunk
        // must be hashed (a contiguous vs slab read, or a different slab shape, is a different binary).
        attrs.has_slab(),
        attrs.slab.has_value() ? attrs.slab->ring_size : 0u,
        attrs.slab.has_value() ? attrs.slab->chunk_size : 0u,
        tensor_args);
}

void IndexerScoreDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // chunk_start, cache slot, kv_len are hash-excluded runtime values -> re-checked on hits.
    validate_runtime_values(attrs, tensor_args);
    validate_chunk_start(attrs, tensor_args);
}

void IndexerScoreDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const auto& w = tensor_args.weights;

    const auto& q_shape = q.logical_shape();
    const auto& k_shape = k.logical_shape();
    const auto& w_shape = w.logical_shape();

    // Blackhole-only: the compute kernel relies on BH fast-untilize + custom BH LLK paths. Enforce in C++
    // so a Wormhole caller fails cleanly instead of hanging at launch.
    const tt::ARCH arch = tt::tt_metal::hal::get_arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "indexer_score is only supported on Blackhole, got {}", arch);

    // The custom blocked bcast-col LLK + half-sync 8-head subblock are validated only for bf16 DEST in
    // half-sync; reject the unvalidated knobs loudly (a silent path could drop the -inf mask).
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(arch, attrs.compute_kernel_config);
    TT_FATAL(
        !fp32_dest_acc_en,
        "indexer_score requires fp32_dest_acc_en=false (bf16 DEST; the custom LLK is not validated for fp32 DEST)");
    TT_FATAL(!dst_full_sync_en, "indexer_score requires dst_full_sync_en=false (the kernel is built half-sync)");

    // Placement/layout/same-device and the non-indexed k batch shape are hash-pinned (miss only); the
    // slot/kv_len runtime values are re-checked every dispatch.
    validate_static(attrs, tensor_args);
    validate_runtime_values(attrs, tensor_args);
    validate_slab(attrs, tensor_args);

    // Shapes: q [B, Hi, Sq, D], k [B, 1, T, D] (single shared head), weights [B, Hi, Sq, 1].
    TT_FATAL(q_shape.rank() == 4 && k_shape.rank() == 4, "q, k must be rank 4");
    TT_FATAL(k_shape[1] == 1, "k must be single-head [B, 1, T, D], got {} heads", k_shape[1]);
    TT_FATAL(q_shape[3] == k_shape[3], "q head dim {} != k head dim {}", q_shape[3], k_shape[3]);
    TT_FATAL(q_shape[0] == 1, "q batch 1 only, got {}", q_shape[0]);
    // The learned-gate weights are validated only when present; MSA synthesizes a constant gate in-kernel
    // (the weights handle is an unused q placeholder), so its shape/dtype carry no meaning.
    if (!attrs.synthesize_gate) {
        TT_FATAL(w_shape.rank() == 4, "weights must be rank 4");
        TT_FATAL(
            w_shape[1] == q_shape[1] && w_shape[2] == q_shape[2] && w_shape[3] == 1,
            "weights must be [B, Hi, Sq, 1] matching q [B, Hi, Sq, D]");
        TT_FATAL(q_shape[0] == w_shape[0], "q/weights batch mismatch ({} vs {})", q_shape[0], w_shape[0]);
        TT_FATAL(w.dtype() == DataType::BFLOAT16, "weights must be bfloat16 (got {})", w.dtype());
        TT_FATAL(w.layout() == Layout::TILE, "weights must be TILE layout");
    }

    // q/k are matmul inputs (never packed), so each may be bfp8_b (halves BW); both bfp8 -> LoFi, any bf16
    // -> HiFi2.
    TT_FATAL(
        q.dtype() == DataType::BFLOAT16 || q.dtype() == DataType::BFLOAT8_B,
        "q must be bfloat16 or bfloat8_b (got {})",
        q.dtype());
    TT_FATAL(
        k.dtype() == DataType::BFLOAT16 || k.dtype() == DataType::BFLOAT8_B,
        "k must be bfloat16 or bfloat8_b (got {})",
        k.dtype());
    TT_FATAL(q.layout() == Layout::TILE && k.layout() == Layout::TILE, "q, k must be TILE layout");

    // Tile alignment and the causal chunk window.
    const uint32_t Hi = q_shape[1];
    const uint32_t Sq = q_shape[2];
    const uint32_t D = q_shape[3];
    const uint32_t T = k_shape[2];
    TT_FATAL(
        Sq % tt::constants::TILE_HEIGHT == 0 && T % tt::constants::TILE_WIDTH == 0 &&
            D % tt::constants::TILE_WIDTH == 0,
        "Sq {}, T {}, D {} must be tile-aligned",
        Sq,
        T,
        D);
    validate_chunk_start(attrs, tensor_args);  // base/stride alignment + worst-device window

    // Work-unit knobs (elements, tile-aligned); see IndexerScoreProgramConfig.
    const auto& cfg = attrs.program_config;
    TT_FATAL(
        cfg.q_chunk_size % tt::constants::TILE_HEIGHT == 0 && cfg.k_chunk_size % tt::constants::TILE_WIDTH == 0,
        "q_chunk_size {} / k_chunk_size {} must be tile-aligned",
        cfg.q_chunk_size,
        cfg.k_chunk_size);
    const uint32_t QC = cfg.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);
    TT_FATAL(
        QC > 0 && (Sq / tt::constants::TILE_HEIGHT) % QC == 0,
        "q_chunk_size {} must divide Sq {}",
        cfg.q_chunk_size,
        Sq);
    TT_FATAL(KC > 0 && KC <= T / tt::constants::TILE_WIDTH, "k_chunk_size {} out of range (T={})", cfg.k_chunk_size, T);
    // KC need not divide Tt: the last unit is then partial (compute runs a full KC strip, writer clips).
    TT_FATAL(HB > 0 && Hi % HB == 0, "head_group_size {} must divide Hi {}", HB, Hi);

    // num_groups: 1 sums all heads (DSA/GLM); G>1 emits G per-group planes (M3). G>1 reuses the all-resident
    // full-strip path one group at a time (the streaming/per-column fallback is not wired for groups).
    const uint32_t G = attrs.num_groups;
    TT_FATAL(G >= 1 && Hi % G == 0, "num_groups {} must be >= 1 and divide Hi {}", G, Hi);
    if (G > 1) {
        TT_FATAL(HB == Hi, "num_groups {}>1 requires all heads resident (head_group_size 0 or Hi); got HB={}", G, HB);
        TT_FATAL(
            KC >= 2,
            "num_groups {}>1 requires k_chunk_size>=64 (the full-strip path); got k_chunk_size={}",
            G,
            cfg.k_chunk_size);
    }

    // block_size: 0 = no pooling; >0 = block-max-pool -> [B,G,Sq,T/block_size] (M3). Guards fire only when
    // pooling. A block must align to T and the k-chunk boundary, and blocks-per-unit must fit TILE_HEIGHT.
    const uint32_t bs = attrs.block_size;
    if (bs > 0) {
        TT_FATAL(
            bs % tt::constants::TILE_WIDTH == 0,
            "block_size {} must be a multiple of {}",
            bs,
            tt::constants::TILE_WIDTH);
        TT_FATAL(T % bs == 0, "block_size {} must divide T {}", bs, T);
        TT_FATAL(
            cfg.k_chunk_size % bs == 0,
            "block_size {} must divide k_chunk_size {} (a block can't straddle a work unit)",
            bs,
            cfg.k_chunk_size);
        // No partial last unit: every unit emits exactly blocks_per_unit aligned block-scores.
        const uint32_t Tt = T / tt::constants::TILE_WIDTH;
        TT_FATAL(Tt % KC == 0, "block_size pooling requires k_chunk_size {} to divide T {}", cfg.k_chunk_size, T);
        const uint32_t block_tiles = bs / tt::constants::TILE_WIDTH;
        const uint32_t blocks_per_unit = KC / block_tiles;
        // The writer writes each unit's bf16 output slice at offset unit*blocks_per_unit*2; Blackhole's 16 B
        // write/source alignment requires blocks_per_unit a multiple of 8. Net: k_chunk_size/block_size in
        // {8,16,24,32}.
        constexpr uint32_t kBf16AlignBlocks = 8;  // 16 B / 2 B
        TT_FATAL(
            blocks_per_unit % kBf16AlignBlocks == 0,
            "block_size pooling needs k_chunk_size/block_size (= {}) to be a multiple of {} so each unit's "
            "row-major output slice is 16 B-aligned; set k_chunk_size to a multiple of {}*block_size",
            blocks_per_unit,
            kBf16AlignBlocks,
            kBf16AlignBlocks);
        TT_FATAL(
            blocks_per_unit <= tt::constants::TILE_HEIGHT,
            "blocks per work unit (k_chunk_size/block_size = {}) must be <= {}; reduce k_chunk_size",
            blocks_per_unit,
            tt::constants::TILE_HEIGHT);
    }
}

IndexerScoreDeviceOperation::spec_return_value_t IndexerScoreDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const auto& q_shape = tensor_args.q.logical_shape();
    const auto& k_shape = tensor_args.k.logical_shape();
    // score [B, num_groups, Sq, T_out], row-major bf16. num_groups==1 = head-summed (DSA/GLM); >1 = one
    // plane per GQA group (M3). T_out = T, or T/block_size when block-max-pooling.
    const uint32_t T_out = attrs.block_size ? k_shape[2] / attrs.block_size : k_shape[2];
    ttnn::Shape out_shape({q_shape[0], attrs.num_groups, q_shape[2], T_out});
    return TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), tensor_args.q.memory_config()));
}

IndexerScoreDeviceOperation::tensor_return_value_t IndexerScoreDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

// Matmul-FLOP performance model: report ideal matmul cycles = num_mul_adds / (cores * peak), so the
// profiler's ideal/actual ratio equals the math-util test's mm_flops / (cores * device_cycles * peak).
// Every term matches the test: causal-valid output tiles only, actual cores, fidelity-scaled BH peak.
tt::tt_metal::operation::OpPerformanceModelGeneral<IndexerScoreDeviceOperation::tensor_return_value_t>
IndexerScoreDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const tt::tt_metal::operation::Tensors input_tensors = {q, k, tensor_args.weights};

    // Matmul throughput model is Blackhole-specific (the only validated arch).
    const tt::ARCH arch = q.device()->arch();
    if (arch != tt::ARCH::BLACKHOLE) {
        return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(input_tensors, output, 0);
    }

    const auto& q_shape = q.logical_shape();
    const auto& k_shape = k.logical_shape();
    const uint32_t B = q_shape[0];
    const uint32_t Hi = q_shape[1];
    const uint32_t D = q_shape[3];
    const uint32_t Sqt = q_shape[2] / tt::constants::TILE_HEIGHT;
    const uint32_t Tt = k_shape[2] / tt::constants::TILE_WIDTH;
    const uint32_t chunk_t = attrs.chunk_start_idx / tt::constants::TILE_WIDTH;

    // Causal-valid output tiles V = sum_rows min(kv_len_tiles, chunk_t + row + 1) (masked future excluded;
    // matches the test's sp7_valid_tiles()). kv_len caps per-row valid columns; nullopt == full Tt.
    const uint32_t kv_len_tiles = attrs.kv_len.has_value() ? attrs.kv_len.value() / tt::constants::TILE_WIDTH : Tt;
    uint64_t valid_tiles = 0;
    for (uint32_t s = 0; s < Sqt; ++s) {
        valid_tiles += std::min<uint64_t>(kv_len_tiles, (uint64_t)chunk_t + s + 1);
    }

    // Per valid 32x32 tile per head: (32*32) outputs x 2*D FLOPs; summed over heads/tiles/batch
    // (matches the test's indexer_mm_flops()).
    const uint64_t num_mul_adds =
        2ull * valid_tiles * Hi * B * static_cast<uint64_t>(tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH) * D;

    // Cores used: the banded schedule's rows_for_groups x cols_for_bands rectangle (shares the factory's
    // mapping, so this equals tracy's CORE COUNT).
    const uint32_t QC = attrs.program_config.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = attrs.program_config.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t group_count = Sqt / QC;
    const uint32_t band_count = units_in_group(KC, Tt);  // ceil(Tt/KC); shared with the factory
    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint64_t num_cores =
        static_cast<uint64_t>(rows_for_groups(group_count, grid.y)) * cols_for_bands(band_count, grid.x);

    // Blackhole matmul peak: 4096 mul-adds/cycle/core at LoFi, scaled by the fidelity multiplier.
    const auto math_fidelity = std::get<0>(ttnn::get_compute_kernel_config_args(arch, attrs.compute_kernel_config));
    constexpr uint64_t tensix_mul_adds_per_cycle_lofi = 4096;
    const int ideal_compute_cycles = static_cast<int>(std::ceil(
        (static_cast<double>(num_mul_adds) / static_cast<double>(num_cores * tensix_mul_adds_per_cycle_lofi)) *
        static_cast<double>(tt::tt_metal::operation::OpPerformanceModel::fidelity_multiplier(math_fidelity))));

    return tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t>(
        input_tensors, output, ideal_compute_cycles);
}

std::tuple<IndexerScoreDeviceOperation::operation_attributes_t, IndexerScoreDeviceOperation::tensor_args_t>
IndexerScoreDeviceOperation::invoke(
    const Tensor& q,
    const Tensor& k,
    const Tensor& weights,
    uint32_t chunk_start_idx,
    bool apply_relu,
    uint32_t num_groups,
    uint32_t block_size,
    bool synthesize_gate,
    float gate_scale,
    const IndexerScoreProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    std::optional<uint32_t> cluster_axis,
    std::optional<SlabLayout> slab,
    bool mid_slab_boundary) {
    return {
        operation_attributes_t{
            .chunk_start_idx = chunk_start_idx,
            .cluster_axis = cluster_axis,
            .apply_relu = apply_relu,
            .num_groups = num_groups,
            .block_size = block_size,
            .synthesize_gate = synthesize_gate,
            .gate_scale = gate_scale,
            .program_config = program_config,
            .compute_kernel_config = compute_kernel_config,
            .cache_batch_idx = cache_batch_idx,
            .kv_len = kv_len,
            .slab = slab,
            .mid_slab_boundary = mid_slab_boundary},
        tensor_args_t{.q = q, .k = k, .weights = weights}};
}

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

namespace {

// Shared launch path for both frontends: resolve the compute-kernel config from the matmul-input dtypes,
// then pack and launch the one device op. The flavour knobs are decided by the caller below.
ttnn::Tensor launch_indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx,
    bool apply_relu,
    uint32_t num_groups,
    uint32_t block_size,
    bool synthesize_gate,
    float gate_scale,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> slab_sp,
    std::optional<uint32_t> slab_chunk_size,
    bool mid_slab_boundary) {
    using OperationType = ttnn::operations::experimental::indexer_score::IndexerScoreDeviceOperation;
    using ttnn::operations::experimental::indexer_score::SlabLayout;

    const uint32_t Sq = q.logical_shape()[2];

    // The K-cache slab layout is PASSED, not derived: slab_sp = the SP the cache was gathered across (the
    // cache's own SP, independent of how THIS op splits Q) and slab_chunk_size = the global chunk granularity.
    // Both must be given together; slab_sp <= 1 (or unset) means contiguous K (no remap).
    TT_FATAL(
        slab_sp.has_value() == slab_chunk_size.has_value(),
        "indexer_score: slab_sp and slab_chunk_size must be passed together (got slab_sp={}, slab_chunk_size={})",
        slab_sp.has_value(),
        slab_chunk_size.has_value());
    const std::optional<SlabLayout> slab =
        (slab_sp.has_value() && *slab_sp > 1)
            ? std::optional<SlabLayout>{SlabLayout{.ring_size = *slab_sp, .chunk_size = *slab_chunk_size}}
            : std::nullopt;

    // base = the absolute chunk_start of this op's rank 0. Omit it -> deduce: with a slab the gathered chunk is
    // slab_chunk_size, so base = T - slab_chunk_size; contiguous -> base = T - Sq (single device). The deduced
    // window ends at T (incompatible with a growing kv_len < T -- pass chunk_start_idx there).
    uint32_t base = 0;
    if (chunk_start_idx.has_value()) {
        base = *chunk_start_idx;
    } else {
        const uint32_t T = k.logical_shape()[2];
        const uint32_t window = slab.has_value() ? slab->chunk_size : Sq;
        TT_FATAL(
            T >= window,
            "indexer_score: cannot deduce chunk_start_idx -- T={} < {}={}. Pass chunk_start_idx explicitly if K "
            "does not equal history + the gathered chunk.",
            T,
            slab.has_value() ? "slab_chunk_size" : "Sq",
            window);
        base = T - window;
    }

    // Default math_fidelity follows the matmul-input dtypes (both bfp8 -> LoFi, else HiFi2); a caller config
    // overrides per field. fp32-dest acc / full-sync default off (the only modes the custom LLK is validated for).
    const bool both_bfp8 = q.dtype() == DataType::BFLOAT8_B && k.dtype() == DataType::BFLOAT8_B;
    const auto resolved = ttnn::init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/both_bfp8 ? tt::tt_metal::MathFidelity::LoFi : tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false);
    auto [operation_attributes, tensor_args] = OperationType::invoke(
        q,
        k,
        weights,
        base,
        apply_relu,
        num_groups,
        block_size,
        synthesize_gate,
        gate_scale,
        program_config,
        resolved,
        cache_batch_idx,
        kv_len,
        cluster_axis,
        slab,
        mid_slab_boundary);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace

ttnn::Tensor indexer_score_dsa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> slab_sp,
    std::optional<uint32_t> slab_chunk_size,
    bool mid_slab_boundary) {
    // DSA/GLM: relu, learned per-head gates, one head-summed plane, no pooling. Reads its real weights tensor.
    return launch_indexer_score(
        q,
        k,
        weights,
        chunk_start_idx,
        /*apply_relu=*/true,
        /*num_groups=*/1,
        /*block_size=*/0,
        /*synthesize_gate=*/false,
        /*gate_scale=*/1.0f,
        program_config,
        compute_kernel_config,
        cache_batch_idx,
        kv_len,
        cluster_axis,
        slab_sp,
        slab_chunk_size,
        mid_slab_boundary);
}

ttnn::Tensor indexer_score_msa(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    uint32_t num_groups,
    std::optional<uint32_t> chunk_start_idx,
    float scale,
    uint32_t block_size,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<uint32_t> cluster_axis,
    std::optional<uint32_t> slab_sp,
    std::optional<uint32_t> slab_chunk_size,
    bool mid_slab_boundary) {
    // M3 has no learned gates, only a 1/sqrt(d) scale. Rather than materialize a constant [B,Hi,Sq,1] gate
    // tensor (an extra fill op dispatched every call), the reader fills cb_w with `scale` in L1 in-kernel
    // (synthesize_gate); q is passed as the unused weights placeholder so the op infra still has a valid
    // on-device tensor. MSA fixes apply_relu=false; num_groups/block_size are selection knobs.
    return launch_indexer_score(
        q,
        k,
        /*weights=*/q,
        chunk_start_idx,
        /*apply_relu=*/false,
        num_groups,
        block_size,
        /*synthesize_gate=*/true,
        /*gate_scale=*/scale,
        program_config,
        compute_kernel_config,
        /*cache_batch_idx=*/std::nullopt,
        /*kv_len=*/std::nullopt,
        cluster_axis,
        slab_sp,
        slab_chunk_size,
        mid_slab_boundary);
}

}  // namespace ttnn::experimental
