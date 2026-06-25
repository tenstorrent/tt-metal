// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_device_operation.hpp"

#include <algorithm>
#include <cmath>
#include <tuple>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

#include "kernels/indexer_score_work_split.hpp"  // shared banded grid mapping (rows_for_groups / cols_for_bands)
#include "ttnn/operations/ccl/ccl_common.hpp"    // get_linearized_index_from_physical_coord

namespace ttnn::operations::experimental::indexer_score {

namespace {
// Largest per-device chunk_start across the mesh = base + max_rank*Sq, where max_rank is the largest
// linearized index along cluster_axis (0 on a single device). Used by the worst-case window check.
uint32_t max_chunk_start(const operation_attributes_t& attrs, const Tensor& q, uint32_t Sq) {
    uint32_t max_rank = 0;
    if (q.device_storage().get_coords().size() > 1) {
        for (const auto& coord : q.device_storage().get_coords()) {
            max_rank =
                std::max(max_rank, ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, attrs.cluster_axis));
        }
    }
    return attrs.chunk_start_idx + max_rank * Sq;
}

// Miss-only checks: either hash-pinned (placement, non-indexed k batch shape) so they can't differ on a
// hit, or caller errors the framework rules out anyway (device residency, allocation, same-device) and
// never checked on a hit pre-PR. The slot/kv_len values that do differ on a hit live in
// validate_runtime_values.
void validate_static(const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& q = t.q;
    const auto& k = t.k;
    const auto& w = t.weights;
    // On device + allocated. q/weights interleaved; k may be interleaved OR ND-sharded DRAM (the reader's
    // TensorAccessor handles both; a sharded k pins its bank mapping via the hashed spec).
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

    // Non-indexed k must be single-slot [1,1,T,D]. has_value() and kB are both hashed, so this only fires on
    // a miss (the indexed slot < B check is a runtime value -> validate_runtime_values).
    if (!attrs.cache_batch_idx.has_value()) {
        const uint32_t kB = k.logical_shape()[0];
        TT_FATAL(kB == 1, "indexer_score k batch must be 1 unless cache_batch_idx is set (got {})", kB);
    }
}

// The slot/kv_len values: hashed only by has_value(), so they can differ on a cache hit and feed kernel
// addressing (page offset / read width). Cheap integer checks on shape metadata, run on miss AND hit.
void validate_runtime_values(const operation_attributes_t& attrs, const tensor_args_t& t) {
    const auto& k = t.k;

    // Indexed KV cache: k is [B,1,T,D]; cache_batch_idx picks a slot (q/weights are batch 1). An
    // out-of-range slot offsets every k page id OOB.
    if (attrs.cache_batch_idx.has_value()) {
        const uint32_t kB = k.logical_shape()[0];
        TT_FATAL(
            attrs.cache_batch_idx.value() < kB,
            "indexer_score cache_batch_idx ({}) must be < k batch slots ({})",
            attrs.cache_batch_idx.value(),
            kB);
    }

    // Runtime KV length: k is a persistent buffer of (hashed) length T; kv_len <= T is the valid prefix
    // this dispatch (value not hashed -> re-checked here). The chunk-window-vs-kv_len bound lives in
    // validate_chunk_start, which keeps every chunk_start check in one place.
    if (attrs.kv_len.has_value()) {
        const uint32_t T = k.logical_shape()[2];
        const uint32_t kv_len = attrs.kv_len.value();
        TT_FATAL(kv_len % tt::constants::TILE_WIDTH == 0, "indexer_score kv_len {} must be tile-aligned", kv_len);
        TT_FATAL(
            kv_len > 0 && kv_len <= T,
            "indexer_score kv_len {} must be in (0, T={}] (the allocated k length)",
            kv_len,
            T);
    }
}

// Runs on cache miss AND hit (chunk_start is hash-excluded). All chunk_start checks in one place: base
// alignment, and the fullest device's window against T and -- when set -- kv_len. Per-rank stride Sq is
// tile-aligned by the shape check, so only the base needs an alignment check.
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
    // Causal window must also stay inside the valid prefix (kv_len <= T already checked above).
    if (attrs.kv_len.has_value()) {
        const uint32_t kv_len = attrs.kv_len.value();
        TT_FATAL(
            max_cs + Sq <= kv_len,
            "fullest-device causal window [{}, {}+{}) exceeds kv_len={} (cannot attend past the valid keys; "
            "base={}, per-rank stride Sq={})",
            max_cs,
            max_cs,
            Sq,
            kv_len,
            attrs.chunk_start_idx,
            Sq);
    }
}
}  // namespace

IndexerScoreDeviceOperation::program_factory_t IndexerScoreDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::IndexerScoreProgramFactory{};
}

ttsl::hash::hash_t IndexerScoreDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Hash what shapes the binary, NOT the runtime values: chunk_start_idx is EXCLUDED (per-device runtime),
    // cache_batch_idx / kv_len contribute only has_value() -- so distinct slot / kv_len / chunk_start reuse
    // one program. cluster_axis IS hashed (fixes each device's linearized index); tensor_args cover dtype +
    // shape. Do NOT add chunk_start_idx or the slot/kv_len values here or they will recompile.
    return tt::tt_metal::operation::hash_operation<IndexerScoreDeviceOperation>(
        attrs.program_config,
        attrs.compute_kernel_config,
        attrs.cluster_axis.has_value(),
        attrs.cluster_axis.value_or(0u),
        attrs.has_indexed_kv_cache(),
        attrs.has_runtime_kv_len(),
        tensor_args);
}

void IndexerScoreDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // chunk_start, cache slot, and kv_len are all hash-excluded runtime values, so their checks run on hits too.
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

    // Blackhole-only: the compute kernel relies on BH fast-untilize + custom BH LLK paths. Enforce in
    // C++ (not just the pytest skip) so a Wormhole caller fails cleanly instead of hanging at launch.
    const tt::ARCH arch = tt::tt_metal::hal::get_arch();
    TT_FATAL(arch == tt::ARCH::BLACKHOLE, "indexer_score is only supported on Blackhole, got {}", arch);

    // The compute kernel honors the config's math_fidelity (default: dtype-derived), but its custom
    // blocked bcast-col LLK + half-sync 8-head subblock are validated only for bf16 DEST in half-sync.
    // Reject the unvalidated knobs loudly instead of silently building a path that can drop the -inf mask.
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        ttnn::get_compute_kernel_config_args(arch, attrs.compute_kernel_config);
    TT_FATAL(
        !fp32_dest_acc_en,
        "indexer_score requires fp32_dest_acc_en=false (bf16 DEST; the custom LLK is not validated for fp32 DEST)");
    TT_FATAL(!dst_full_sync_en, "indexer_score requires dst_full_sync_en=false (the kernel is built half-sync)");

    // Placement, layout, same-device and the non-indexed k batch shape are hash-pinned (miss only). The
    // slot/kv_len runtime values are re-checked on every dispatch (here and on a cache hit).
    validate_static(attrs, tensor_args);
    validate_runtime_values(attrs, tensor_args);

    // Shapes: q [B, Hi, Sq, D], k [B, 1, T, D] (single shared head), weights [B, Hi, Sq, 1].
    TT_FATAL(q_shape.rank() == 4 && k_shape.rank() == 4 && w_shape.rank() == 4, "q, k, weights must be rank 4");
    TT_FATAL(k_shape[1] == 1, "k must be single-head [B, 1, T, D], got {} heads", k_shape[1]);
    TT_FATAL(q_shape[3] == k_shape[3], "q head dim {} != k head dim {}", q_shape[3], k_shape[3]);
    TT_FATAL(
        w_shape[1] == q_shape[1] && w_shape[2] == q_shape[2] && w_shape[3] == 1,
        "weights must be [B, Hi, Sq, 1] matching q [B, Hi, Sq, D]");
    // q/weights are always batch 1; k's batch is the cache-slot count B (checked in validate_static /
    // validate_runtime_values), so it is intentionally NOT tied to q's batch here.
    TT_FATAL(q_shape[0] == w_shape[0], "q/weights batch mismatch ({} vs {})", q_shape[0], w_shape[0]);
    TT_FATAL(q_shape[0] == 1, "q/weights batch 1 only, got {}", q_shape[0]);

    // weights are bf16 tiles. q and k are matmul inputs (q is srcB, k is srcA), neither is ever
    // packed, so each may be bfp8_b (halves its BW). q+k both bfp8 drops the matmul to LoFi (1 phase);
    // any bf16 input keeps HiFi2. Mismatched dtype/layout corrupts or hangs.
    TT_FATAL(w.dtype() == DataType::BFLOAT16, "weights must be bfloat16 (got {})", w.dtype());
    TT_FATAL(
        q.dtype() == DataType::BFLOAT16 || q.dtype() == DataType::BFLOAT8_B,
        "q must be bfloat16 or bfloat8_b (got {})",
        q.dtype());
    TT_FATAL(
        k.dtype() == DataType::BFLOAT16 || k.dtype() == DataType::BFLOAT8_B,
        "k must be bfloat16 or bfloat8_b (got {})",
        k.dtype());
    TT_FATAL(
        q.layout() == Layout::TILE && k.layout() == Layout::TILE && w.layout() == Layout::TILE,
        "q, k, weights must be TILE layout");

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
    validate_chunk_start(attrs, tensor_args);  // base/stride alignment + worst-device window (also runs on cache hit)

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
    // KC need not divide Tt: the last unit is then partial. Compute still runs a full KC-wide strip
    // (pad cols matmul stale k, overwritten with full -inf) and the writer clips to the valid width.
    TT_FATAL(HB > 0 && Hi % HB == 0, "head_group_size {} must divide Hi {}", HB, Hi);
}

IndexerScoreDeviceOperation::spec_return_value_t IndexerScoreDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& tensor_args) {
    const auto& q_shape = tensor_args.q.logical_shape();
    const auto& k_shape = tensor_args.k.logical_shape();
    // score [B, 1, Sq, T], row-major bf16 (consumed by the row-major topk)
    ttnn::Shape out_shape({q_shape[0], 1, q_shape[2], k_shape[2]});
    return TensorSpec(
        out_shape,
        tt::tt_metal::TensorLayout(
            DataType::BFLOAT16, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), tensor_args.q.memory_config()));
}

IndexerScoreDeviceOperation::tensor_return_value_t IndexerScoreDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.q.device());
}

// Matmul-FLOP performance model. Mirrors SDPAOperation::create_op_performance_model: report ideal
// matmul cycles = num_mul_adds / (cores * peak), and the profiler's ideal/actual ratio is then exactly
// the math-util test's math_util = mm_flops / (cores * device_cycles * peak). Every term is matched to
// the test (test_indexer_score_sp7_math_util): causal-valid output tiles only, actual cores, and the
// fidelity-scaled Blackhole matmul peak.
tt::tt_metal::operation::OpPerformanceModelGeneral<IndexerScoreDeviceOperation::tensor_return_value_t>
IndexerScoreDeviceOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& q = tensor_args.q;
    const auto& k = tensor_args.k;
    const tt::tt_metal::operation::Tensors input_tensors = {q, k, tensor_args.weights};

    // Matmul throughput model is Blackhole-specific (the only validated arch). q is always on device.
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

    // Causal-valid output tiles V = sum over q-tile-rows of min(kv_len_tiles, chunk_t + row + 1) -- useful
    // matmul work, masked future/unwritten tiles excluded (matches the test's sp7_valid_tiles()). kv_len
    // caps the per-row valid columns; nullopt == full Tt.
    const uint32_t kv_len_tiles = attrs.kv_len.has_value() ? attrs.kv_len.value() / tt::constants::TILE_WIDTH : Tt;
    uint64_t valid_tiles = 0;
    for (uint32_t s = 0; s < Sqt; ++s) {
        valid_tiles += std::min<uint64_t>(kv_len_tiles, (uint64_t)chunk_t + s + 1);
    }

    // Per valid 32x32 output tile, per head: a 32x32 x D matmul = (32*32) outputs x 2*D FLOPs (2/FMA);
    // summed over heads, valid tiles, and batch (matches the test's indexer_mm_flops()).
    const uint64_t num_mul_adds =
        2ull * valid_tiles * Hi * B * static_cast<uint64_t>(tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH) * D;

    // Cores used: the banded schedule's rows_for_groups x cols_for_bands rectangle. Shares the mapping
    // with the factory, so this count equals the factory's (and tracy's CORE COUNT).
    const uint32_t QC = attrs.program_config.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = attrs.program_config.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint32_t group_count = Sqt / QC;
    const uint32_t band_count = units_in_group(KC, Tt);  // ceil(Tt/KC); shared with the factory
    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint64_t num_cores =
        static_cast<uint64_t>(rows_for_groups(group_count, grid.y)) * cols_for_bands(band_count, grid.x);

    // Blackhole matmul peak: 4096 mul-adds/cycle/core at LoFi, scaled by the fidelity multiplier (the
    // test's peak table is 4096 / multiplier). Fidelity comes from the resolved compute config.
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
    const IndexerScoreProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    std::optional<uint32_t> cluster_axis) {
    return {
        operation_attributes_t{
            .chunk_start_idx = chunk_start_idx,
            .cluster_axis = cluster_axis,
            .program_config = program_config,
            .compute_kernel_config = compute_kernel_config,
            .cache_batch_idx = cache_batch_idx,
            .kv_len = kv_len},
        tensor_args_t{.q = q, .k = k, .weights = weights}};
}

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    std::optional<uint32_t> chunk_start_idx,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config,
    const std::optional<ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    std::optional<uint32_t> cache_batch_idx,
    std::optional<uint32_t> kv_len,
    std::optional<uint32_t> cluster_axis) {
    using OperationType = ttnn::operations::experimental::indexer_score::IndexerScoreDeviceOperation;

    // base = rank 0's absolute chunk_start. Multichip: omit it -> deduce base = T - sp_ring*Sq (assumes K is
    // history + the full SP-gathered chunk). Caveats: (1) deduced window ends exactly at T, so it's incompatible
    // with a growing kv_len < T -- pass chunk_start_idx explicitly there; (2) single-chip default flipped to
    // nullopt (was 0), so omitting now gives base = T - Sq, not 0.
    uint32_t base = 0;
    if (chunk_start_idx.has_value()) {
        base = *chunk_start_idx;
    } else {
        const uint32_t Sq = q.logical_shape()[2];
        const uint32_t T = k.logical_shape()[2];
        // sp_ring with the per-device rank's min-relative convention (get_linearized_index returns coord-min, so
        // sp_ring = max_rank + 1). get_topological_dimension (max+1) would over-count on a nonzero-offset sub-mesh.
        uint32_t max_rank = 0;
        for (const auto& coord : q.device_storage().get_coords()) {  // single device -> one coord, max_rank 0
            max_rank = std::max(max_rank, ttnn::ccl::get_linearized_index_from_physical_coord(q, coord, cluster_axis));
        }
        const uint32_t sp_ring = max_rank + 1;
        TT_FATAL(
            T >= sp_ring * Sq,
            "indexer_score: cannot deduce chunk_start_idx -- T={} < sp_ring({})*Sq({}). Pass chunk_start_idx "
            "explicitly if K does not equal history + the SP-gathered chunk.",
            T,
            sp_ring,
            Sq);
        base = T - sp_ring * Sq;
    }
    // Default math_fidelity follows the matmul-input dtypes (both bfp8 -> LoFi for the 2x peak, else
    // HiFi2 to keep the bf16 mantissa); a caller-supplied config overrides per field. fp32-dest acc and
    // full-sync default off -- the only modes this op's custom LLK is validated for (see validate).
    const bool both_bfp8 = q.dtype() == DataType::BFLOAT8_B && k.dtype() == DataType::BFLOAT8_B;
    const auto resolved = ttnn::init_device_compute_kernel_config(
        q.device()->arch(),
        compute_kernel_config,
        /*default_fidelity=*/both_bfp8 ? tt::tt_metal::MathFidelity::LoFi : tt::tt_metal::MathFidelity::HiFi2,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/false,
        /*default_l1_acc=*/false,
        /*default_dst_full_sync_en=*/false);
    // Reuse invoke() so attribute/tensor packing lives in one place.
    auto [operation_attributes, tensor_args] =
        OperationType::invoke(q, k, weights, base, program_config, resolved, cache_batch_idx, kv_len, cluster_axis);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
