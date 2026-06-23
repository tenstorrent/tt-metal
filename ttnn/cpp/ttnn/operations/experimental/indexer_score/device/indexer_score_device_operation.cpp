// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_device_operation.hpp"

#include <algorithm>
#include <cmath>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

#include "ttnn/operations/ccl/ccl_common.hpp"  // get_linearized_index_from_physical_coord

namespace ttnn::operations::experimental::indexer_score {

namespace {
// Largest per-device chunk_start across the mesh = base + max_rank*Sq, where max_rank is the largest
// linearized index along cluster_axis (0 on a single device). Shared by the worst-case window check.
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

// chunk_start is hash-excluded (runtime), so this runs on BOTH cache miss and hit. Checks the base and
// the worst (fullest) device's window against T -- so no device can run off the end of K. The per-rank
// stride is Sq (tile-aligned by the shape check), so only the base needs an alignment check.
void validate_chunk_start(const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    const uint32_t Sq = tensor_args.q.logical_shape()[2];
    const uint32_t T = tensor_args.k.logical_shape()[2];
    TT_FATAL(
        attrs.chunk_start_idx % tt::constants::TILE_WIDTH == 0,
        "chunk_start_idx {} must be tile-aligned",
        attrs.chunk_start_idx);
    const uint32_t max_cs = max_chunk_start(attrs, tensor_args.q, Sq);
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
}  // namespace

IndexerScoreDeviceOperation::program_factory_t IndexerScoreDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::IndexerScoreProgramFactory{};
}

ttsl::hash::hash_t IndexerScoreDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    // Everything that shapes the compiled program -- but NOT chunk_start_idx / chunk_start_stride (runtime,
    // per-device). cluster_axis IS hashed: it fixes each device's stored linearized index in the cached
    // workload. q/k/w specs cover dtype (bfp8 vs bf16 changes formats) and shape (Sqt/Tt/Dt CT args).
    auto hash = tt::tt_metal::operation::hash_operation<IndexerScoreDeviceOperation>(
        attrs.program_config.q_chunk_size,
        attrs.program_config.k_chunk_size,
        attrs.program_config.head_group_size,
        attrs.compute_kernel_config,
        attrs.cluster_axis.has_value(),
        attrs.cluster_axis.value_or(0u));
    for (const auto& t : {std::cref(tensor_args.q), std::cref(tensor_args.k), std::cref(tensor_args.weights)}) {
        hash = ttsl::hash::hash_objects(
            hash, t.get().padded_shape(), t.get().layout(), t.get().dtype(), t.get().memory_config());
    }
    return hash;
}

void IndexerScoreDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
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

    // On-device, allocated, interleaved, same-device: the factory dereferences each buffer's address
    // in q's device address space and the kernels assume interleaved DRAM/L1 (mcast deal + TensorAccessor).
    for (const auto& [t, name] : {std::pair{&q, "q"}, std::pair{&k, "k"}, std::pair{&w, "weights"}}) {
        TT_FATAL(t->storage_type() == StorageType::DEVICE, "indexer_score {} must be on device", name);
        TT_FATAL(t->buffer() != nullptr, "indexer_score {} must have an allocated buffer", name);
        TT_FATAL(!t->is_sharded(), "indexer_score {} must use interleaved memory", name);
    }
    TT_FATAL(
        q.device() == k.device() && q.device() == w.device(), "indexer_score q, k, weights must be on the same device");

    // Shapes: q [B, Hi, Sq, D], k [B, 1, T, D] (single shared head), weights [B, Hi, Sq, 1].
    TT_FATAL(q_shape.rank() == 4 && k_shape.rank() == 4 && w_shape.rank() == 4, "q, k, weights must be rank 4");
    TT_FATAL(k_shape[1] == 1, "k must be single-head [B, 1, T, D], got {} heads", k_shape[1]);
    TT_FATAL(q_shape[3] == k_shape[3], "q head dim {} != k head dim {}", q_shape[3], k_shape[3]);
    TT_FATAL(
        w_shape[1] == q_shape[1] && w_shape[2] == q_shape[2] && w_shape[3] == 1,
        "weights must be [B, Hi, Sq, 1] matching q [B, Hi, Sq, D]");
    TT_FATAL(q_shape[0] == k_shape[0] && q_shape[0] == w_shape[0], "batch mismatch");
    TT_FATAL(q_shape[0] == 1, "batch 1 only, got {}", q_shape[0]);

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

    // Causal-valid output tiles V = sum over q-tile-rows of min(Tt, chunk_t + row + 1) -- the useful
    // matmul work, masked future tiles excluded (matches the test's sp7_valid_tiles()).
    uint64_t valid_tiles = 0;
    for (uint32_t s = 0; s < Sqt; ++s) {
        valid_tiles += std::min<uint64_t>(Tt, (uint64_t)chunk_t + s + 1);
    }

    // Per valid 32x32 output tile, per head: a 32x32 x D matmul = (32*32) outputs x 2*D FLOPs (2/FMA);
    // summed over heads, valid tiles, and batch (matches the test's indexer_mm_flops()).
    const uint64_t num_mul_adds =
        2ull * valid_tiles * Hi * B * static_cast<uint64_t>(tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH) * D;

    // Actual cores used: total_units = groups x ceil(Tt/KC), clamped to the grid (matches the factory),
    // so the perf model's core count equals tracy's CORE COUNT and the utilization ratio lines up.
    const uint32_t QC = attrs.program_config.q_chunk_size / tt::constants::TILE_HEIGHT;
    const uint32_t KC = attrs.program_config.k_chunk_size / tt::constants::TILE_WIDTH;
    const uint64_t total_units = static_cast<uint64_t>(Sqt / QC) * ((Tt + KC - 1) / KC);
    const auto grid = q.device()->compute_with_storage_grid_size();
    const uint64_t num_cores = std::min<uint64_t>(total_units, static_cast<uint64_t>(grid.x) * grid.y);

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
    std::optional<uint32_t> cluster_axis) {
    return {
        operation_attributes_t{
            .chunk_start_idx = chunk_start_idx,
            .cluster_axis = cluster_axis,
            .program_config = program_config,
            .compute_kernel_config = compute_kernel_config},
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
    std::optional<uint32_t> cluster_axis) {
    using OperationType = ttnn::operations::experimental::indexer_score::IndexerScoreDeviceOperation;

    // Resolve the device-0 base. Multichip: omit chunk_start_idx -> deduce from the chunked-prefill
    // invariant (K = history + the SP-gathered chunk of num_devices*Sq queries), so base = T - num_devices*Sq.
    // Single chip (device count != SP ring size, e.g. simulating one rank): the caller passes it.
    uint32_t base = 0;
    if (chunk_start_idx.has_value()) {
        base = *chunk_start_idx;
    } else {
        const uint32_t Sq = q.logical_shape()[2];
        const uint32_t T = k.logical_shape()[2];
        const uint32_t num_devices = static_cast<uint32_t>(q.device_storage().get_coords().size());
        TT_FATAL(
            T >= num_devices * Sq,
            "indexer_score: cannot deduce chunk_start_idx -- T={} < num_devices({})*Sq({}). Pass chunk_start_idx "
            "explicitly if K does not equal history + the SP-gathered chunk.",
            T,
            num_devices,
            Sq);
        base = T - num_devices * Sq;
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
        OperationType::invoke(q, k, weights, base, program_config, resolved, cluster_axis);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
