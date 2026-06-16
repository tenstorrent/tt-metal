// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "indexer_score_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::experimental::indexer_score {

IndexerScoreDeviceOperation::program_factory_t IndexerScoreDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::IndexerScoreProgramFactory{};
}

void IndexerScoreDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;

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
        Sq % TILE_HEIGHT == 0 && T % TILE_WIDTH == 0 && D % TILE_WIDTH == 0,
        "Sq {}, T {}, D {} must be tile-aligned",
        Sq,
        T,
        D);
    TT_FATAL(attrs.chunk_start_idx % TILE_WIDTH == 0, "chunk_start_idx {} must be tile-aligned", attrs.chunk_start_idx);
    TT_FATAL(
        attrs.chunk_start_idx + Sq <= T,
        "chunk window [{}, {}+{}) exceeds T={}",
        attrs.chunk_start_idx,
        attrs.chunk_start_idx,
        Sq,
        T);

    // Work-unit knobs (elements, tile-aligned); see IndexerScoreProgramConfig.
    const auto& cfg = attrs.program_config;
    TT_FATAL(
        cfg.q_chunk_size % TILE_HEIGHT == 0 && cfg.k_chunk_size % TILE_WIDTH == 0,
        "q_chunk_size {} / k_chunk_size {} must be tile-aligned",
        cfg.q_chunk_size,
        cfg.k_chunk_size);
    const uint32_t QC = cfg.q_chunk_size / TILE_HEIGHT;
    const uint32_t KC = cfg.k_chunk_size / TILE_WIDTH;
    const uint32_t HB = resolve_head_group(cfg, Hi);
    TT_FATAL(QC > 0 && (Sq / TILE_HEIGHT) % QC == 0, "q_chunk_size {} must divide Sq {}", cfg.q_chunk_size, Sq);
    TT_FATAL(KC > 0 && KC <= T / TILE_WIDTH, "k_chunk_size {} out of range (T={})", cfg.k_chunk_size, T);
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

std::tuple<IndexerScoreDeviceOperation::operation_attributes_t, IndexerScoreDeviceOperation::tensor_args_t>
IndexerScoreDeviceOperation::invoke(
    const Tensor& q,
    const Tensor& k,
    const Tensor& weights,
    uint32_t chunk_start_idx,
    const IndexerScoreProgramConfig& program_config) {
    return {
        operation_attributes_t{.chunk_start_idx = chunk_start_idx, .program_config = program_config},
        tensor_args_t{.q = q, .k = k, .weights = weights}};
}

}  // namespace ttnn::operations::experimental::indexer_score

namespace ttnn::experimental {

ttnn::Tensor indexer_score(
    const ttnn::Tensor& q,
    const ttnn::Tensor& k,
    const ttnn::Tensor& weights,
    uint32_t chunk_start_idx,
    const ttnn::operations::experimental::indexer_score::IndexerScoreProgramConfig& program_config) {
    using OperationType = ttnn::operations::experimental::indexer_score::IndexerScoreDeviceOperation;
    // Reuse invoke() so attribute/tensor packing lives in one place.
    auto [operation_attributes, tensor_args] = OperationType::invoke(q, k, weights, chunk_start_idx, program_config);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::experimental
