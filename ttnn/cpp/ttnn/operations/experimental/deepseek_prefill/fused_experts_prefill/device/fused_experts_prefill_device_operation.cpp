// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fused_experts_prefill_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill {

FusedExpertsPrefillDeviceOperation::program_factory_t FusedExpertsPrefillDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MultiCore{};
}

void FusedExpertsPrefillDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    constexpr uint32_t kTile = tt::constants::TILE_HEIGHT;
    const auto& x = tensor_args.x_tok;

    TT_FATAL(x.storage_type() == StorageType::DEVICE, "fused_experts_prefill: x_tok must be on device");
    TT_FATAL(
        x.layout() == tt::tt_metal::Layout::ROW_MAJOR,
        "fused_experts_prefill: x_tok must be ROW_MAJOR (token rows are gathered and tilized on device), got {}",
        x.layout());
    TT_FATAL(
        x.dtype() == tt::tt_metal::DataType::BFLOAT16,
        "fused_experts_prefill: x_tok must be BFLOAT16, got {}",
        x.dtype());
    TT_FATAL(
        x.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM && !x.memory_config().is_sharded(),
        "fused_experts_prefill: x_tok must be DRAM interleaved (rows are read by page id)");
    TT_FATAL(
        x.logical_shape().rank() == 4 && x.logical_shape()[0] == 1 && x.logical_shape()[1] == 1,
        "fused_experts_prefill: x_tok must be [1, 1, T, H], got {}",
        x.logical_shape());
    const uint32_t hidden = static_cast<uint32_t>(x.logical_shape()[-1]);
    const uint32_t tokens = static_cast<uint32_t>(x.logical_shape()[-2]);
    TT_FATAL(
        hidden % (kTile * kCoresPerGroup * 2) == 0 && (hidden / kTile) % kRmChunkTiles == 0 &&
            hidden / kTile >= 2 * kRmChunkTiles,
        "fused_experts_prefill: H ({}) must be a multiple of {} and at least {}",
        hidden,
        kTile * kCoresPerGroup * 2,
        2 * kRmChunkTiles * kTile);
    TT_FATAL(
        tokens % kTile == 0 && tokens <= kMaxTokens,
        "fused_experts_prefill: T ({}) must be a multiple of {} and at most {}",
        tokens,
        kTile,
        kMaxTokens);

    TT_FATAL(
        attributes.top_k > 0 && attributes.top_k <= kMaxTopK,
        "fused_experts_prefill: top_k ({}) must be in [1, {}]",
        attributes.top_k,
        kMaxTopK);

    const size_t num_experts = tensor_args.gate_up_weights.size();
    TT_FATAL(num_experts > 0, "fused_experts_prefill: need at least one expert");
    TT_FATAL(
        tensor_args.down_weights.size() == num_experts,
        "fused_experts_prefill: gate_up_weights ({}) and down_weights ({}) must have the same length",
        num_experts,
        tensor_args.down_weights.size());

    // Routing inputs: TILE bf16 score rows; the selection comes from exactly one of ids / ranking.
    const auto check_score_tensor = [&](const Tensor& t, const char* name) {
        TT_FATAL(t.storage_type() == StorageType::DEVICE, "fused_experts_prefill: {} must be on device", name);
        TT_FATAL(
            t.layout() == tt::tt_metal::Layout::TILE && t.dtype() == tt::tt_metal::DataType::BFLOAT16 &&
                t.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM && !t.memory_config().is_sharded(),
            "fused_experts_prefill: {} must be a TILE BFLOAT16 DRAM interleaved tensor",
            name);
        TT_FATAL(
            t.logical_shape().rank() == 4 && static_cast<uint32_t>(t.logical_shape()[-2]) == tokens &&
                static_cast<uint32_t>(t.logical_shape()[-1]) == num_experts,
            "fused_experts_prefill: {} must be [1, 1, {}, {}], got {}",
            name,
            tokens,
            num_experts,
            t.logical_shape());
    };
    check_score_tensor(tensor_args.routing_scores, "routing_scores");
    TT_FATAL(
        tensor_args.routing_indices.has_value() != tensor_args.ranking_scores.has_value(),
        "fused_experts_prefill: pass exactly one of routing_indices / ranking_scores");
    if (tensor_args.ranking_scores.has_value()) {
        check_score_tensor(*tensor_args.ranking_scores, "ranking_scores");
    } else {
        const auto& ids = *tensor_args.routing_indices;
        TT_FATAL(ids.storage_type() == StorageType::DEVICE, "fused_experts_prefill: routing_indices must be on device");
        TT_FATAL(
            ids.layout() == tt::tt_metal::Layout::TILE &&
                (ids.dtype() == tt::tt_metal::DataType::UINT16 || ids.dtype() == tt::tt_metal::DataType::BFLOAT16) &&
                ids.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM && !ids.memory_config().is_sharded(),
            "fused_experts_prefill: routing_indices must be a TILE UINT16 / BFLOAT16 DRAM interleaved tensor");
        TT_FATAL(
            ids.logical_shape().rank() == 4 && static_cast<uint32_t>(ids.logical_shape()[-2]) == tokens &&
                static_cast<uint32_t>(ids.logical_shape()[-1]) == attributes.top_k,
            "fused_experts_prefill: routing_indices must be [1, 1, {}, {}], got {}",
            tokens,
            attributes.top_k,
            ids.logical_shape());
    }

    // gate_up: [H, 2I], DRAM ND-sharded [H, 64] (one [gate_32 | up_32] pair per shard, I/32 shards).
    // Each of the 8 cores of a group covers I/32/8 consecutive shards.
    const uint32_t two_i = static_cast<uint32_t>(tensor_args.gate_up_weights[0].logical_shape()[-1]);
    const uint32_t inter = two_i / 2;
    TT_FATAL(
        inter == attributes.intermediate_size,
        "fused_experts_prefill: intermediate_size ({}) does not match gate_up_weights ({})",
        attributes.intermediate_size,
        inter);
    TT_FATAL(
        inter % (kTile * kCoresPerGroup) == 0,
        "fused_experts_prefill: local intermediate size ({}) must be a multiple of {} (I-tiles split over {} cores)",
        inter,
        kTile * kCoresPerGroup,
        kCoresPerGroup);
    for (size_t e = 0; e < num_experts; ++e) {
        const auto& w = tensor_args.gate_up_weights[e];
        TT_FATAL(w.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "gate_up_weights[{}] must be in DRAM", e);
        TT_FATAL(
            static_cast<uint32_t>(w.logical_shape()[-2]) == hidden &&
                static_cast<uint32_t>(w.logical_shape()[-1]) == two_i,
            "fused_experts_prefill: gate_up_weights[{}] must be [{}, {}], got {}",
            e,
            hidden,
            two_i,
            w.logical_shape());
        const auto& nd = w.memory_config().nd_shard_spec();
        TT_FATAL(nd.has_value(), "fused_experts_prefill: gate_up_weights[{}] must be ND-sharded", e);
        TT_FATAL(
            static_cast<uint32_t>(nd->shard_shape[-1]) == 2 * kTile &&
                static_cast<uint32_t>(nd->shard_shape[-2]) == hidden,
            "fused_experts_prefill: gate_up_weights[{}] shard must be [{}, {}] (decode layout), got [{}, {}]",
            e,
            hidden,
            2 * kTile,
            nd->shard_shape[-2],
            nd->shard_shape[-1]);
    }

    // down: [I, H], DRAM ND-sharded [I, 64]; the H/64 shards are split over the 8 cores of a group.
    for (size_t e = 0; e < num_experts; ++e) {
        const auto& w = tensor_args.down_weights[e];
        TT_FATAL(w.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM, "down_weights[{}] must be in DRAM", e);
        TT_FATAL(
            static_cast<uint32_t>(w.logical_shape()[-2]) == inter &&
                static_cast<uint32_t>(w.logical_shape()[-1]) == hidden,
            "fused_experts_prefill: down_weights[{}] must be [{}, {}], got {}",
            e,
            inter,
            hidden,
            w.logical_shape());
        const auto& nd = w.memory_config().nd_shard_spec();
        TT_FATAL(nd.has_value(), "fused_experts_prefill: down_weights[{}] must be ND-sharded", e);
        TT_FATAL(
            static_cast<uint32_t>(nd->shard_shape[-1]) == 2 * kTile &&
                static_cast<uint32_t>(nd->shard_shape[-2]) == inter,
            "fused_experts_prefill: down_weights[{}] shard must be [{}, {}] (decode layout), got [{}, {}]",
            e,
            inter,
            2 * kTile,
            nd->shard_shape[-2],
            nd->shard_shape[-1]);
    }
}

void FusedExpertsPrefillDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(attributes, tensor_args);
}

tt::tt_metal::operation::Hash FusedExpertsPrefillDeviceOperation::compute_program_hash(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    std::vector<uint32_t> weight_addresses;
    weight_addresses.reserve(tensor_args.gate_up_weights.size() + tensor_args.down_weights.size());
    for (const auto& w : tensor_args.gate_up_weights) {
        weight_addresses.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    for (const auto& w : tensor_args.down_weights) {
        weight_addresses.push_back(static_cast<uint32_t>(w.buffer()->address()));
    }
    return tt::tt_metal::operation::hash_operation<FusedExpertsPrefillDeviceOperation>(
        attributes.intermediate_size,
        attributes.swiglu_limit,
        attributes.top_k,
        attributes.routed_scaling_factor,
        attributes.routing_eps,
        attributes.output_memory_config,
        tensor_args.x_tok,
        tensor_args.routing_scores,
        tensor_args.routing_indices,
        tensor_args.ranking_scores,
        tensor_args.gate_up_weights.front(),
        tensor_args.down_weights.front(),
        weight_addresses);
}

FusedExpertsPrefillDeviceOperation::spec_return_value_t FusedExpertsPrefillDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    // One row per (slot, token): [1, top_k, T, H], slot-major so slot j is a contiguous [T, H] block.
    const auto& x = tensor_args.x_tok;
    const auto& shape = x.logical_shape();
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, attributes.top_k, shape[-2], shape[-1]}),
        tt::tt_metal::TensorLayout(
            tt::tt_metal::DataType::BFLOAT16,
            tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR),
            attributes.output_memory_config));
}

FusedExpertsPrefillDeviceOperation::tensor_return_value_t FusedExpertsPrefillDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.x_tok.device());
}

std::
    tuple<FusedExpertsPrefillDeviceOperation::operation_attributes_t, FusedExpertsPrefillDeviceOperation::tensor_args_t>
    FusedExpertsPrefillDeviceOperation::invoke(
        const Tensor& x_tok,
        const Tensor& routing_scores,
        const std::vector<Tensor>& gate_up_weights,
        const std::vector<Tensor>& down_weights,
        uint32_t intermediate_size,
        float swiglu_limit,
        uint32_t top_k,
        float routed_scaling_factor,
        float routing_eps,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& routing_indices,
        const std::optional<Tensor>& ranking_scores) {
    operation_attributes_t attributes{
        .intermediate_size = intermediate_size,
        .swiglu_limit = swiglu_limit,
        .top_k = top_k,
        .routed_scaling_factor = routed_scaling_factor,
        .routing_eps = routing_eps,
        .output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG),
    };
    tensor_args_t tensor_args{
        .x_tok = x_tok,
        .routing_scores = routing_scores,
        .routing_indices = routing_indices,
        .ranking_scores = ranking_scores,
        .gate_up_weights = gate_up_weights,
        .down_weights = down_weights,
    };
    return {std::move(attributes), std::move(tensor_args)};
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill

namespace ttnn::prim {
ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill::FusedExpertsPrefillDeviceOperation::
    tensor_return_value_t
    fused_experts_prefill(
        const Tensor& x_tok,
        const Tensor& routing_scores,
        const std::vector<Tensor>& gate_up_weights,
        const std::vector<Tensor>& down_weights,
        uint32_t intermediate_size,
        float swiglu_limit,
        uint32_t top_k,
        float routed_scaling_factor,
        float routing_eps,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& routing_indices,
        const std::optional<Tensor>& ranking_scores) {
    using OperationType =
        ttnn::operations::experimental::deepseek_prefill::fused_experts_prefill::FusedExpertsPrefillDeviceOperation;
    auto [operation_attributes, tensor_args] = OperationType::invoke(
        x_tok,
        routing_scores,
        gate_up_weights,
        down_weights,
        intermediate_size,
        swiglu_limit,
        top_k,
        routed_scaling_factor,
        routing_eps,
        memory_config,
        routing_indices,
        ranking_scores);
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
