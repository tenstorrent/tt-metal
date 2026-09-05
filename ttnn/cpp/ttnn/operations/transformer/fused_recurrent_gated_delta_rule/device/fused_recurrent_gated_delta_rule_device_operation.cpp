// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_gated_delta_rule_device_operation.hpp"

#include <initializer_list>

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

FusedRecurrentGatedDeltaRuleDeviceOperation::program_factory_t
FusedRecurrentGatedDeltaRuleDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return FusedRecurrentGatedDeltaRuleProgramFactory{};
}

void FusedRecurrentGatedDeltaRuleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    // Attribute contract (the kernel derives every tile count from these).
    TT_FATAL(attrs.T >= 1, "fused_recurrent_gated_delta_rule: T must be >= 1, got {}", attrs.T);
    TT_FATAL(attrs.BH >= 1, "fused_recurrent_gated_delta_rule: B*HV must be >= 1, got {}", attrs.BH);
    TT_FATAL(
        attrs.key_dim >= TILE_WIDTH && attrs.key_dim % TILE_WIDTH == 0,
        "fused_recurrent_gated_delta_rule: key_dim must be a non-zero multiple of {}, got {}",
        TILE_WIDTH,
        attrs.key_dim);
    TT_FATAL(
        attrs.val_dim >= TILE_WIDTH && attrs.val_dim % TILE_WIDTH == 0,
        "fused_recurrent_gated_delta_rule: val_dim must be a non-zero multiple of {}, got {}",
        TILE_WIDTH,
        attrs.val_dim);
    const uint32_t BHT = attrs.BH * attrs.T;
    const uint32_t K = attrs.key_dim;
    const uint32_t V = attrs.val_dim;

    // Tensor contract: fp32 TILE DRAM-interleaved, all on q's device, exact logical shapes. The
    // reader/writer index pages as [BH*T,1,K] / [BH*T,1,V] / [BH*T,1,1] / [BH,K,V]; a mismatch
    // would read past the buffer instead of failing, so every dim is checked here.
    auto* device = in.q.device();
    auto check = [&](const Tensor& t, const char* name, std::initializer_list<uint32_t> expected) {
        TT_FATAL(
            t.storage_type() == StorageType::DEVICE && t.buffer() != nullptr,
            "fused_recurrent_gated_delta_rule: {} must be allocated on device",
            name);
        TT_FATAL(t.device() == device, "fused_recurrent_gated_delta_rule: {} must be on the same device as q", name);
        TT_FATAL(
            t.buffer()->buffer_type() == BufferType::DRAM,
            "fused_recurrent_gated_delta_rule: {} must be in DRAM",
            name);
        TT_FATAL(!t.is_sharded(), "fused_recurrent_gated_delta_rule: {} must be interleaved, not sharded", name);
        TT_FATAL(t.layout() == Layout::TILE, "fused_recurrent_gated_delta_rule: {} must be TILE layout", name);
        TT_FATAL(
            t.dtype() == DataType::FLOAT32,
            "fused_recurrent_gated_delta_rule: {} must be fp32, got {}",
            name,
            t.dtype());
        const auto& shape = t.logical_shape();
        TT_FATAL(
            shape.rank() == expected.size(),
            "fused_recurrent_gated_delta_rule: {} must be rank {}, got rank {}",
            name,
            expected.size(),
            shape.rank());
        size_t i = 0;
        for (uint32_t e : expected) {
            TT_FATAL(
                static_cast<uint32_t>(shape[i]) == e,
                "fused_recurrent_gated_delta_rule: {} dim[{}] must be {}, got {}",
                name,
                i,
                e,
                shape[i]);
            ++i;
        }
    };
    check(in.q, "q", {BHT, 1u, K});
    check(in.k, "k", {BHT, 1u, K});
    check(in.v, "v", {BHT, 1u, V});
    check(in.decay, "decay", {BHT, 1u, 1u});
    check(in.beta, "beta", {BHT, 1u, 1u});
    if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state", {attrs.BH, K, V});
    }
}

FusedRecurrentGatedDeltaRuleDeviceOperation::spec_return_value_t
FusedRecurrentGatedDeltaRuleDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    const auto layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    // o: [BH*T, 1, V]  (one row per (head, token); host folds to [B,T,HV,V]).
    ttnn::Shape o_shape({attrs.BH * attrs.T, 1, attrs.val_dim});
    // state: per-token [BH*T, K, V] for verify slots, else final [BH, K, V].
    ttnn::Shape s_shape = attrs.output_per_token_state ? ttnn::Shape({attrs.BH * attrs.T, attrs.key_dim, attrs.val_dim})
                                                       : ttnn::Shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    return {tt::tt_metal::TensorSpec(o_shape, layout), tt::tt_metal::TensorSpec(s_shape, layout)};
}

FusedRecurrentGatedDeltaRuleDeviceOperation::tensor_return_value_t
FusedRecurrentGatedDeltaRuleDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    auto specs = compute_output_specs(attrs, in);
    auto* device = in.q.device();
    std::vector<Tensor> outs;
    outs.reserve(specs.size());
    for (const auto& spec : specs) {
        outs.push_back(create_device_tensor(spec, device));
    }
    return outs;
}

std::vector<Tensor> fused_recurrent_gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& decay,
    const Tensor& beta,
    const std::optional<Tensor>& initial_state,
    uint32_t T,
    bool output_final_state,
    bool output_per_token_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = FusedRecurrentGatedDeltaRuleDeviceOperation;

    const auto& q_shape = q.logical_shape();  // [BH*T, 1, K]
    const auto& v_shape = v.logical_shape();  // [BH*T, 1, V]
    TT_FATAL(T >= 1, "fused_recurrent_gated_delta_rule: T must be >= 1, got {}", T);
    TT_FATAL(
        q_shape.rank() == 3 && v_shape.rank() == 3,
        "fused_recurrent_gated_delta_rule: q and v must be rank 3, got ranks {} and {}",
        q_shape.rank(),
        v_shape.rank());
    const uint32_t BHT = q_shape[0];
    TT_FATAL(BHT % T == 0, "q dim0 ({}) must be divisible by T ({})", BHT, T);

    auto attrs = OperationType::operation_attributes_t{
        .BH = BHT / T,
        .T = T,
        .key_dim = q_shape[2],
        .val_dim = v_shape[2],
        .output_final_state = output_final_state,
        .output_per_token_state = output_per_token_state,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .decay = decay,
        .beta = beta,
        .initial_state = initial_state,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
