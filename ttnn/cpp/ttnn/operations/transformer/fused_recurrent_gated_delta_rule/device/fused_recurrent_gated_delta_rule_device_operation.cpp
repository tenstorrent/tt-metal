// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "fused_recurrent_gated_delta_rule_device_operation.hpp"

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
    auto check = [](const Tensor& t, const char* name) {
        TT_FATAL(t.layout() == Layout::TILE, "fused_recurrent_gated_delta_rule: {} must be TILE layout", name);
        TT_FATAL(t.dtype() == DataType::FLOAT32, "fused_recurrent_gated_delta_rule: {} must be fp32", name);
        TT_FATAL(t.buffer() != nullptr, "fused_recurrent_gated_delta_rule: {} must be on device", name);
    };
    check(in.q, "q");
    check(in.k, "k");
    check(in.v, "v");
    check(in.decay, "decay");
    check(in.beta, "beta");
    if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state");
    }
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
    TT_FATAL(attrs.T >= 1, "T must be >= 1");
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
