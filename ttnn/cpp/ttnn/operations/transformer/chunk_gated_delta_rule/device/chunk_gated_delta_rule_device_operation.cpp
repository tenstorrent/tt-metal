// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_gated_delta_rule_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

ChunkGatedDeltaRuleDeviceOperation::program_factory_t ChunkGatedDeltaRuleDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ChunkGatedDeltaRuleProgramFactory{};
}

void ChunkGatedDeltaRuleDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace tt::constants;
    auto check = [](const Tensor& t, const char* name, DataType dt) {
        TT_FATAL(t.layout() == Layout::TILE, "chunk_gated_delta_rule: {} must be TILE layout", name);
        TT_FATAL(t.dtype() == dt, "chunk_gated_delta_rule: {} has wrong dtype", name);
        TT_FATAL(t.buffer() != nullptr, "chunk_gated_delta_rule: {} must be on device", name);
    };
    // GPU-style mixed precision: q/k/v are bf16; gate/decay, state and mask constants stay fp32.
    check(in.q, "q", DataType::BFLOAT16);
    check(in.k, "k", DataType::BFLOAT16);
    check(in.v, "v", DataType::BFLOAT16);
    check(in.g, "g", DataType::FLOAT32);
    check(in.beta, "beta", DataType::FLOAT32);
    check(in.eye_c, "eye_c", DataType::FLOAT32);
    check(in.tril_c, "tril_c", DataType::FLOAT32);
    check(in.ones_c, "ones_c", DataType::FLOAT32);
    if (in.initial_state.has_value()) {
        check(*in.initial_state, "initial_state", DataType::FLOAT32);
    }
    TT_FATAL(attrs.chunk_size % TILE_HEIGHT == 0, "chunk_size must be a multiple of 32");
    TT_FATAL(attrs.key_dim % TILE_WIDTH == 0, "key_dim must be a multiple of 32");
    TT_FATAL(attrs.val_dim % TILE_WIDTH == 0, "val_dim must be a multiple of 32");
}

ChunkGatedDeltaRuleDeviceOperation::spec_return_value_t ChunkGatedDeltaRuleDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    // o is bf16 (matches cb_out); the recurrent final state stays fp32.
    const auto o_layout = TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_mem_config);
    const auto s_layout = TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), attrs.output_mem_config);
    ttnn::Shape o_shape({attrs.BH, attrs.num_chunks, attrs.chunk_size, attrs.val_dim});
    ttnn::Shape s_shape({attrs.BH, attrs.key_dim, attrs.val_dim});
    return {tt::tt_metal::TensorSpec(o_shape, o_layout), tt::tt_metal::TensorSpec(s_shape, s_layout)};
}

ChunkGatedDeltaRuleDeviceOperation::tensor_return_value_t ChunkGatedDeltaRuleDeviceOperation::create_output_tensors(
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

std::vector<Tensor> chunk_gated_delta_rule(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye_c,
    const Tensor& tril_c,
    const Tensor& ones_c,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    bool output_final_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using OperationType = ChunkGatedDeltaRuleDeviceOperation;

    const auto& q_shape = q.logical_shape();  // [BH, NC, C, K]
    const auto& v_shape = v.logical_shape();  // [BH, NC, C, V]

    auto attrs = OperationType::operation_attributes_t{
        .BH = q_shape[0],
        .num_chunks = q_shape[1],
        .chunk_size = chunk_size,
        .key_dim = q_shape[3],
        .val_dim = v_shape[3],
        .has_initial_state = initial_state.has_value(),
        .output_final_state = output_final_state,
        .output_mem_config = output_mem_config,
        .compute_kernel_config = compute_kernel_config,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .q = q,
        .k = k,
        .v = v,
        .g = g,
        .beta = beta,
        .eye_c = eye_c,
        .tril_c = tril_c,
        .ones_c = ones_c,
        .initial_state = initial_state,
    };
    return ttnn::device_operation::launch<OperationType>(attrs, tensor_args);
}

}  // namespace ttnn::prim
