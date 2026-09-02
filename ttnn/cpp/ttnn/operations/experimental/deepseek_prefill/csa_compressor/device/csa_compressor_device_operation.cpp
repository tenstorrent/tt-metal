// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "csa_compressor_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {
namespace {

constexpr uint32_t kProjectionDim = 1024;
constexpr uint32_t kHeadDim = 512;
constexpr uint32_t kStateRows = 64;
constexpr uint32_t kRatio = 4;

void validate_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.storage_type() == StorageType::DEVICE, "{} must be a device tensor", name);
    TT_FATAL(tensor.dtype() == DataType::BFLOAT16, "{} must be BFLOAT16", name);
    TT_FATAL(tensor.layout() == Layout::TILE, "{} must use TILE layout", name);
    TT_FATAL(!tensor.is_sharded(), "{} must use interleaved memory", name);
    TT_FATAL(tensor.logical_shape().rank() == 4, "{} must be rank 4", name);
}

template <typename Inputs>
void validate_common(const CsaRuntimeParams& params, const Inputs& args) {
    validate_tensor(args.kv, "kv");
    validate_tensor(args.gate, "gate");
    validate_tensor(args.position_bias, "position_bias");
    TT_FATAL(args.kv.logical_shape() == args.gate.logical_shape(), "kv and gate shapes must match");
    const auto& shape = args.kv.logical_shape();
    TT_FATAL(shape[0] == 1 && shape[1] == 1 && shape[-1] == kProjectionDim, "kv/gate must be [1,1,S,1024]");
    TT_FATAL(shape[-2] >= kRatio && shape[-2] % kRatio == 0, "S_local must be a positive multiple of 4");
    TT_FATAL(
        args.position_bias.logical_shape() == Shape({1, 1, kRatio, kProjectionDim}),
        "position_bias must be [1,1,4,1024]");
    TT_FATAL(params.cluster_axis < 2, "cluster_axis must be 0 or 1");
    TT_FATAL(
        params.first_token_position % kRatio == 0, "first_token_position must start on a ratio-4 compression boundary");
    TT_FATAL(args.kv.device() == args.gate.device(), "all inputs must share one mesh device");
    TT_FATAL(args.kv.device() == args.position_bias.device(), "all inputs must share one mesh device");
    const auto mesh_shape = args.kv.device()->shape();
    TT_FATAL(mesh_shape.dims() == 2, "csa_compressor requires a 2D mesh");
    TT_FATAL(
        params.seq_len_actual <= shape[-2] * mesh_shape[params.cluster_axis],
        "seq_len_actual exceeds the global padded slab");
}

void validate_states(const Tensor& kv_state, const Tensor& score_state, const Tensor& kv) {
    validate_tensor(kv_state, "kv_state");
    validate_tensor(score_state, "score_state");
    TT_FATAL(kv_state.logical_shape() == Shape({1, 1, kStateRows, kHeadDim}), "KV state must be [1,1,64,512] locally");
    TT_FATAL(score_state.logical_shape() == kv_state.logical_shape(), "state shapes must match");
    TT_FATAL(kv_state.tensor_spec() == score_state.tensor_spec(), "state tensor specs must match");
    TT_FATAL(kv_state.device() == kv.device(), "states and slab must share one mesh device");
    TT_FATAL(score_state.device() == kv.device(), "states and slab must share one mesh device");
}

template <std::size_t N>
std::array<Tensor, N> make_outputs(const std::array<tt::tt_metal::TensorSpec, N>& specs, const Tensor& input) {
    std::array<Tensor, N> outputs;
    for (std::size_t i = 0; i < N; ++i) {
        outputs[i] = create_device_tensor(specs[i], input.device());
    }
    return outputs;
}

}  // namespace

void CsaStatePreparationDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& args) {
    validate_common(params, args);
    validate_states(args.base_kv_state, args.base_score_state, args.kv);
}

CsaStatePreparationDeviceOperation::spec_return_value_t CsaStatePreparationDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    return {args.base_kv_state.tensor_spec(), args.base_score_state.tensor_spec()};
}

CsaStatePreparationDeviceOperation::topology_return_value_t
CsaStatePreparationDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& args) {
    return {args.base_kv_state.tensor_topology(), args.base_score_state.tensor_topology()};
}

CsaStatePreparationDeviceOperation::tensor_return_value_t CsaStatePreparationDeviceOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& args) {
    return make_outputs(compute_output_specs(params, args), args.kv);
}

void CsaCompressionDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& params, const tensor_args_t& args) {
    validate_common(params, args);
    validate_states(args.predecessor_kv_state, args.predecessor_score_state, args.kv);
}

CsaCompressionDeviceOperation::spec_return_value_t CsaCompressionDeviceOperation::compute_output_specs(
    const operation_attributes_t&, const tensor_args_t& args) {
    const auto& input_layout = args.kv.tensor_spec().tensor_layout();
    const auto pooled_spec =
        tt::tt_metal::TensorSpec(Shape({1, 1, args.kv.logical_shape()[-2] / kRatio, kHeadDim}), input_layout);
    return {pooled_spec, args.predecessor_kv_state.tensor_spec(), args.predecessor_score_state.tensor_spec()};
}

CsaCompressionDeviceOperation::topology_return_value_t CsaCompressionDeviceOperation::compute_output_topologies(
    const operation_attributes_t&, const tensor_args_t& args) {
    return {
        args.kv.tensor_topology(),
        args.predecessor_kv_state.tensor_topology(),
        args.predecessor_score_state.tensor_topology()};
}

CsaCompressionDeviceOperation::tensor_return_value_t CsaCompressionDeviceOperation::create_output_tensors(
    const operation_attributes_t& params, const tensor_args_t& args) {
    return make_outputs(compute_output_specs(params, args), args.kv);
}

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::array<Tensor, 2> csa_prepare_state(
    const Tensor& kv,
    const Tensor& gate,
    const Tensor& position_bias,
    const Tensor& base_kv_state,
    const Tensor& base_score_state,
    uint32_t seq_len_actual,
    uint32_t first_token_position,
    uint32_t cluster_axis) {
    using Op = ttnn::experimental::prim::CsaStatePreparationDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        {seq_len_actual, first_token_position, cluster_axis},
        {kv, gate, position_bias, base_kv_state, base_score_state});
}

std::array<Tensor, 3> csa_compress(
    const Tensor& kv,
    const Tensor& gate,
    const Tensor& position_bias,
    const Tensor& predecessor_kv_state,
    const Tensor& predecessor_score_state,
    uint32_t seq_len_actual,
    uint32_t first_token_position,
    uint32_t cluster_axis) {
    using Op = ttnn::experimental::prim::CsaCompressionDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        {seq_len_actual, first_token_position, cluster_axis},
        {kv, gate, position_bias, predecessor_kv_state, predecessor_score_state});
}

}  // namespace ttnn::prim
