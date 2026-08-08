// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_affine_prefix_device_operation.hpp"

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {
void validate_affine_prefix_transform(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.layout() == Layout::TILE, "KDA affine prefix: {} must be tiled", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "KDA affine prefix: {} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(tensor.buffer() != nullptr, "KDA affine prefix: {} must be on device", name);
}
}  // namespace

KdaAffinePrefixOperation::program_factory_t KdaAffinePrefixOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaAffinePrefixProgramFactory{};
}

void KdaAffinePrefixOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    validate_affine_prefix_transform(in.transform_a, "transform_a");
    validate_affine_prefix_transform(in.transform_b, "transform_b");
    TT_FATAL(in.transform_a.dtype() == in.transform_b.dtype(), "KDA affine summaries must have matching dtypes");
    TT_FATAL(
        in.initial_state.layout() == Layout::TILE && in.initial_state.dtype() == DataType::FLOAT32 &&
            in.initial_state.buffer(),
        "KDA affine prefix initial_state must be device FLOAT32 TILE");
    const auto& a = in.transform_a.logical_shape();
    const auto& b = in.transform_b.logical_shape();
    const auto& state = in.initial_state.logical_shape();
    TT_FATAL(
        attrs.groups_per_head > 0 && a.rank() == 3 && b.rank() == 3 && state.rank() == 3,
        "KDA affine prefix expects rank-3 tensors");
    TT_FATAL(
        a[0] == attrs.batch_heads * attrs.groups_per_head && b[0] == a[0] && a[1] == attrs.key_dim &&
            a[2] == attrs.key_dim && b[1] == attrs.key_dim && b[2] == attrs.value_dim,
        "KDA affine prefix transform shape mismatch");
    TT_FATAL(
        state[0] == attrs.batch_heads && state[1] == attrs.key_dim && state[2] == attrs.value_dim,
        "KDA affine prefix state shape mismatch");
}

KdaAffinePrefixOperation::spec_return_value_t KdaAffinePrefixOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    return {TensorSpec(
        Shape({a.batch_heads * a.groups_per_head, a.key_dim, a.value_dim}),
        TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), a.output_mem_config))};
}
KdaAffinePrefixOperation::tensor_return_value_t KdaAffinePrefixOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(a, in)[0], in.transform_a.device())};
}
Tensor kda_affine_prefix(
    const Tensor& a,
    const Tensor& b,
    const Tensor& state,
    uint32_t groups,
    const tt::tt_metal::MemoryConfig& mem,
    const DeviceComputeKernelConfig& cfg) {
    TT_FATAL(groups > 0, "KDA affine prefix groups_per_head must be positive");
    const auto& shape = a.logical_shape();
    auto outputs = ::ttnn::device_operation::launch<KdaAffinePrefixOperation>(
        KdaAffinePrefixParams{
            .batch_heads = static_cast<uint32_t>(shape[0]) / groups,
            .groups_per_head = groups,
            .key_dim = static_cast<uint32_t>(shape[1]),
            .value_dim = static_cast<uint32_t>(b.logical_shape()[2]),
            .output_mem_config = mem,
            .compute_kernel_config = cfg},
        KdaAffinePrefixInputs{.transform_a = a, .transform_b = b, .initial_state = state});
    return outputs[0];
}
}  // namespace ttnn::prim
