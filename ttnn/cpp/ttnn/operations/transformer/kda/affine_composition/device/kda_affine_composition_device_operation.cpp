// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "kda_affine_composition_device_operation.hpp"

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {
namespace {
void validate_affine_composition_transform(const Tensor& tensor, const char* name) {
    TT_FATAL(tensor.layout() == Layout::TILE, "KDA affine composition: {} must be tiled", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "KDA affine composition: {} must be FLOAT32 or BFLOAT16",
        name);
    TT_FATAL(tensor.buffer() != nullptr, "KDA affine composition: {} must be on device", name);
}
}  // namespace

KdaAffineCompositionOperation::program_factory_t KdaAffineCompositionOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return KdaAffineCompositionProgramFactory{};
}
void KdaAffineCompositionOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    validate_affine_composition_transform(in.transform_a, "transform_a");
    validate_affine_composition_transform(in.transform_b, "transform_b");
    TT_FATAL(in.transform_a.dtype() == in.transform_b.dtype(), "KDA affine summaries must have matching dtypes");
    const auto& a = in.transform_a.logical_shape();
    const auto& b = in.transform_b.logical_shape();
    TT_FATAL(
        attrs.groups_per_head > 0 && a.rank() == 3 && b.rank() == 3,
        "KDA affine composition expects rank-3 transforms");
    TT_FATAL(
        a[0] == attrs.batch_heads * attrs.groups_per_head && b[0] == a[0] && a[1] == attrs.key_dim &&
            a[2] == attrs.key_dim && b[1] == attrs.key_dim && b[2] == attrs.value_dim,
        "KDA affine composition transform shape mismatch");
}
KdaAffineCompositionOperation::spec_return_value_t KdaAffineCompositionOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    const auto layout = [&](const Shape& shape) {
        return TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), a.output_mem_config));
    };
    return {
        layout(Shape({a.batch_heads, a.key_dim, a.key_dim})), layout(Shape({a.batch_heads, a.key_dim, a.value_dim}))};
}
KdaAffineCompositionOperation::tensor_return_value_t KdaAffineCompositionOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    auto specs = compute_output_specs(a, in);
    return {
        create_device_tensor(specs[0], in.transform_a.device()),
        create_device_tensor(specs[1], in.transform_a.device())};
}
std::pair<Tensor, Tensor> kda_affine_compose(
    const Tensor& a,
    const Tensor& b,
    uint32_t groups,
    const tt::tt_metal::MemoryConfig& mem,
    const DeviceComputeKernelConfig& cfg) {
    TT_FATAL(groups > 0, "KDA affine composition groups_per_head must be positive");
    const auto& shape = a.logical_shape();
    auto outputs = ::ttnn::device_operation::launch<KdaAffineCompositionOperation>(
        KdaAffineCompositionParams{
            .batch_heads = static_cast<uint32_t>(shape[0]) / groups,
            .groups_per_head = groups,
            .key_dim = static_cast<uint32_t>(shape[1]),
            .value_dim = static_cast<uint32_t>(b.logical_shape()[2]),
            .output_mem_config = mem,
            .compute_kernel_config = cfg},
        KdaAffineCompositionInputs{.transform_a = a, .transform_b = b});
    return {outputs[0], outputs[1]};
}
}  // namespace ttnn::prim
