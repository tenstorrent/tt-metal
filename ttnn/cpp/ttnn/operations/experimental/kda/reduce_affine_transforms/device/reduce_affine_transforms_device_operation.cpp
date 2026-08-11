// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms_device_operation.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {
namespace {
void check_affine_tensor(const Tensor& tensor, const char* name) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.buffer() != nullptr,
        "reduce_affine_transforms: {} must be an allocated device tensor",
        name);
    TT_FATAL(tensor.layout() == Layout::TILE, "reduce_affine_transforms: {} must use TILE layout", name);
    TT_FATAL(
        tensor.dtype() == DataType::FLOAT32 || tensor.dtype() == DataType::BFLOAT16,
        "reduce_affine_transforms: {} must be FLOAT32 or BFLOAT16",
        name);
}
}  // namespace

ReduceAffineTransformsOperation::program_factory_t ReduceAffineTransformsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ReduceAffineTransformsProgramFactory{};
}
void ReduceAffineTransformsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    check_affine_tensor(in.a, "a");
    check_affine_tensor(in.b, "b");
    TT_FATAL(in.a.device() == in.b.device(), "reduce_affine_transforms: all inputs must be on the same device");
    TT_FATAL(in.a.dtype() == in.b.dtype(), "reduce_affine_transforms: inputs must have matching dtypes");
    TT_FATAL(attrs.groups_per_head > 0, "reduce_affine_transforms: groups_per_head must be positive");
    TT_FATAL(
        !attrs.output_mem_config.is_sharded(),
        "reduce_affine_transforms: output memory configuration must be interleaved");

    const auto& a_shape = in.a.logical_shape();
    const auto& b_shape = in.b.logical_shape();
    TT_FATAL(a_shape.rank() == 3 && b_shape.rank() == 3, "reduce_affine_transforms: inputs must be rank 3");
    TT_FATAL(a_shape[0] > 0, "reduce_affine_transforms: leading dimension must be positive");
    TT_FATAL(
        a_shape[0] % attrs.groups_per_head == 0,
        "reduce_affine_transforms: leading dimension must be divisible by groups_per_head");
    TT_FATAL(a_shape[0] == b_shape[0], "reduce_affine_transforms: inputs must have matching leading dimensions");
    TT_FATAL(a_shape[1] == a_shape[2], "reduce_affine_transforms: a must contain square KxK matrices");
    TT_FATAL(a_shape[1] == b_shape[1], "reduce_affine_transforms: a and b must have matching K dimensions");
    TT_FATAL(
        a_shape[1] > 0 && b_shape[2] > 0 && a_shape[1] % tt::constants::TILE_WIDTH == 0 &&
            b_shape[2] % tt::constants::TILE_WIDTH == 0,
        "reduce_affine_transforms: K and V must be positive and tile aligned");
    TT_FATAL(
        a_shape[0] == attrs.batch_heads * attrs.groups_per_head && a_shape[1] == attrs.key_dim &&
            b_shape[2] == attrs.value_dim,
        "reduce_affine_transforms: input shapes must match operation attributes");
}
ReduceAffineTransformsOperation::spec_return_value_t ReduceAffineTransformsOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    const auto layout = [&](const Shape& shape) {
        return TensorSpec(shape, TensorLayout(DataType::FLOAT32, PageConfig(Layout::TILE), a.output_mem_config));
    };
    return {
        layout(Shape({a.batch_heads, a.key_dim, a.key_dim})), layout(Shape({a.batch_heads, a.key_dim, a.value_dim}))};
}
ReduceAffineTransformsOperation::tensor_return_value_t ReduceAffineTransformsOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    auto specs = compute_output_specs(a, in);
    return {create_device_tensor(specs[0], in.a.device()), create_device_tensor(specs[1], in.a.device())};
}
std::pair<Tensor, Tensor> reduce_affine_transforms(
    const Tensor& a,
    const Tensor& b,
    uint32_t groups,
    const tt::tt_metal::MemoryConfig& mem,
    const DeviceComputeKernelConfig& cfg) {
    TT_FATAL(groups > 0, "reduce_affine_transforms: groups_per_head must be positive");
    const auto& shape = a.logical_shape();
    const auto& b_shape = b.logical_shape();
    TT_FATAL(shape.rank() == 3 && b_shape.rank() == 3, "reduce_affine_transforms: inputs must be rank 3");
    TT_FATAL(shape[0] > 0, "reduce_affine_transforms: leading dimension must be positive");
    TT_FATAL(
        shape[0] % groups == 0, "reduce_affine_transforms: leading dimension must be divisible by groups_per_head");
    auto outputs = ttnn::device_operation::launch<ReduceAffineTransformsOperation>(
        ReduceAffineTransformsParams{
            .batch_heads = static_cast<uint32_t>(shape[0]) / groups,
            .groups_per_head = groups,
            .key_dim = static_cast<uint32_t>(shape[1]),
            .value_dim = static_cast<uint32_t>(b.logical_shape()[2]),
            .output_mem_config = mem,
            .compute_kernel_config = cfg},
        ReduceAffineTransformsInputs{.a = a, .b = b});
    return {outputs[0], outputs[1]};
}
}  // namespace ttnn::experimental::prim
