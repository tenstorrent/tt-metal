// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms_device_operation.hpp"

#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

namespace ttnn::experimental::prim {

ReduceAffineTransformsOperation::program_factory_t ReduceAffineTransformsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ReduceAffineTransformsProgramFactory{};
}
void ReduceAffineTransformsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "reduce_affine_transforms";
    constexpr std::array accepted_summary_dtypes = {tt::tt_metal::DataType::FLOAT32, tt::tt_metal::DataType::BFLOAT16};
    kda_factory_detail::check_allocated_device_tensor(in.a, operation_name, "a");
    kda_factory_detail::check_layout(in.a, tt::tt_metal::Layout::TILE, operation_name, "a");
    kda_factory_detail::check_dtype_in(in.a, accepted_summary_dtypes, "FLOAT32 or BFLOAT16", operation_name, "a");
    kda_factory_detail::check_allocated_device_tensor(in.b, operation_name, "b");
    kda_factory_detail::check_layout(in.b, tt::tt_metal::Layout::TILE, operation_name, "b");
    kda_factory_detail::check_dtype_in(in.b, accepted_summary_dtypes, "FLOAT32 or BFLOAT16", operation_name, "b");
    kda_factory_detail::check_same_device(in.a, in.b, operation_name, "b");
    kda_factory_detail::check_matching_dtype(in.a, in.b, operation_name, "inputs");
    const auto check_input_memory_layout = [operation_name](const ttnn::Tensor& tensor, std::string_view name) {
        const auto memory_layout = tensor.memory_config().memory_layout();
        TT_FATAL(
            memory_layout == tt::tt_metal::TensorMemoryLayout::INTERLEAVED ||
                memory_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "{}: {} must use interleaved or height-sharded memory",
            operation_name,
            name);
    };
    check_input_memory_layout(in.a, "a");
    check_input_memory_layout(in.b, "b");
    TT_FATAL(attrs.groups_per_head > 0, "reduce_affine_transforms: groups_per_head must be positive");
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, operation_name);

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

    const auto grid = in.a.device()->compute_with_storage_grid_size();
    const uint32_t worker_limit = grid.x * grid.y;
    const uint32_t group_workers = attrs.batch_heads * attrs.groups_per_head;
    TT_FATAL(
        group_workers <= worker_limit,
        "reduce_affine_transforms: supports at most {} group workers on this device, got {}",
        worker_limit,
        group_workers);
}
ReduceAffineTransformsOperation::spec_return_value_t ReduceAffineTransformsOperation::compute_output_specs(
    const operation_attributes_t& a, const tensor_args_t&) {
    const auto layout = [&](const Shape& shape) {
        return tt::tt_metal::TensorSpec(
            shape,
            tt::tt_metal::TensorLayout(
                tt::tt_metal::DataType::FLOAT32,
                tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE),
                a.output_mem_config));
    };
    return {
        layout(Shape({a.batch_heads, a.key_dim, a.key_dim})), layout(Shape({a.batch_heads, a.key_dim, a.value_dim}))};
}
ReduceAffineTransformsOperation::tensor_return_value_t ReduceAffineTransformsOperation::create_output_tensors(
    const operation_attributes_t& a, const tensor_args_t& in) {
    auto specs = compute_output_specs(a, in);
    return {create_device_tensor(specs[0], in.a.device()), create_device_tensor(specs[1], in.a.device())};
}
std::pair<ttnn::Tensor, ttnn::Tensor> reduce_affine_transforms(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    uint32_t groups,
    const tt::tt_metal::MemoryConfig& mem,
    const ttnn::DeviceComputeKernelConfig& cfg) {
    // Cache-miss validation cannot protect attribute construction on cache hits. Keep these guards here because the
    // launcher divides by groups and indexes both shapes before dispatching validation.
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
