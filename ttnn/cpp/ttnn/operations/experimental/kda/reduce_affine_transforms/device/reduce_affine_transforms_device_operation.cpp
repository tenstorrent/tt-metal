// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "reduce_affine_transforms_device_operation.hpp"

#include <algorithm>
#include <array>

#include <tt-metalium/constants.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"
#include "ttnn/operations/experimental/kda/kda_performance_model.hpp"

namespace ttnn::experimental::prim {

ReduceAffineTransformsOperation::program_factory_t ReduceAffineTransformsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ReduceAffineTransformsProgramFactory{};
}
void ReduceAffineTransformsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    constexpr std::string_view operation_name = "reduce_affine_transforms";
    TT_FATAL(attrs.groups_per_head > 0, "reduce_affine_transforms: groups_per_head must be positive");
    TT_FATAL(
        attrs.local_rows > 0 && attrs.local_rows % tt::constants::TILE_HEIGHT == 0 &&
            (attrs.local_rows / tt::constants::TILE_HEIGHT) % attrs.groups_per_head == 0,
        "{}: local_rows must contain a positive whole number of 32-token chunks per group",
        operation_name);
    kda_factory_detail::check_actual_start(in.summaries, in.actual_start, operation_name);
    if (in.actual_end) {
        kda_factory_detail::check_actual_start(in.actual_start, *in.actual_end, "reduce_affine_transforms");
    }
    const auto summary = kda_factory_detail::check_packed_summaries(in.summaries, operation_name);
    kda_factory_detail::check_output_interleaved(attrs.output_mem_config, operation_name);
    kda_factory_detail::check_compute_config(attrs.compute_kernel_config, operation_name);
    TT_FATAL(
        summary.transforms == attrs.batch_heads * attrs.groups_per_head && summary.key_dim == attrs.key_dim &&
            summary.value_dim == attrs.value_dim,
        "reduce_affine_transforms: input shapes must match operation attributes");

    constexpr uint32_t max_coordinate_table_workers = 128;
    const auto grid = in.summaries.device()->compute_with_storage_grid_size();
    const uint32_t worker_limit = std::min<uint32_t>(grid.x * grid.y, max_coordinate_table_workers);
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
    return {
        create_device_tensor(specs[0], in.summaries.device()), create_device_tensor(specs[1], in.summaries.device())};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<ReduceAffineTransformsOperation::tensor_return_value_t>
ReduceAffineTransformsOperation::create_op_performance_model(
    const operation_attributes_t& attrs, const tensor_args_t& in, tensor_return_value_t& outputs) {
    using namespace kda_performance_model;

    const double key_dim = attrs.key_dim;
    const double value_dim = attrs.value_dim;
    const double compositions = static_cast<double>(attrs.batch_heads) * (attrs.groups_per_head - 1.0);
    const KdaFpuWork work{
        .fpu_matrix_flops = compositions * (2.0 * key_dim * key_dim * key_dim + 2.0 * key_dim * key_dim * value_dim),
        .fpu_add_ops = compositions * key_dim * value_dim,
    };
    const std::array<const Tensor*, 1> inputs = {&in.summaries};
    return make_profiler_model(work, inputs, outputs, attrs.compute_kernel_config.math_fidelity);
}

std::pair<ttnn::Tensor, ttnn::Tensor> reduce_affine_transforms(
    const ttnn::Tensor& summaries,
    uint32_t groups,
    const tt::tt_metal::MemoryConfig& mem,
    const ttnn::DeviceComputeKernelConfig& cfg,
    const Tensor& actual_start,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    const std::optional<Tensor>& actual_end) {
    // Cache-miss validation cannot protect attribute construction on cache hits. Keep these guards here because the
    // launcher divides by groups and indexes the shape before dispatching validation.
    TT_FATAL(groups > 0, "reduce_affine_transforms: groups_per_head must be positive");
    const auto& shape = summaries.logical_shape();
    TT_FATAL(shape.rank() == 4, "reduce_affine_transforms: summaries must be rank 4 [S, N, K, K+V]");
    TT_FATAL(shape[1] > 0, "reduce_affine_transforms: summaries must hold at least one transform");
    TT_FATAL(
        shape[1] % groups == 0, "reduce_affine_transforms: transforms per slot must be divisible by groups_per_head");
    TT_FATAL(shape[3] > shape[2], "reduce_affine_transforms: summary rows must be [A | B]");
    auto outputs = ttnn::device_operation::launch<ReduceAffineTransformsOperation>(
        ReduceAffineTransformsParams{
            .batch_heads = static_cast<uint32_t>(shape[1]) / groups,
            .groups_per_head = groups,
            .key_dim = static_cast<uint32_t>(shape[2]),
            .value_dim = static_cast<uint32_t>(shape[3] - shape[2]),
            .sequence_parallel_axis = sequence_parallel_axis,
            .local_rows = local_rows,
            .output_mem_config = mem,
            .compute_kernel_config = cfg},
        ReduceAffineTransformsInputs{.summaries = summaries, .actual_start = actual_start, .actual_end = actual_end});
    return {outputs[0], outputs[1]};
}
}  // namespace ttnn::experimental::prim
