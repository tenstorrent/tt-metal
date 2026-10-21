// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_device_operation.hpp"

#include <string_view>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

using namespace tt::tt_metal;

namespace {

void refuse_axis_unless(bool supported, std::string_view axis, const std::string& value, std::string_view allowed) {
    if (!supported) {
        throw UnsupportedAxisValue(
            fmt::format("matmul_reduce_scatter: {}={} not in SUPPORTED [{}]", axis, value, allowed));
    }
}

void validate_output(const MatmulReduceScatterParams& attrs, const MatmulReduceScatterInputs& t) {
    if (!t.output.has_value()) {
        return;
    }
    const Tensor& output = *t.output;
    const TensorSpec expected = MatmulReduceScatterDeviceOperation::compute_output_specs(attrs, t);
    TT_FATAL(
        output.logical_shape() == expected.logical_shape(),
        "matmul_reduce_scatter: output_tensor shape {} does not match {}",
        output.logical_shape(),
        expected.logical_shape());
    TT_FATAL(
        output.dtype() == expected.data_type(),
        "matmul_reduce_scatter: output_tensor dtype {} does not match {}",
        output.dtype(),
        expected.data_type());
    TT_FATAL(
        output.layout() == expected.layout(),
        "matmul_reduce_scatter: output_tensor layout {} does not match {}",
        output.layout(),
        expected.layout());
    TT_FATAL(
        output.memory_config() == expected.memory_config(),
        "matmul_reduce_scatter: output_tensor memory_config {} does not match {}",
        output.memory_config(),
        expected.memory_config());
    TT_FATAL(
        output.tensor_spec() == expected,
        "matmul_reduce_scatter: output_tensor spec {} does not match {}",
        output.tensor_spec(),
        expected);
    // Every core streams A and W from DRAM across many entries while other cores already store finished tiles,
    // so an output sharing an input's buffer would be overwritten before it is read.
    TT_FATAL(
        output.buffer() != t.input_tensor.buffer() && output.buffer() != t.weight.buffer() &&
            output.buffer() != t.staging.buffer(),
        "matmul_reduce_scatter: output_tensor must not share a buffer with an input (in place is refused)");
}

}  // namespace

void check_support(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links) {
    const auto& a_shape = input_tensor.logical_shape();
    const auto& w_shape = weight.logical_shape();
    TT_FATAL(a_shape.rank() >= 2, "matmul_reduce_scatter: A must be rank 2-4, got {}", a_shape);

    const DataType dtype = input_tensor.dtype();
    refuse_axis_unless(
        dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B,
        "dtype",
        fmt::format("{}", dtype),
        "BFLOAT16, BFLOAT8_B");
    const DataType weight_dtype = weight.dtype();
    refuse_axis_unless(
        weight_dtype == DataType::BFLOAT16 || weight_dtype == DataType::BFLOAT8_B ||
            weight_dtype == DataType::BFLOAT4_B,
        "weight_dtype",
        fmt::format("{}", weight_dtype),
        "BFLOAT16, BFLOAT8_B, BFLOAT4_B");
    const Layout layout = input_tensor.layout() != Layout::TILE ? input_tensor.layout() : weight.layout();
    refuse_axis_unless(layout == Layout::TILE, "layout", fmt::format("{}", layout), "TILE");
    const uint32_t m = a_shape[-2];
    const uint32_t k = a_shape[-1];
    const uint32_t n = w_shape[-1];
    const bool aligned =
        m % tt::constants::TILE_HEIGHT == 0 && k % tt::constants::TILE_WIDTH == 0 && n % tt::constants::TILE_WIDTH == 0;
    refuse_axis_unless(aligned, "alignment", aligned ? "tile_aligned" : "non_aligned", "tile_aligned");
    refuse_axis_unless(cluster_axis == 0 || cluster_axis == 1, "cluster_axis", std::to_string(cluster_axis), "0, 1");
    refuse_axis_unless(scatter_dim == -1 || scatter_dim == -2, "scatter_dim", std::to_string(scatter_dim), "-1, -2");
    refuse_axis_unless(
        topology == tt::tt_fabric::Topology::Linear || topology == tt::tt_fabric::Topology::Ring,
        "topology",
        std::to_string(static_cast<uint32_t>(topology)),
        "Linear, Ring");
    if (num_links.has_value()) {
        refuse_axis_unless(*num_links == 1 || *num_links == 2, "num_links", std::to_string(*num_links), "1, 2");
    }
}

MatmulReduceScatterDeviceOperation::program_factory_t MatmulReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return MatmulReduceScatterProgramFactory{};
}

void MatmulReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    // num_links is resolved (it may be every usable link), so only the given-value axis is skipped here.
    check_support(
        t.input_tensor,
        t.weight,
        static_cast<int32_t>(attrs.cluster_axis),
        attrs.scatter_dim,
        attrs.topology,
        std::nullopt);
    validate_output(attrs, t);
}

void MatmulReduceScatterDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    validate_output(attrs, t);
}

MatmulReduceScatterDeviceOperation::spec_return_value_t MatmulReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    ttnn::Shape shape = t.input_tensor.logical_shape();
    const uint32_t n = t.weight.logical_shape()[-1];
    shape[-1] = n;
    if (attrs.scatter_dim == -2) {
        shape[-2] = shape[-2] / attrs.group_size;
    } else {
        shape[-1] = n / attrs.group_size;
    }
    return TensorSpec(shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_memory_config));
}

MatmulReduceScatterDeviceOperation::tensor_return_value_t MatmulReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return *t.output;
    }
    return create_device_tensor(compute_output_specs(attrs, t), t.input_tensor.device());
}

}  // namespace ttnn::operations::matmul_reduce_scatter

namespace ttnn::prim {

Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& staging,
    const std::optional<Tensor>& output,
    const ttnn::operations::matmul_reduce_scatter::MatmulReduceScatterParams& params) {
    using OperationType = ttnn::operations::matmul_reduce_scatter::MatmulReduceScatterDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .weight = weight, .staging = staging, .output = output});
}

}  // namespace ttnn::prim
