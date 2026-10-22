// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "matmul_reduce_scatter_device_operation.hpp"

#include <string_view>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

using namespace tt::tt_metal;

namespace {

void refuse_unless(bool supported, std::string_view axis, std::string_view value, std::string_view allowed) {
    if (!supported) {
        throw UnsupportedAxisValue(
            fmt::format("matmul_reduce_scatter: {}={} not in SUPPORTED [{}]", axis, value, allowed));
    }
}

bool is_dram_interleaved_tile(const Tensor& t) {
    const MemoryConfig& mc = t.memory_config();
    return mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED &&
           t.layout() == Layout::TILE;
}

}  // namespace

void check_support(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    tt::tt_fabric::Topology topology,
    const std::optional<int32_t>& num_links) {
    const auto& a_shape = input_tensor.logical_shape();
    const auto& w_shape = weight.logical_shape();
    if (a_shape.rank() < 2 || w_shape.rank() < 1) {
        throw std::out_of_range("matmul_reduce_scatter: input_tensor must have at least two dims");
    }
    const auto rank = static_cast<int32_t>(a_shape.rank());
    const int32_t canonical_scatter_dim = scatter_dim >= 0 ? scatter_dim - rank : scatter_dim;
    const Layout layout = input_tensor.layout() != Layout::TILE ? input_tensor.layout() : weight.layout();
    const uint32_t tile = tt::constants::TILE_HEIGHT;
    const bool aligned = a_shape[-2] % tile == 0 && a_shape[-1] % tile == 0 && w_shape[-1] % tile == 0;

    const DataType dtype = input_tensor.dtype();
    refuse_unless(
        dtype == DataType::BFLOAT16 || dtype == DataType::BFLOAT8_B,
        "dtype",
        fmt::format("{}", dtype),
        "BFLOAT16, BFLOAT8_B");
    const DataType weight_dtype = weight.dtype();
    refuse_unless(
        weight_dtype == DataType::BFLOAT16 || weight_dtype == DataType::BFLOAT8_B ||
            weight_dtype == DataType::BFLOAT4_B,
        "weight_dtype",
        fmt::format("{}", weight_dtype),
        "BFLOAT16, BFLOAT8_B, BFLOAT4_B");
    refuse_unless(layout == Layout::TILE, "layout", fmt::format("{}", layout), "TILE");
    refuse_unless(aligned, "alignment", "non_aligned", "tile_aligned");
    refuse_unless(cluster_axis == 0 || cluster_axis == 1, "cluster_axis", fmt::format("{}", cluster_axis), "0, 1");
    refuse_unless(
        canonical_scatter_dim == -1 || canonical_scatter_dim == -2,
        "scatter_dim",
        fmt::format("{}", canonical_scatter_dim),
        "-1, -2");
    refuse_unless(
        topology == tt::tt_fabric::Topology::Linear || topology == tt::tt_fabric::Topology::Ring,
        "topology",
        fmt::format("{}", static_cast<int>(topology)),
        "Linear, Ring");
    if (num_links.has_value()) {
        refuse_unless(*num_links == 1 || *num_links == 2, "num_links", fmt::format("{}", *num_links), "1, 2");
    }
}

MatmulReduceScatterDeviceOperation::program_factory_t MatmulReduceScatterDeviceOperation::select_program_factory(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return MatmulReduceScatterProgramFactory{};
}

void MatmulReduceScatterDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    TT_FATAL(is_dram_interleaved_tile(t.input), "matmul_reduce_scatter: input_tensor must be DRAM interleaved");
    TT_FATAL(is_dram_interleaved_tile(t.weight), "matmul_reduce_scatter: weight must be DRAM interleaved");
    if (!t.output.has_value()) {
        return;
    }
    const Tensor& output = *t.output;
    const TensorSpec expected = compute_output_specs(attrs, t);
    const TensorSpec& given = output.tensor_spec();
    TT_FATAL(
        given.logical_shape() == expected.logical_shape(),
        "matmul_reduce_scatter: output_tensor shape {} does not match {}",
        given.logical_shape(),
        expected.logical_shape());
    TT_FATAL(
        given.data_type() == expected.data_type(),
        "matmul_reduce_scatter: output_tensor dtype {} does not match {}",
        given.data_type(),
        expected.data_type());
    TT_FATAL(
        given.layout() == expected.layout(),
        "matmul_reduce_scatter: output_tensor layout {} does not match {}",
        given.layout(),
        expected.layout());
    TT_FATAL(
        given.memory_config() == expected.memory_config(),
        "matmul_reduce_scatter: output_tensor memory_config {} does not match {}",
        given.memory_config(),
        expected.memory_config());
    TT_FATAL(given == expected, "matmul_reduce_scatter: output_tensor spec {} does not match {}", given, expected);
    // Fin writers store output tiles while other cores still read A and W, so the output may not alias them.
    for (const Tensor* input : {&t.input, &t.weight}) {
        TT_FATAL(
            output.device() != input->device() ||
                output.buffer()->address() != input->buffer()->address(),  // smuggled-rta-ok: aliasing check only
            "matmul_reduce_scatter: output_tensor may not share a buffer with an input");
    }
}

MatmulReduceScatterDeviceOperation::spec_return_value_t MatmulReduceScatterDeviceOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    const uint32_t group_size = t.input.device()->shape()[attrs.cluster_axis];
    ttnn::Shape out_shape = t.input.logical_shape();
    const uint32_t n = t.weight.logical_shape()[-1];
    out_shape[-1] = n;
    if (attrs.scatter_dim == -2) {
        out_shape[-2] = out_shape[-2] / group_size;
    } else {
        out_shape[-1] = n / group_size;
    }
    return TensorSpec(
        out_shape, TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attrs.output_memory_config));
}

MatmulReduceScatterDeviceOperation::tensor_return_value_t MatmulReduceScatterDeviceOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& t) {
    if (t.output.has_value()) {
        return *t.output;
    }
    return create_device_tensor(compute_output_specs(attrs, t), t.input.device());
}

}  // namespace ttnn::operations::matmul_reduce_scatter

namespace ttnn::prim {

Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& staging,
    const Tensor& mailbox,
    const std::optional<Tensor>& output_tensor,
    const ttnn::operations::matmul_reduce_scatter::MatmulReduceScatterParams& params) {
    using OperationType = ttnn::operations::matmul_reduce_scatter::MatmulReduceScatterDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        params,
        OperationType::tensor_args_t{
            .input = input_tensor, .weight = weight, .staging = staging, .mailbox = mailbox, .output = output_tensor});
}

}  // namespace ttnn::prim
