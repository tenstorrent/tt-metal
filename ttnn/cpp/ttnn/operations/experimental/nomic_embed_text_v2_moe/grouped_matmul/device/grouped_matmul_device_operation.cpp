// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul_device_operation.hpp"

#include <algorithm>
#include <string_view>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::TensorMemoryLayout;

constexpr uint32_t kTileSize = tt::constants::TILE_HEIGHT;
// The data-movement kernels keep two words per expert on their small stacks.
constexpr uint32_t kMaxExperts = 64;
// Output tiles a subblock may hold, before the DST limit.
constexpr uint32_t kMaxSubblockWidth = 4;

// The framework rejects host tensors before this runs.
void check_tile_operand(const Tensor& tensor, std::string_view name) {
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B,
        "grouped_matmul: {} must be BFLOAT16 or BFLOAT8_B, got {}",
        name,
        tensor.dtype());
    TT_FATAL(tensor.layout() == Layout::TILE, "grouped_matmul: {} must be TILE layout, got {}", name, tensor.layout());
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "grouped_matmul: {} must be interleaved, got {}",
        name,
        tensor.memory_config().memory_layout());
    const auto tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == kTileSize && tile.get_width() == kTileSize && !tile.get_transpose_of_faces() &&
            !tile.get_transpose_within_face(),
        "grouped_matmul: {} must use plain 32 x 32 tiles",
        name);
    const auto& shape = tensor.logical_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[0] == 1 && shape[1] == 1,
        "grouped_matmul: {} must be 1 x 1 x rows x columns, got {}",
        name,
        shape);
    TT_FATAL(
        shape[2] % kTileSize == 0 && shape[3] % kTileSize == 0,
        "grouped_matmul: {} must be whole tiles, got {}",
        name,
        shape);
}

}  // namespace

GroupedMatmulLayout GroupedMatmulDeviceOperation::compute_layout(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& counts = tensor_args.expert_counts;
    check_tile_operand(input, "input_tensor");
    check_tile_operand(weight, "weight");
    TT_FATAL(
        weight.device() == input.device() && counts.device() == input.device(),
        "grouped_matmul: weight and expert_counts must be on the device of input_tensor");

    TT_FATAL(
        counts.dtype() == DataType::UINT32 && counts.layout() == Layout::ROW_MAJOR &&
            counts.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "grouped_matmul: expert_counts must be UINT32, ROW_MAJOR and interleaved, as routing_metadata writes it");
    const auto& counts_shape = counts.logical_shape();
    TT_FATAL(
        counts_shape.rank() == 2 && counts_shape[0] == 1,
        "grouped_matmul: expert_counts must be 1 x num_experts, got {}",
        counts_shape);
    const uint32_t num_experts = counts_shape[1];
    TT_FATAL(
        num_experts >= 1 && num_experts <= kMaxExperts,
        "grouped_matmul: num_experts must be 1 to {}, got {}",
        kMaxExperts,
        num_experts);

    // weight is (1, 1, K, E * N), or (1, 1, N, E * K) read transposed: either way expert e owns the
    // e-th E-th of its columns.
    const auto& input_shape = input.logical_shape();
    const auto& weight_shape = weight.logical_shape();
    const uint32_t k = input_shape[3];
    TT_FATAL(
        weight_shape[3] % num_experts == 0,
        "grouped_matmul: the weight's {} columns do not split into {} experts",
        weight_shape[3],
        num_experts);
    const uint32_t expert_columns = weight_shape[3] / num_experts;
    const uint32_t weight_k = attributes.transpose_weight ? expert_columns : weight_shape[2];
    const uint32_t n = attributes.transpose_weight ? weight_shape[2] : expert_columns;
    TT_FATAL(
        weight_k == k,
        "grouped_matmul: input_tensor has K = {} but the weight's experts take K = {}{}",
        k,
        weight_k,
        attributes.transpose_weight ? " (transpose_weight)" : "");
    TT_FATAL(n % kTileSize == 0, "grouped_matmul: each expert's N must be whole tiles, got {}", n);

    TT_FATAL(
        attributes.dtype == DataType::BFLOAT16 || attributes.dtype == DataType::BFLOAT8_B,
        "grouped_matmul: dtype must be BFLOAT16 or BFLOAT8_B, got {}",
        attributes.dtype);
    const bool row_major_output = attributes.output_layout == Layout::ROW_MAJOR;
    TT_FATAL(
        attributes.output_layout == Layout::TILE || row_major_output,
        "grouped_matmul: output_layout must be TILE or ROW_MAJOR, got {}",
        attributes.output_layout);
    TT_FATAL(
        !row_major_output || attributes.dtype == DataType::BFLOAT16,
        "grouped_matmul: a ROW_MAJOR output must be BFLOAT16, got {}",
        attributes.dtype);
    TT_FATAL(
        attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "grouped_matmul: memory_config must be interleaved, got {}",
        attributes.memory_config.memory_layout());
    if (attributes.activation.has_value()) {
        const auto& activation = *attributes.activation;
        const auto params = activation.get_params();
        TT_FATAL(
            activation.type() == ttnn::operations::unary::UnaryOpType::GELU_TANH && params.size() <= 1 &&
                (params.empty() || params[0] == 0.0f || params[0] == 1.0f),
            "grouped_matmul: activation must be GELU_TANH with no parameter, 0 (accurate) or 1 (fast)");
    }

    // Lanes are the core rows and columns split N, as ttnn.matmul's 2D layout without transpose_mcast.
    // Per-expert lanes give each expert a row of its own.
    const auto grid = input.device()->compute_with_storage_grid_size();
    TT_FATAL(
        !attributes.per_expert_lanes || num_experts <= grid.y,
        "grouped_matmul: per_expert_lanes gives each expert a core row, so num_experts must be at most the grid's {} "
        "rows, got {}",
        grid.y,
        num_experts);
    const uint32_t n_tiles = n / kTileSize;
    const uint32_t column_n_tiles = tt::div_up(n_tiles, grid.x);
    const uint32_t num_columns = tt::div_up(n_tiles, column_n_tiles);
    const uint32_t last_column_n_tiles = n_tiles - (num_columns - 1) * column_n_tiles;
    const uint32_t dest_tiles = std::min(kMaxSubblockWidth, get_dest_reg_count(attributes.compute_kernel_config));
    uint32_t subblock_w = dest_tiles;
    while (column_n_tiles % subblock_w != 0 || last_column_n_tiles % subblock_w != 0) {
        --subblock_w;
    }
    // Rows are untilized as whole output-row segments of a core's columns.
    TT_FATAL(
        !row_major_output || last_column_n_tiles == column_n_tiles,
        "grouped_matmul: a ROW_MAJOR output needs N split evenly over the core columns; {} tiles over {} columns "
        "does not",
        n_tiles,
        num_columns);

    return GroupedMatmulLayout{
        .num_experts = num_experts,
        .rows = input_shape[2],
        .k_tiles = k / kTileSize,
        .n_tiles = n_tiles,
        .num_lanes = attributes.per_expert_lanes ? num_experts : static_cast<uint32_t>(grid.y),
        .num_columns = num_columns,
        .column_n_tiles = column_n_tiles,
        .last_column_n_tiles = last_column_n_tiles,
        .subblock_w = subblock_w,
    };
}

void GroupedMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    compute_layout(attributes, tensor_args);
}

GroupedMatmulDeviceOperation::spec_return_value_t GroupedMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;

    const auto layout = compute_layout(attributes, tensor_args);
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, layout.rows, layout.n_tiles * kTileSize}),
        TensorLayout(attributes.dtype, PageConfig(attributes.output_layout), attributes.memory_config));
}

GroupedMatmulDeviceOperation::tensor_return_value_t GroupedMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor grouped_matmul(
    const Tensor& input,
    const Tensor& weight,
    const Tensor& expert_counts,
    tt::tt_metal::DataType dtype,
    bool transpose_weight,
    const std::optional<ttnn::operations::unary::UnaryWithParam>& activation,
    tt::tt_metal::Layout output_layout,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool per_expert_lanes) {
    using Operation = ttnn::experimental::nomic_embed_text_v2_moe::GroupedMatmulDeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{
            .dtype = dtype,
            .transpose_weight = transpose_weight,
            .activation = activation,
            .output_layout = output_layout,
            .memory_config = memory_config,
            .compute_kernel_config = compute_kernel_config,
            .per_expert_lanes = per_expert_lanes},
        Operation::tensor_args_t{.input = input, .weight = weight, .expert_counts = expert_counts});
}

}  // namespace ttnn::prim
