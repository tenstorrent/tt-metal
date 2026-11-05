// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "routed_combine_device_operation.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
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
constexpr uint32_t kMaxTopK = 16;
// Chunk widths to try, widest first: a unit's rows are gathered and tilized a chunk at a time.
constexpr std::array<uint32_t, 6> kChunkTiles = {8, 6, 4, 3, 2, 1};
// A unit's fixed cost (its index, weight and bias reads, the tilize and multiply inits), in output
// tiles; it keeps the narrowest chunks from winning on many tiny units.
constexpr uint32_t kUnitOverheadTiles = 2;

// The framework rejects host tensors before this runs.
void check_operand(const Tensor& tensor, std::string_view name, DataType dtype, Layout layout) {
    TT_FATAL(tensor.dtype() == dtype, "routed_combine: {} must be {}, got {}", name, dtype, tensor.dtype());
    TT_FATAL(tensor.layout() == layout, "routed_combine: {} must be {} layout, got {}", name, layout, tensor.layout());
    TT_FATAL(
        tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "routed_combine: {} must be interleaved, got {}",
        name,
        tensor.memory_config().memory_layout());
}

}  // namespace

RoutedCombineLayout RoutedCombineDeviceOperation::compute_layout(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& rows = tensor_args.expert_rows;
    const auto& index = tensor_args.combine_index;
    const auto& weights = tensor_args.combine_weights;
    const auto& bias = tensor_args.bias;
    check_operand(rows, "expert_rows", DataType::BFLOAT16, Layout::ROW_MAJOR);
    check_operand(index, "combine_index", DataType::UINT32, Layout::ROW_MAJOR);
    check_operand(weights, "combine_weights", DataType::FLOAT32, Layout::TILE);
    check_operand(bias, "bias", DataType::BFLOAT16, Layout::TILE);
    for (const Tensor* tensor : {&index, &weights, &bias}) {
        TT_FATAL(tensor->device() == rows.device(), "routed_combine: every input must be on the device of expert_rows");
    }

    const auto& rows_shape = rows.logical_shape();
    TT_FATAL(
        rows_shape.rank() == 4 && rows_shape[0] == 1 && rows_shape[1] == 1,
        "routed_combine: expert_rows must be 1 x 1 x rows x hidden, got {}",
        rows_shape);
    const uint32_t hidden = rows_shape[3];
    TT_FATAL(hidden % kTileSize == 0, "routed_combine: hidden must be whole tiles, got {}", hidden);

    const auto& index_shape = index.logical_shape();
    TT_FATAL(
        index_shape.rank() == 2, "routed_combine: combine_index must be top_k x padded_tokens, got {}", index_shape);
    const uint32_t top_k = index_shape[0];
    const uint32_t padded_tokens = index_shape[1];
    TT_FATAL(top_k >= 1 && top_k <= kMaxTopK, "routed_combine: top_k must be 1 to {}, got {}", kMaxTopK, top_k);
    TT_FATAL(
        attributes.num_tokens >= 1 && tt::round_up(attributes.num_tokens, kTileSize) == padded_tokens,
        "routed_combine: num_tokens {} does not match combine_index's {} padded tokens",
        attributes.num_tokens,
        padded_tokens);
    // routing_metadata gives every slot a row, so fewer rows cannot be its layout. The index values
    // themselves stay on the device.
    const uint32_t min_rows = tt::round_up(top_k * attributes.num_tokens, kTileSize);
    TT_FATAL(
        rows_shape[2] >= min_rows,
        "routed_combine: expert_rows has {} rows, fewer than the {} that {} tokens at top_k {} fill",
        rows_shape[2],
        min_rows,
        attributes.num_tokens,
        top_k);
    TT_FATAL(
        weights.logical_shape() == ttnn::Shape({1, top_k, padded_tokens, 1}),
        "routed_combine: combine_weights must be 1 x {} x {} x 1, got {}",
        top_k,
        padded_tokens,
        weights.logical_shape());
    TT_FATAL(
        bias.logical_shape() == ttnn::Shape({1, 1, 1, hidden}),
        "routed_combine: bias must be 1 x 1 x 1 x {}, got {}",
        hidden,
        bias.logical_shape());
    TT_FATAL(
        attributes.memory_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "routed_combine: memory_config must be interleaved, got {}",
        attributes.memory_config.memory_layout());
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(rows.device()->arch(), attributes.compute_kernel_config);
    TT_FATAL(fp32_dest_acc_en, "routed_combine: compute_kernel_config must enable fp32_dest_acc_en");

    // The chunk width that leaves the busiest core the least work, the widest of equals.
    const auto grid = rows.device()->compute_with_storage_grid_size();
    const uint32_t grid_cores = grid.x * grid.y;
    const uint32_t token_blocks = padded_tokens / kTileSize;
    const uint32_t hidden_tiles = hidden / kTileSize;
    uint32_t chunk_tiles = 1;
    uint32_t best_cost = UINT32_MAX;
    for (uint32_t width : kChunkTiles) {
        if (hidden_tiles % width != 0) {
            continue;
        }
        const uint32_t units_per_core = tt::div_up(token_blocks * (hidden_tiles / width), grid_cores);
        const uint32_t cost = units_per_core * (width + kUnitOverheadTiles);
        if (cost < best_cost) {
            best_cost = cost;
            chunk_tiles = width;
        }
    }
    const uint32_t num_units = token_blocks * (hidden_tiles / chunk_tiles);
    return RoutedCombineLayout{
        .top_k = top_k,
        .padded_tokens = padded_tokens,
        .hidden_tiles = hidden_tiles,
        .chunk_tiles = chunk_tiles,
        .num_units = num_units,
        .num_cores = std::min(num_units, grid_cores),
    };
}

void RoutedCombineDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    compute_layout(attributes, tensor_args);
}

RoutedCombineDeviceOperation::spec_return_value_t RoutedCombineDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;

    compute_layout(attributes, tensor_args);
    const uint32_t hidden = tensor_args.expert_rows.logical_shape()[3];
    return tt::tt_metal::TensorSpec(
        ttnn::Shape({1, 1, attributes.num_tokens, hidden}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::TILE), attributes.memory_config));
}

RoutedCombineDeviceOperation::tensor_return_value_t RoutedCombineDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.expert_rows.device());
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor routed_combine(
    const Tensor& expert_rows,
    const Tensor& combine_index,
    const Tensor& combine_weights,
    const Tensor& bias,
    uint32_t num_tokens,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config) {
    using Operation = ttnn::experimental::nomic_embed_text_v2_moe::RoutedCombineDeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{
            .num_tokens = num_tokens, .memory_config = memory_config, .compute_kernel_config = compute_kernel_config},
        Operation::tensor_args_t{
            .expert_rows = expert_rows,
            .combine_index = combine_index,
            .combine_weights = combine_weights,
            .bias = bias});
}

}  // namespace ttnn::prim
