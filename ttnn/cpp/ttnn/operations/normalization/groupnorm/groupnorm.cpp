// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm.hpp"
#include "device/groupnorm_device_operation.hpp"
#include "groupnorm_grid_utils.hpp"
#include "groupnorm_input_mask.hpp"

#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"

namespace {

using ttnn::operations::normalization::compute_num_virtual_cols;
using ttnn::operations::normalization::find_expected_dram_grid;

// Validates that the requested core grid satisfies the DRAM group-norm constraints.
// If the requested grid is invalid, fatals with an error suggesting the largest valid sub-grid.
void validate_dram_grid(const ttnn::CoreGrid& requested, uint32_t W, uint32_t Ht, int num_groups) {
    uint32_t nvc = compute_num_virtual_cols(requested.x, num_groups, W);
    if (nvc > 0) {
        uint32_t rows_per_y = requested.x / nvc;
        if (rows_per_y > 0) {
            uint32_t num_virtual_rows = rows_per_y * requested.y;
            if (Ht >= num_virtual_rows && Ht % num_virtual_rows == 0) {
                // Valid grid found
                return;
            }
        }
    }

    auto suggested = find_expected_dram_grid(requested.x, requested.y, W, num_groups, Ht * ttnn::types::TILE_SIZE);

    // User supplied invalid grid, but suggested a valid grid
    if (suggested.has_value()) {
        TT_THROW(
            "group_norm: Requested core_grid (x={}, y={}) is invalid for the input dimensions "
            "(Ht={}, W={}, num_groups={}). The grid must satisfy "
            "num_virtual_rows = (grid_x / num_virtual_cols) * grid_y <= Ht, and "
            "Ht must be divisible by num_virtual_rows (otherwise remainder tiles are dropped). "
            "The largest valid grid that fits is (x={}, y={}).",
            requested.x,
            requested.y,
            Ht,
            W,
            num_groups,
            suggested->x,
            suggested->y);
    } else {
        TT_THROW(
            "group_norm: Cannot find any valid core grid for the given configuration. "
            "Input height in tiles (Ht={}) is too small for any grid with W={}, num_groups={}. "
            "Requested grid was (x={}, y={}).",
            Ht,
            W,
            num_groups,
            requested.x,
            requested.y);
    }
}

}  // namespace

namespace ttnn::operations::normalization {

ttnn::Tensor get_mask_tensor(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& input_mask,
    const std::optional<ttnn::Tensor>& negative_mask,
    std::optional<CoreGrid> core_grid,
    const int num_groups) {
    ttnn::Tensor mask = input_mask.value_or(ttnn::Tensor());
    if (!input_mask.has_value() and !negative_mask.has_value()) {
        // create input mask
        int64_t num_channel = input_tensor.padded_shape()[-1];
        int64_t num_cores_across_channel;
        if (input_tensor.memory_config().buffer_type() == BufferType::L1) {
            auto mem_layout = input_tensor.memory_config().memory_layout();
            if (mem_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
                num_cores_across_channel = 1;
            } else if (mem_layout == TensorMemoryLayout::BLOCK_SHARDED && core_grid.has_value()) {
                auto shard_spec = input_tensor.memory_config().shard_spec();
                bool is_row_major = shard_spec.has_value() && shard_spec->orientation == ShardOrientation::ROW_MAJOR;
                num_cores_across_channel = is_row_major ? core_grid.value().x : core_grid.value().y;
            } else {
                num_cores_across_channel = core_grid.has_value() ? core_grid.value().y : 1;
            }
        } else {
            uint32_t num_virtual_cols =
                compute_num_virtual_cols(core_grid.value().x, num_groups, static_cast<uint32_t>(num_channel));
            TT_FATAL(
                num_virtual_cols > 0,
                "group_norm: Cannot determine num_virtual_cols for core_grid x={}, num_groups={}, "
                "num_channels={}. num_virtual_cols must satisfy (num_channels / nvc) %% TILE_SIZE == 0 "
                "and num_groups %% nvc == 0.",
                core_grid.value().x,
                num_groups,
                num_channel);
            num_cores_across_channel = static_cast<int64_t>(num_virtual_cols);
        }
        mask = create_group_norm_input_mask(
            num_channel,
            num_groups,
            num_cores_across_channel,
            tt::tt_metal::DataType::BFLOAT16,
            input_tensor.tensor_spec().tile().get_height(),
            input_tensor.tensor_spec().tile().get_width());
        mask = mask.to_device(input_tensor.device());
    }
    return mask;
}

}  // namespace ttnn::operations::normalization

namespace ttnn {

Tensor group_norm(
    const Tensor& input_tensor,
    const int num_groups,
    const float epsilon,
    const std::optional<Tensor>& input_mask,
    const std::optional<Tensor>& weight,
    const std::optional<Tensor>& bias,
    const std::optional<Tensor>& reciprocals,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> /*dtype*/,
    std::optional<CoreGrid> core_grid,
    std::optional<bool> inplace,
    std::optional<Layout> output_layout,
    std::optional<int> num_out_blocks,
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config,
    const std::optional<Tensor>& negative_mask,
    bool use_welford) {
    if (input_tensor.layout() == Layout::TILE and inplace.has_value()) {
        TT_FATAL(
            !inplace.value(),
            "In-place operation not supported: Tile layout requires non-inplace tensors. (inplace={})",
            inplace.value());
    }

    if (output_layout.has_value() and inplace.has_value()) {
        if (output_layout != input_tensor.layout()) {
            TT_FATAL(
                !inplace.value(),
                "In-place operation not allowed: Input and output tensor layouts differ. (input_layout={}, "
                "output_layout={})",
                input_tensor.layout(),
                output_layout.value());
        }
    }

    TT_FATAL(
        core_grid.has_value(),
        "Automatic grid size determination is not supported. Please specify the grid size explicitly.");

    TT_FATAL(
        input_tensor.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
        "Unsupported memory layout: Input tensor cannot be width-sharded.");

    const auto& input_shape = input_tensor.logical_shape();
    TT_FATAL(
        input_shape.rank() == 4, "Invalid tensor shape: Input tensor must have rank 4. (rank={})", input_shape.rank());

    TT_FATAL(
        input_shape[-1] % num_groups == 0,
        "Invalid channel configuration: Number of channels ({}) must be divisible by the number of groups ({}).",
        input_shape[-1],
        num_groups);

    const auto& input_padded_shape = input_tensor.padded_shape();
    const auto nhw = input_padded_shape[0] * input_padded_shape[1] * input_padded_shape[2];
    TT_FATAL(
        (nhw % ttnn::types::TILE_SIZE) == 0,
        "Invalid tensor dimensions: The product of NHW dimensions ({}) must be divisible by the tile size ({}).",
        nhw,
        ttnn::types::TILE_SIZE);

    // For 0V tensors
    if (input_tensor.logical_volume() == 0) [[unlikely]] {
        return ttnn::clone(input_tensor, /*dtype=*/std::nullopt, memory_config, /*compute_kernel_config=*/std::nullopt);
    }

    const std::optional<ttnn::Tensor>& gamma =
        weight.has_value() ? std::optional<ttnn::Tensor>(ttnn::unsqueeze_to_4D(weight.value())) : std::nullopt;
    const std::optional<ttnn::Tensor>& beta =
        bias.has_value() ? std::optional<ttnn::Tensor>(ttnn::unsqueeze_to_4D(bias.value())) : std::nullopt;

    const MemoryConfig& dram_memory_config =
        tt::tt_metal::MemoryConfig{tt::tt_metal::TensorMemoryLayout::INTERLEAVED, tt::tt_metal::BufferType::DRAM};
    const MemoryConfig& output_mem_config = memory_config.value_or(dram_memory_config);

    // Initialize compute kernel config
    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Invalid input tensor storage type: Input tensor must be on device. (storage type={})",
        input_tensor.storage_type());
    const auto arch = input_tensor.device()->arch();
    const auto math_fidelity = MathFidelity::HiFi4;
    const auto approx_mode = true;
    const auto fp32_acc = use_welford;
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, math_fidelity, approx_mode, fp32_acc);

    // For non-sharded DRAM tensors, validate that the requested core grid is not too
    // large for the input spatial dimensions. The constraint is Ht >= num_virtual_rows,
    // where num_virtual_rows = (grid_x / num_virtual_cols) * grid_y and Ht = NHW / TILE_SIZE.
    // Violating this leads to per_core_Mt == 0, causing division-by-zero in kernels.
    if (!input_tensor.is_sharded()) {
        const uint32_t W = input_padded_shape[3];
        const uint32_t Ht = nhw / ttnn::types::TILE_SIZE;
        validate_dram_grid(core_grid.value(), W, Ht, num_groups);
    }

    // auto generate mask tensor if both input_mask and negative_mask are not provided
    ttnn::Tensor mask =
        operations::normalization::get_mask_tensor(input_tensor, input_mask, negative_mask, core_grid, num_groups);

    if (input_tensor.is_sharded()) {
        const ttnn::prim::GroupNormShardedMultiCoreProgramConfig program_config = {
            .compute_with_storage_grid_size = core_grid.value().to_CoreCoord(),
            .im_data_format = DataType::BFLOAT16,
            .out_data_format = DataType::BFLOAT16,
            .inplace = inplace.value_or(false),
            .output_layout = output_layout.value_or(input_tensor.layout())};
        return ttnn::prim::group_norm(
            input_tensor,
            epsilon,
            static_cast<uint32_t>(num_groups),
            output_mem_config,
            program_config,
            kernel_config_val,
            use_welford,
            gamma,
            beta,
            mask,
            negative_mask,
            reciprocals);
    }
    const ttnn::prim::GroupNormMultiCoreProgramConfig program_config = {
        .compute_with_storage_grid_size = core_grid.value().to_CoreCoord(),
        .im_data_format = DataType::BFLOAT16,
        .out_data_format = DataType::BFLOAT16,
        .inplace = inplace.value_or(false),
        .output_layout = output_layout.value_or(input_tensor.layout()),
        .num_out_blocks = num_out_blocks.value_or(1)};
    return ttnn::prim::group_norm(
        input_tensor,
        epsilon,
        static_cast<uint32_t>(num_groups),
        output_mem_config,
        program_config,
        kernel_config_val,
        use_welford,
        gamma,
        beta,
        mask,
        negative_mask,
        reciprocals);
}

}  // namespace ttnn
