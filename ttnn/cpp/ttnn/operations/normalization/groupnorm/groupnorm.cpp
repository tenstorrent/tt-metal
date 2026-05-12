// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
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
void validate_dram_grid(
    const ttnn::CoreGrid& requested, uint32_t W, uint32_t Ht, int num_groups, uint32_t num_batches) {
    uint32_t nvc = compute_num_virtual_cols(requested.x, num_groups, W);
    if (nvc > 0) {
        uint32_t rows_per_y = requested.x / nvc;
        if (rows_per_y > 0) {
            uint32_t num_virtual_rows = rows_per_y * requested.y;
            if (Ht >= num_virtual_rows && Ht % num_virtual_rows == 0 &&
                (num_virtual_rows < num_batches || num_virtual_rows % num_batches == 0)) {
                return;
            }
        }
    }

    auto suggested =
        find_expected_dram_grid(requested.x, requested.y, W, num_groups, Ht * ttnn::types::TILE_SIZE, num_batches);

    uint32_t nvc_for_msg = compute_num_virtual_cols(requested.x, num_groups, W);
    uint32_t nvr_for_msg = nvc_for_msg > 0 ? (requested.x / nvc_for_msg) * requested.y : 0;

    // User supplied invalid grid, but suggested a valid grid
    if (suggested.has_value()) {
        TT_THROW(
            "group_norm: Requested core_grid (x={}, y={}) is invalid for the input dimensions "
            "(Ht={}, W={}, num_groups={}, num_batches={}). The grid must satisfy: "
            "num_virtual_rows = (grid_x / num_virtual_cols) * grid_y <= Ht, "
            "Ht must be divisible by num_virtual_rows, and "
            "num_virtual_rows must be divisible by num_batches (for uniform multicast groups). "
            "Computed num_virtual_rows={}, num_virtual_cols={}. "
            "The largest valid grid that fits is (x={}, y={}).",
            requested.x,
            requested.y,
            Ht,
            W,
            num_groups,
            num_batches,
            nvr_for_msg,
            nvc_for_msg,
            suggested->x,
            suggested->y);
    } else {
        TT_THROW(
            "group_norm: Cannot find any valid core grid for the given configuration. "
            "Input height in tiles (Ht={}) is too small for any grid with W={}, num_groups={}, "
            "num_batches={}. Computed num_virtual_rows={}, num_virtual_cols={}. "
            "Requested grid was (x={}, y={}).",
            Ht,
            W,
            num_groups,
            num_batches,
            nvr_for_msg,
            nvc_for_msg,
            requested.x,
            requested.y);
    }
}

int64_t get_group_norm_cores_across_channel(
    tt::tt_metal::TensorMemoryLayout memory_layout,
    const ttnn::CoreGrid& core_grid,
    const std::optional<tt::tt_metal::ShardOrientation>& shard_orientation) {
    using tt::tt_metal::ShardOrientation;
    using tt::tt_metal::TensorMemoryLayout;
    if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        if (shard_orientation == ShardOrientation::ROW_MAJOR) {
            return static_cast<int64_t>(core_grid.x);
        }
        return static_cast<int64_t>(core_grid.y);
    }
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        return 1;
    }
    return static_cast<int64_t>(core_grid.x * core_grid.y);
}

}  // namespace

namespace ttnn::operations::normalization {

ttnn::Tensor get_mask_tensor(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& input_mask,
    const std::optional<ttnn::Tensor>& negative_mask,
    const CoreGrid& core_grid,
    const int num_groups) {
    ttnn::Tensor mask = input_mask.value_or(ttnn::Tensor());
    if (!input_mask.has_value() and !negative_mask.has_value()) {
        // create input mask
        int64_t num_channel = input_tensor.padded_shape()[-1];
        int64_t num_cores_across_channel;
        if (input_tensor.memory_config().buffer_type() == BufferType::L1) {
            const auto mem_layout = input_tensor.memory_config().memory_layout();
            const auto shard_spec = input_tensor.memory_config().shard_spec();
            std::optional<ShardOrientation> shard_orientation = std::nullopt;
            if (shard_spec.has_value()) {
                shard_orientation = shard_spec->orientation;
            }
            num_cores_across_channel = get_group_norm_cores_across_channel(mem_layout, core_grid, shard_orientation);
        } else {
            uint32_t num_virtual_cols =
                compute_num_virtual_cols(core_grid.x, num_groups, static_cast<uint32_t>(num_channel));
            TT_FATAL(
                num_virtual_cols > 0,
                "group_norm: Cannot determine num_virtual_cols for core_grid x={}, num_groups={}, "
                "num_channels={}. num_virtual_cols must satisfy (num_channels / nvc) % TILE_SIZE == 0 "
                "and num_groups % nvc == 0.",
                core_grid.x,
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
    const auto math_fidelity = tt::tt_metal::MathFidelity::HiFi4;
    const auto approx_mode = true;
    const auto fp32_acc = use_welford;
    auto kernel_config_val =
        init_device_compute_kernel_config(arch, compute_kernel_config, math_fidelity, approx_mode, fp32_acc);

    // Reciprocals must be sharded to L1 via the legacy ShardSpec representation:
    // the program factory reads shard_spec().value().numel() as the compile-time
    // reciprocal_size and binds the cb_reciprocals CB to the per-bank addresses
    // of the buffer. Interleaved, DRAM-sharded, or NdShardSpec reciprocals would
    // either trip bad_optional_access or violate the per-core-L1-bank assumption
    // downstream.
    if (reciprocals.has_value()) {
        TT_FATAL(
            reciprocals->is_sharded() && reciprocals->memory_config().buffer_type() == BufferType::L1,
            "group_norm: reciprocals tensor must be sharded to L1 (got is_sharded={}, buffer_type={}); "
            "interleaved or DRAM-sharded reciprocals are not supported.",
            reciprocals->is_sharded(),
            reciprocals->memory_config().buffer_type());
        TT_FATAL(
            reciprocals->shard_spec().has_value(),
            "group_norm: reciprocals tensor must use the legacy ShardSpec "
            "representation (NdShardSpec sharding is not currently supported).");
    }

    const bool core_grid_auto_selected = !core_grid.has_value();

    if (core_grid_auto_selected) {
        if (input_tensor.is_sharded()) {
            const auto& shard_spec_opt = input_tensor.shard_spec();
            TT_FATAL(
                shard_spec_opt.has_value(),
                "group_norm: Sharded input must have a shard spec when core_grid is not provided.");
            // Derive the grid directly from the tensor's existing shard layout
            // rather than recomputing from scratch, so that program_config's
            // grid_size matches the cores where kernels are actually placed.
            const auto bbox = shard_spec_opt->grid.bounding_box();
            core_grid = ttnn::CoreGrid(bbox.end_coord.x + 1, bbox.end_coord.y + 1);
        } else if (reciprocals.has_value() && reciprocals->is_sharded()) {
            // The reciprocals LUT is sharded on a specific grid; its length
            // encodes num_virtual_rows which must match the compute grid.
            // Infer the grid from the reciprocals tensor so the kernel sees a
            // consistent LUT.
            const auto bbox = reciprocals->shard_spec()->grid.bounding_box();
            core_grid = ttnn::CoreGrid(bbox.end_coord.x + 1, bbox.end_coord.y + 1);
        } else {
            const auto dev_grid = input_tensor.device()->compute_with_storage_grid_size();
            auto dram_grid = ttnn::operations::normalization::find_expected_dram_grid(
                dev_grid.x, dev_grid.y, input_padded_shape[3], num_groups, nhw, input_padded_shape[0]);
            TT_FATAL(
                dram_grid.has_value(),
                "group_norm: Could not determine a valid DRAM core grid for channels={}, num_groups={}, nhw={}. "
                "Specify core_grid explicitly or adjust configuration.",
                input_padded_shape[3],
                num_groups,
                nhw);
            core_grid = dram_grid.value();
        }
    }

    // num_out_blocks only affects the non-sharded (interleaved/DRAM) program factory
    // (as stated in docstring in nanobind documents), so the assert that "auto-grid implies
    // no explicit chunking" rule only applies to the non-sharded path.
    TT_FATAL(
        (input_tensor.is_sharded() || !(core_grid_auto_selected && num_out_blocks.has_value())),
        "group_norm: num_out_blocks cannot be specified when core_grid is auto-selected. "
        "Either provide an explicit core_grid or omit num_out_blocks.");

    if (!input_tensor.is_sharded() && num_out_blocks.has_value()) {
        // Reject obviously-out-of-range values of num_out_blocks up front, but only for
        // non-sharded inputs, where num_out_blocks actually has effect.
        // Accepted user-facing values are -1 (use the program-factory's auto-heuristic)
        // or an explicit chunk count in [1, block_h]; the upper bound and >= 1 are checked
        // later against the resolved block_h.
        // Catching < -1 here prevents the value, which gets reinterpreted as a huge
        // uint32_t at some point, from producing a confusing downstream error.
        TT_FATAL(
            *num_out_blocks >= -1,
            "group_norm: num_out_blocks ({}) is invalid. Use -1 to request the auto-heuristic, "
            "or an explicit chunk count in [1, block_h].",
            *num_out_blocks);
    }

    // The per-shard numel (consumed by the compute kernel as the compile-time
    // `reciprocal_size`) and the per-bank addresses bound to the reciprocals CB
    // are baked for a specific grid. The compute kernel runs on `core_grid`,
    // so the reciprocals must be sharded on that same grid; otherwise the LUT
    // is the wrong length and/or lives on the wrong banks. This covers all
    // three paths to picking core_grid (sharded input, reciprocals inference,
    // and an explicit user-provided core_grid).
    if (reciprocals.has_value()) {
        // Precondition above guarantees is_sharded() and shard_spec().has_value()
        // whenever reciprocals is provided.
        const auto recip_bbox = reciprocals->shard_spec()->grid.bounding_box();
        const uint32_t recip_x = recip_bbox.end_coord.x + 1;
        const uint32_t recip_y = recip_bbox.end_coord.y + 1;
        TT_FATAL(
            recip_x == core_grid->x && recip_y == core_grid->y,
            "group_norm: reciprocals shard grid (x={}, y={}) must match the compute core_grid "
            "(x={}, y={}). The reciprocals LUT length and per-bank addresses are baked for a "
            "specific grid; running the kernel on a different grid will read past the LUT or "
            "use unallocated banks.",
            recip_x,
            recip_y,
            core_grid->x,
            core_grid->y);
    }

    // For non-sharded DRAM tensors, validate that the requested core grid is not too
    // large for the input spatial dimensions. The constraint is Ht >= num_virtual_rows,
    // where num_virtual_rows = (grid_x / num_virtual_cols) * grid_y and Ht = NHW / TILE_SIZE.
    // Violating this leads to per_core_Mt == 0, causing division-by-zero in kernels.
    if (!input_tensor.is_sharded()) {
        const uint32_t W = input_padded_shape[3];
        const uint32_t Ht = nhw / ttnn::types::TILE_SIZE;
        validate_dram_grid(core_grid.value(), W, Ht, num_groups, input_padded_shape[0]);
    }

    // auto generate mask tensor if both input_mask and negative_mask are not provided
    ttnn::Tensor mask = operations::normalization::get_mask_tensor(
        input_tensor, input_mask, negative_mask, core_grid.value(), num_groups);

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
    // When the user did not pin a core grid, defer num_out_blocks to the program
    // factory's heuristic via the -1 sentinel (see GroupNormMultiCoreProgramConfig).
    // Otherwise honor the explicit num_out_blocks (defaulting to 1 = no chunking).
    const ttnn::prim::GroupNormMultiCoreProgramConfig program_config = {
        .compute_with_storage_grid_size = core_grid.value().to_CoreCoord(),
        .im_data_format = DataType::BFLOAT16,
        .out_data_format = DataType::BFLOAT16,
        .inplace = inplace.value_or(false),
        .output_layout = output_layout.value_or(input_tensor.layout()),
        .num_out_blocks = core_grid_auto_selected ? -1 : num_out_blocks.value_or(1)};
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
