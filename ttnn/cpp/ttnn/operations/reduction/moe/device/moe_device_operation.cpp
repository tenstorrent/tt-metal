// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/moe/device/moe_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <optional>

#include "ttnn/operations/reduction/moe/device/moe_device_operation_types.hpp"
#include "ttnn/operations/reduction/moe/device/moe_program_factory.hpp"
#include <tt-metalium/work_split.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {
void MoeDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& expert_mask_tensor = tensor_args.expert_mask;
    const auto& topk_mask_tensor = tensor_args.topk_mask;

    auto input_shape = input_tensor.padded_shape();
    auto topk_shape = topk_mask_tensor.padded_shape();
    auto expert_shape = expert_mask_tensor.padded_shape();
    auto input_logical_shape = input_tensor.logical_shape();
    auto topk_logical_shape = topk_mask_tensor.logical_shape();
    auto expert_logical_shape = expert_mask_tensor.logical_shape();

    TT_FATAL(input_shape.rank() == 4, "Input shape must be 4D, got {}", input_shape.rank());
    TT_FATAL(args.k == 32, "K must be equal to 32, pad with -infinity if necessary to get 32, got {}", args.k);

    TT_FATAL(
        input_shape[-1] >= 64,
        "Input shape inner dim {} must be a multiple of 64, pad with -infinity if necessary",
        input_shape[-1]);
    TT_FATAL(
        (input_shape[-1] & (input_shape[-1] - 1)) == 0,
        "Input shape inner dim {} must be a power of 2, pad with -infinity if necessary",
        input_shape[-1]);
    TT_FATAL(
        (input_shape[0] * input_shape[1] * input_shape[2]) % 32 == 0,
        "Input height (combined input_shape[0-2]) {} must be a multiple of 32",
        input_shape[0] * input_shape[1] * input_shape[2]);

    TT_FATAL(args.output_memory_config.is_sharded() == false, "Sharded implementation not supported yet");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "The input must be in tiled format");

    auto is_row_broadcastable_mask = [](const auto& shape, uint32_t expected_last_dim) {
        if (shape.rank() == 0 || shape[-1] != expected_last_dim) {
            return false;
        }
        for (uint32_t d = 0; d + 1 < shape.rank(); ++d) {
            if (shape[d] != 1) {
                return false;
            }
        }
        return true;
    };

    TT_FATAL(
        is_row_broadcastable_mask(topk_logical_shape, args.k),
        "Topk mask must be row-broadcastable with last dim == k. Got rank={} and shape={} for k={}",
        topk_logical_shape.rank(),
        topk_logical_shape,
        args.k);
    TT_FATAL(
        is_row_broadcastable_mask(expert_logical_shape, input_logical_shape[-1]),
        "Expert mask must be row-broadcastable with last dim == input_shape[-1]. Got rank={} and shape={} for "
        "input_shape[-1]={}",
        expert_logical_shape.rank(),
        expert_logical_shape,
        input_logical_shape[-1]);
    TT_FATAL(topk_shape[-2] == 32, "Topk shape inner dim must be padded to 32, got {}", topk_shape[-2]);
    TT_FATAL(expert_shape[-2] == 32, "Expert shape inner dim must be padded to 32, got {}", expert_shape[-2]);
        TT_FATAL(expert_shape[-1] == input_shape[-1],
        "Expert shape inner dim must be equal to input_shape[-1], got {}",
        expert_shape[-1]);
    TT_FATAL(topk_shape[-2] == 32, "Topk shape inner dim must be equal to 32, got {}", topk_shape[-2]);
    TT_FATAL(expert_shape[-2] == 32, "Expert shape inner dim must be equal to 32, got {}", expert_shape[-2]);

    {
        auto validate_tensor_padded_spatial = [](const Tensor& t, const char* tensor_name) {
            const auto& padded_shape = t.padded_shape();
            const uint32_t tile_height = t.tensor_spec().tile().get_height();
            const uint32_t tile_width = t.tensor_spec().tile().get_width();
            TT_FATAL(
                padded_shape.rank() >= 2,
                "MoE {} padded_shape rank {} must be at least 2 for H/W checks",
                tensor_name,
                padded_shape.rank());
            TT_FATAL(
                padded_shape[-2] > 0 && padded_shape[-1] > 0,
                "MoE {} padded spatial dims must be positive: height={}, width={}",
                tensor_name,
                padded_shape[-2],
                padded_shape[-1]);
            TT_FATAL(
                padded_shape[-2] % tile_height == 0,
                "MoE {} padded_height={} must be tile-height-aligned ({})",
                tensor_name,
                padded_shape[-2],
                tile_height);
            TT_FATAL(
                padded_shape[-1] % tile_width == 0,
                "MoE {} padded_width={} must be tile-width-aligned ({})",
                tensor_name,
                padded_shape[-1],
                tile_width);
        };

        validate_tensor_padded_spatial(input_tensor, "input");
        validate_tensor_padded_spatial(expert_mask_tensor, "expert_mask");
        validate_tensor_padded_spatial(topk_mask_tensor, "topk_mask");
        if (tensor_args.preallocated_output.has_value()) {
            validate_tensor_padded_spatial(tensor_args.preallocated_output.value(), "preallocated_output");
        }
    }

    {
        const int32_t logical_rank = input_tensor.logical_shape().rank();
        TT_FATAL(logical_rank > 0, "MoE requires positive logical rank, got {}", logical_rank);
        constexpr int32_t fixed_dim_negative = -1;
        const int32_t fixed_dim_normalized = fixed_dim_negative + logical_rank;
        TT_FATAL(
            fixed_dim_normalized >= 0 && fixed_dim_normalized < logical_rank,
            "MoE fixed reduction dim {} normalized to {} is out of range for logical rank {}",
            fixed_dim_negative,
            fixed_dim_normalized,
            logical_rank);
    }

    {
        const auto device_grid_size = input_tensor.device()->compute_with_storage_grid_size();
        TT_FATAL(
            device_grid_size.x > 0 && device_grid_size.y > 0,
            "MoE requires non-empty device compute grid, got ({}, {})",
            device_grid_size.x,
            device_grid_size.y);
        const CoreRangeSet device_grid =
            num_cores_to_corerangeset(device_grid_size.x * device_grid_size.y, device_grid_size, false);
        const CoreRange moe_program_core({0, 0}, {0, 0});
        const CoreRangeSet moe_program_grid({moe_program_core});
        TT_FATAL(
            device_grid.contains(moe_program_grid),
            "MoE program core grid {} must be contained in device grid {}",
            moe_program_grid,
            device_grid);

        auto validate_sharded_operand = [&](const Tensor& t, const char* tensor_name) {
            const auto& tensor_memory_config = t.memory_config();
            const uint32_t tile_height = t.tensor_spec().tile().get_height();
            const uint32_t tile_width = t.tensor_spec().tile().get_width();
            if (tensor_memory_config.shard_spec().has_value()) {
                const auto& shard_grid = tensor_memory_config.shard_spec().value().grid;
                const auto& shard_shape = tensor_memory_config.shard_spec().value().shape;
                TT_FATAL(
                    shard_shape[0] > 0 && shard_shape[1] > 0,
                    "MoE {} shard_shape must be positive, got [{}, {}]",
                    tensor_name,
                    shard_shape[0],
                    shard_shape[1]);
                TT_FATAL(
                    shard_shape[0] % tile_height == 0,
                    "MoE {} shard_shape[0]={} must be tile-height-aligned ({})",
                    tensor_name,
                    shard_shape[0],
                    tile_height);
                TT_FATAL(
                    shard_shape[1] % tile_width == 0,
                    "MoE {} shard_shape[1]={} must be tile-width-aligned ({})",
                    tensor_name,
                    shard_shape[1],
                    tile_width);
                TT_FATAL(
                    device_grid.contains(shard_grid),
                    "MoE {} shard grid {} must be contained in device grid {}",
                    tensor_name,
                    shard_grid,
                    device_grid);
                TT_FATAL(
                    moe_program_grid.contains(shard_grid),
                    "MoE {} shard grid {} must be contained in single-core program grid {}",
                    tensor_name,
                    shard_grid,
                    moe_program_grid);
            }
            if (tensor_memory_config.nd_shard_spec().has_value()) {
                const auto& nd_shard_shape = tensor_memory_config.nd_shard_spec().value().shard_shape;
                const auto& nd_grid = tensor_memory_config.nd_shard_spec().value().grid;
                if (nd_shard_shape.rank() >= 2) {
                    TT_FATAL(
                        nd_shard_shape[-2] > 0 && nd_shard_shape[-1] > 0,
                        "MoE {} ND shard last-2 dims must be positive, got [..., {}, {}]",
                        tensor_name,
                        nd_shard_shape[-2],
                        nd_shard_shape[-1]);
                    TT_FATAL(
                        nd_shard_shape[-2] % tile_height == 0,
                        "MoE {} ND shard_shape[-2]={} must be tile-height-aligned ({})",
                        tensor_name,
                        nd_shard_shape[-2],
                        tile_height);
                    TT_FATAL(
                        nd_shard_shape[-1] % tile_width == 0,
                        "MoE {} ND shard_shape[-1]={} must be tile-width-aligned ({})",
                        tensor_name,
                        nd_shard_shape[-1],
                        tile_width);
                }
                TT_FATAL(
                    device_grid.contains(nd_grid),
                    "MoE {} ND shard grid {} must be contained in device grid {}",
                    tensor_name,
                    nd_grid,
                    device_grid);
                TT_FATAL(
                    moe_program_grid.contains(nd_grid),
                    "MoE {} ND shard grid {} must be contained in single-core program grid {}",
                    tensor_name,
                    nd_grid,
                    moe_program_grid);
            }
        };

        validate_sharded_operand(input_tensor, "input");
        validate_sharded_operand(expert_mask_tensor, "expert_mask");
        validate_sharded_operand(topk_mask_tensor, "topk_mask");
        if (tensor_args.preallocated_output.has_value()) {
            validate_sharded_operand(tensor_args.preallocated_output.value(), "preallocated_output");
        }
    }
    TT_FATAL(
        is_row_broadcastable_mask(expert_logical_shape, input_logical_shape[-1]),
        "Expert mask must be row-broadcastable with last dim == input_shape[-1]. Got rank={} and shape={} for "
        "input_shape[-1]={}",
        expert_logical_shape.rank(),
        expert_logical_shape,
        input_logical_shape[-1]);
    TT_FATAL(topk_shape[-2] == 32, "Topk shape inner dim must be padded to 32, got {}", topk_shape[-2]);
    TT_FATAL(expert_shape[-2] == 32, "Expert shape inner dim must be padded to 32, got {}", expert_shape[-2]);
}

TensorSpec MoeDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input;
    auto output_shape = input_tensor.logical_shape();
    output_shape[-1] = 1;
    return TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), args.output_memory_config));
}

Tensor MoeDeviceOperation::create_output_tensors(const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}
ttnn::Tensor moe(
    const Tensor& input_tensor,
    const Tensor& expert_mask_tensor,
    const Tensor& topk_mask_tensor,
    uint16_t k,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config,
    const std::optional<Tensor>& preallocated_output_tensor) {
    return ttnn::device_operation::launch<MoeDeviceOperation>(
        MoeParams{
            .k = k,
            .output_memory_config = memory_config.value_or(tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG)},
        MoeInputs{
            .input = input_tensor,
            .expert_mask = expert_mask_tensor,
            .topk_mask = topk_mask_tensor,
            .preallocated_output = preallocated_output_tensor});
}

}  // namespace ttnn::prim
