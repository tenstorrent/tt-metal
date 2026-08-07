// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <utility>

#include "ttnn/tensor/types.hpp"
#include "permute_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/transpose/device/transpose_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::operations::data_movement {
PermuteDeviceOperation::program_factory_t PermuteDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& dims = operation_attributes.dims;
    if (tensor_args.input_tensor.layout() == Layout::ROW_MAJOR) {
        // Last dimension unchanged — rows can be copied without reordering elements.
        if (dims.back() == tensor_args.input_tensor.logical_shape().rank() - 1) {
            return MultiCoreRowInvariant{};
        }
        // Last dimension moved — need a blocked transpose in RM.
        return MultiCoreBlockedGeneric{};
    }
    // Tiled layout selection.
    uint32_t rank = tensor_args.input_tensor.logical_shape().rank();
    // Tile dims stay in the last two positions (identity or WH swap).
    if ((dims[rank - 1] == rank - 1 && dims[rank - 2] == rank - 2) ||
        (dims[rank - 1] == rank - 2 && dims[rank - 2] == rank - 1)) {
        return MultiCoreTileInvariant{};
    }
    // Only one tile dimension is moved out.
    if (dims[rank - 1] == rank - 1 || dims[rank - 1] == rank - 2) {
        return MultiCoreTileRowInvariant{};
    }
    // Both tile dimensions are moved.
    return MultiCoreTiledGeneric{};
}

void PermuteDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    TT_FATAL(
        attributes.dims.size() == tensor_args.input_tensor.logical_shape().rank(),
        "Permute dimensions must match input tensor rank");
}

void PermuteDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/) {}

PermuteDeviceOperation::spec_return_value_t PermuteDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input_tensor;
    auto input_shape = input_tensor.logical_shape();

    ttsl::SmallVector<uint32_t> output_shape_vec(attributes.dims.size());
    std::transform(attributes.dims.begin(), attributes.dims.end(), output_shape_vec.begin(), [&](auto dim) {
        return input_shape[dim];
    });
    auto output_shape = Shape(std::move(output_shape_vec));

    auto output_mem_config = attributes.output_mem_config;

    // Derive shard_spec when sharded output lacks one.
    if (output_mem_config.is_sharded() && !output_mem_config.shard_spec().has_value()) {
        ttsl::SmallVector<uint32_t> output_padded_vec(output_shape.view().begin(), output_shape.view().end());
        if (input_tensor.layout() == Layout::TILE && output_shape.rank() >= 2) {
            const auto& tile = input_tensor.tensor_spec().tile();
            auto r = output_shape.rank();
            output_padded_vec[r - 2] = tt::round_up(output_padded_vec[r - 2], tile.get_height());
            output_padded_vec[r - 1] = tt::round_up(output_padded_vec[r - 1], tile.get_width());
        }
        auto output_padded_shape = Shape(std::move(output_padded_vec));

        // Adapt input shard spec only when input/output share the same shard layout.
        bool derived = false;
        if (input_tensor.is_sharded() && input_tensor.shard_spec().has_value() &&
            input_tensor.memory_config().memory_layout() == output_mem_config.memory_layout()) {
            const auto& from_shape = input_tensor.padded_shape();
            auto vol_except_last = [](const auto& shape) {
                return std::accumulate(shape.cbegin(), shape.cend() - 1, uint64_t{1}, std::multiplies<uint64_t>());
            };
            uint64_t from_vol = vol_except_last(from_shape);
            uint64_t to_vol = vol_except_last(output_padded_shape);
            uint64_t from_w = from_shape[-1], to_w = output_padded_shape[-1];
            auto shard = input_tensor.shard_spec()->shape;
            uint64_t h_num = static_cast<uint64_t>(shard[0]) * to_vol;
            uint64_t w_num = static_cast<uint64_t>(shard[1]) * to_w;
            if (from_vol > 0 && from_w > 0 && h_num % from_vol == 0 && w_num % from_w == 0) {
                auto adjusted =
                    transpose::adjust_shard_spec_to_shape(*input_tensor.shard_spec(), from_shape, output_padded_shape);
                if (adjusted.has_value()) {
                    const bool tile_layout = input_tensor.layout() == Layout::TILE;
                    const bool tile_aligned = adjusted->shape[0] % tt::constants::TILE_HEIGHT == 0 &&
                                              adjusted->shape[1] % tt::constants::TILE_WIDTH == 0;
                    if (!tile_layout || tile_aligned) {
                        output_mem_config = MemoryConfig(
                            output_mem_config.memory_layout(), output_mem_config.buffer_type(), std::move(adjusted));
                        derived = true;
                    }
                }
            }
        }
        if (!derived) {
            // Generate a fresh shard spec for the permuted shape.
            auto shard_spec = transpose::generate_transpose_shard_spec(
                input_tensor, output_padded_shape, output_mem_config.memory_layout());
            output_mem_config =
                MemoryConfig(output_mem_config.memory_layout(), output_mem_config.buffer_type(), shard_spec);
        }
    }

    return tt::tt_metal::TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(input_tensor.layout()), output_mem_config));
}

tt::tt_metal::operation::OpPerformanceModelGeneral<PermuteDeviceOperation::tensor_return_value_t>
PermuteDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*op_attr*/, const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output, false, 0, true);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

PermuteDeviceOperation::tensor_return_value_t PermuteDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.optional_output_tensor.has_value()) {
        return tensor_args.optional_output_tensor.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), tensor_args.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteDeviceOperation::tensor_return_value_t permute(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    float pad_value) {
    using OperationType = ttnn::operations::data_movement::PermuteDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dims = dims,
            .output_mem_config = memory_config.value_or(input_tensor.memory_config()),
            .pad_value = pad_value},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)});
}
}  // namespace ttnn::prim
