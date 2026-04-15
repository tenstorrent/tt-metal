// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_utils.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::data_movement::transpose {

using namespace tt::tt_metal;

static const std::optional<ShardSpec>& get_shard_spec_from_tensor_spec(const TensorSpec& tensor_spec) {
    return tensor_spec.memory_config().shard_spec();
}

bool is_uneven(const TensorSpec& t) {
    if (!t.memory_config().is_sharded()) {
        return false;
    }
    const auto& shard_spec = get_shard_spec_from_tensor_spec(t);
    if (!shard_spec.has_value()) {
        return true;
    }
    const auto& shape = t.padded_shape();
    const auto& shard = shard_spec->shape;
    const auto rank = shape.rank();
    TT_FATAL(rank >= 2, "Rank must be at least 2");
    uint64_t volume_except_last = 1;
    for (int i = 0; i < static_cast<int>(rank) - 1; ++i) {
        volume_except_last *= shape[i];
    }
    return (volume_except_last % shard[0]) != 0 || (shape[-1] % shard[1]) != 0;
}

bool is_native_transpose_sharding(const TensorSpec& input_spec, const MemoryConfig& output_memory_config) {
    if (!output_memory_config.is_sharded()) {
        return false;
    }
    if (!input_spec.memory_config().is_sharded()) {
        return false;
    }
    if (is_uneven(input_spec)) {
        return false;
    }
    if (input_spec.memory_config().buffer_type() == BufferType::DRAM ||
        output_memory_config.buffer_type() == BufferType::DRAM) {
        return false;
    }
    if (input_spec.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output_memory_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        return false;
    }
    if (output_memory_config.shard_spec().has_value() && input_spec.memory_config().shard_spec().has_value()) {
        const auto& in_grid = input_spec.memory_config().shard_spec()->grid;
        const auto& out_grid = output_memory_config.shard_spec()->grid;
        if (in_grid != out_grid) {
            return false;
        }
    }
    return true;
}

std::optional<TransposeShardSpecs> get_transpose_shard_specs(
    const TensorSpec& input_spec, const TensorSpec& output_spec) {
    const bool input_sharded = input_spec.memory_config().is_sharded();
    const bool output_sharded = output_spec.memory_config().is_sharded();

    if (!input_sharded && !output_sharded) {
        return std::nullopt;
    }

    if (!is_native_transpose_sharding(input_spec, output_spec.memory_config()) || is_uneven(output_spec)) {
        return std::nullopt;
    }

    if (input_spec.layout() == Layout::ROW_MAJOR) {
        auto is_shard_tile_aligned = [](const TensorSpec& spec) {
            const auto& shard = *get_shard_spec_from_tensor_spec(spec);
            const auto tile_hw = spec.tile().get_tile_hw();
            const uint64_t shard_elements = static_cast<uint64_t>(shard.shape[0]) * shard.shape[1];
            return shard_elements % tile_hw == 0;
        };

        if ((input_sharded && !is_shard_tile_aligned(input_spec)) ||
            (output_sharded && !is_shard_tile_aligned(output_spec))) {
            return std::nullopt;
        }
    }

    TT_FATAL(
        get_shard_spec_from_tensor_spec(output_spec).has_value(),
        "Output must have shard spec when using native sharded path");

    if (input_sharded) {
        const auto& in_shard = *get_shard_spec_from_tensor_spec(input_spec);
        const auto& out_shard = *get_shard_spec_from_tensor_spec(output_spec);
        return TransposeShardSpecs{.input_shard_spec = in_shard, .output_shard_spec = out_shard};
    }

    const auto& out_shard = *get_shard_spec_from_tensor_spec(output_spec);
    const auto& out_shape = output_spec.padded_shape();
    const auto& in_shape = input_spec.padded_shape();
    auto adjusted_in = adjust_shard_spec_to_shape(out_shard, out_shape, in_shape);
    return TransposeShardSpecs{.input_shard_spec = adjusted_in, .output_shard_spec = out_shard};
}

ShardSpec adjust_shard_spec_to_shape(
    const ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape) {
    auto ret = shard_spec;
    uint32_t from_volume_except_width = 1;
    uint32_t to_volume_except_width = 1;
    const auto from_rank = static_cast<int>(from_shape.rank());
    const auto to_rank = static_cast<int>(to_shape.rank());
    for (int i = 0; i < from_rank - 1; ++i) {
        from_volume_except_width *= from_shape[i];
    }
    for (int i = 0; i < to_rank - 1; ++i) {
        to_volume_except_width *= to_shape[i];
    }
    uint32_t from_width = from_shape[-1];
    uint32_t to_width = to_shape[-1];
    TT_FATAL(from_volume_except_width > 0, "Invalid from_shape: volume is zero");
    TT_FATAL(from_width > 0, "Invalid from_shape: width dimension is zero");
    ret.shape[0] = std::max((ret.shape[0] * to_volume_except_width) / from_volume_except_width, 32u);
    ret.shape[1] = std::max((ret.shape[1] * to_width) / from_width, 32u);
    return ret;
}

ShardSpec generate_transpose_shard_spec(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, TensorMemoryLayout memory_layout) {
    auto* device = input_tensor.device();
    auto compute_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
    uint32_t num_cores = all_cores.num_cores();

    uint32_t tensor_height = 1;
    for (int i = 0; i < static_cast<int>(padded_out_shape.rank()) - 1; ++i) {
        tensor_height *= padded_out_shape[i];
    }
    uint32_t tensor_width = padded_out_shape[-1];

    std::array<uint32_t, 2> shard_shape = {0, 0};
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto height_padded = tt::round_up(tensor_height, num_cores * tt::constants::TILE_HEIGHT);
        auto shard_height = tt::round_up(tt::div_up(height_padded, num_cores), tt::constants::TILE_HEIGHT);
        shard_shape = {shard_height, tensor_width};
    } else if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        auto shard_width = tt::round_up(tt::div_up(tensor_width, num_cores), tt::constants::TILE_WIDTH);
        shard_shape = {tensor_height, shard_width};
    } else {
        CoreCoord grid_size = all_cores.bounding_box().grid_size();
        auto height_padded = tt::round_up(tensor_height, grid_size.y * tt::constants::TILE_HEIGHT);
        auto shard_height = tt::round_up(tt::div_up(height_padded, grid_size.y), tt::constants::TILE_HEIGHT);
        auto shard_width = tt::round_up(tt::div_up(tensor_width, grid_size.x), tt::constants::TILE_WIDTH);
        shard_shape = {shard_height, shard_width};
    }
    return ShardSpec(all_cores, shard_shape, ShardOrientation::ROW_MAJOR);
}

CoreRangeSet get_transpose_worker_grid(const Tensor& input_tensor, const MemoryConfig& output_memory_config) {
    if (output_memory_config.is_sharded() && output_memory_config.shard_spec().has_value()) {
        const auto& grid = output_memory_config.shard_spec()->grid;
        auto* device = input_tensor.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(grid)) {
                return sub_device_workers;
            }
        }
    }

    if (input_tensor.is_sharded() && is_native_transpose_sharding(input_tensor.tensor_spec(), output_memory_config)) {
        const auto& grid = input_tensor.shard_spec()->grid;
        auto* device = input_tensor.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers = device->worker_cores(HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(grid)) {
                return sub_device_workers;
            }
        }
    }

    auto* device = input_tensor.device();
    return device->worker_cores(HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
}

std::uint32_t* copy_transpose_common_runtime_args(const Buffer& buffer, std::uint32_t* dst) {
    const auto src =
        TensorAccessorArgs(buffer, tensor_accessor::ArgConfig::RuntimeTensorShape).get_common_runtime_args();
    return std::copy(src.begin(), src.end(), dst);
}

}  // namespace ttnn::operations::data_movement::transpose
