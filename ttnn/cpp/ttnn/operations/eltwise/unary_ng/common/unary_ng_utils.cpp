// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/eltwise/unary_ng/common/unary_ng_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::unary_ng {

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const TensorSpec& tensor_spec) {
    return tensor_spec.memory_config().shard_spec();
}

bool is_uneven(const TensorSpec& t) {
    if (!t.memory_config().is_sharded()) {
        return false;
    }
    const auto& shape = t.padded_shape();
    const auto& shard = get_shard_spec(t)->shape;
    const auto rank = shape.rank();
    TT_FATAL(rank >= 2, "Rank must be at least 2");
    uint64_t volume_except_last = 1;
    for (int i = 0; i < static_cast<int>(rank) - 1; ++i) {
        volume_except_last *= shape[i];
    }
    return (volume_except_last % shard[0]) != 0 || (shape[-1] % shard[1]) != 0;
}

bool is_native_L1_sharding(const TensorSpec& input_spec, const MemoryConfig& output_memory_config) {
    if (!output_memory_config.is_sharded()) {
        return false;
    }
    if (!input_spec.memory_config().is_sharded()) {
        return false;
    }
    if (is_uneven(input_spec)) {
        return false;
    }
    if (input_spec.memory_config().buffer_type() == tt::tt_metal::BufferType::DRAM ||
        output_memory_config.buffer_type() == tt::tt_metal::BufferType::DRAM) {
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

std::optional<UnaryShardSpecs> get_shard_specs(const TensorSpec& input_spec, const TensorSpec& output_spec) {
    const bool input_sharded = input_spec.memory_config().is_sharded();
    const bool output_sharded = output_spec.memory_config().is_sharded();

    if (!input_sharded && !output_sharded) {
        return std::nullopt;
    }

    if (!is_native_L1_sharding(input_spec, output_spec.memory_config()) || is_uneven(output_spec)) {
        return std::nullopt;
    }

    TT_FATAL(get_shard_spec(output_spec).has_value(), "Output must have shard spec when using native sharded path");

    const auto& out_shape = output_spec.padded_shape();
    const auto& in_shape = input_spec.padded_shape();

    if (input_sharded) {
        const auto& in_shard = *get_shard_spec(input_spec);
        const auto& out_shard = *get_shard_spec(output_spec);
        return UnaryShardSpecs{.input_shard_spec = in_shard, .output_shard_spec = out_shard};
    }

    const auto& out_shard = *get_shard_spec(output_spec);
    auto adjusted_in = adjust_to_shape(out_shard, out_shape, in_shape);
    return UnaryShardSpecs{.input_shard_spec = adjusted_in, .output_shard_spec = out_shard};
}

tt::tt_metal::ShardSpec adjust_to_shape(
    const tt::tt_metal::ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape) {
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

CoreRangeSet get_worker_grid(
    const Tensor& input_tensor,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const MemoryConfig& memory_config_actual) {
    if (sub_core_grids.has_value()) {
        log_debug(tt::LogOp, "UnaryNg: Using provided sub_core_grids for worker grid {}", sub_core_grids->str());
        return sub_core_grids.value();
    }

    auto get_tensor_grid = [](const Tensor& tensor) -> CoreRangeSet {
        const auto& grid = tensor.shard_spec()->grid;
        auto* device = tensor.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers =
                device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(grid)) {
                return sub_device_workers;
            }
        }
        __builtin_unreachable();
    };

    if (output_tensor.has_value() && output_tensor->is_sharded()) {
        log_debug(
            tt::LogOp, "UnaryNg: Using output tensor grid for worker grid {}", output_tensor->shard_spec()->grid.str());
        return get_tensor_grid(*output_tensor);
    }

    if (memory_config.has_value() && memory_config->is_sharded() && memory_config->shard_spec().has_value()) {
        const auto& grid = memory_config->shard_spec()->grid;
        log_debug(tt::LogOp, "UnaryNg: Using memory config shard spec grid for worker grid {}", grid.str());
        auto* device = input_tensor.device();
        for (const auto& sub_device_id : device->get_sub_device_ids()) {
            const auto& sub_device_workers =
                device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
            if (sub_device_workers.intersects(grid)) {
                return sub_device_workers;
            }
        }
    }

    if (output_tensor.has_value() || memory_config.has_value()) {
        log_debug(tt::LogOp, "UnaryNg: Using all worker cores (output or memory config not sharded)");
        auto* device = input_tensor.device();
        return device->worker_cores(
            tt::tt_metal::HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
    }

    if (input_tensor.is_sharded() && is_native_L1_sharding(input_tensor.tensor_spec(), memory_config_actual)) {
        log_debug(
            tt::LogOp, "UnaryNg: Native L1 sharding using input tensor grid {}", input_tensor.shard_spec()->grid.str());
        return get_tensor_grid(input_tensor);
    }

    log_debug(tt::LogOp, "UnaryNg: Using all worker cores of the device");
    auto* device = input_tensor.device();
    return device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
}

tt::tt_metal::ShardSpec generate_shard_spec_all_cores(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, tt::tt_metal::TensorMemoryLayout memory_layout) {
    using namespace tt::tt_metal;
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
    log_debug(tt::LogOp, "UnaryNg: Generated shard spec using all {} worker cores", num_cores);
    return ShardSpec(all_cores, shard_shape, ShardOrientation::ROW_MAJOR);
}

}  // namespace ttnn::operations::unary_ng
