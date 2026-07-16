// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/device/repeat_utils.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::data_movement::repeat {

using namespace tt::tt_metal;

namespace {

// True if padded shape doesn't divide evenly into shard (or no spec).
bool is_unevenly_sharded(const TensorSpec& t) {
    const auto& shard_spec = t.memory_config().shard_spec();
    if (!shard_spec.has_value()) {
        return true;
    }
    const auto& shape = t.padded_shape();
    const auto rank = shape.rank();
    if (rank < 2) {
        return true;
    }
    const auto& shard = shard_spec->shape;
    uint64_t volume_except_last = 1;
    for (int i = 0; i < static_cast<int>(rank) - 1; ++i) {
        volume_except_last *= static_cast<uint64_t>(shape[i]);
    }
    return (volume_except_last % shard[0]) != 0 || (shape[-1] % shard[1]) != 0;
}

// L1-sharded only; DRAM-sharded uses composite.
bool side_native(const MemoryConfig& mc, Layout /*layout*/) {
    if (!mc.is_sharded()) {
        return false;
    }
    if (mc.buffer_type() == BufferType::DRAM) {
        return false;
    }
    return true;
}

// Product of dims after repeat axis but before W; equals row-stride between successive k-values.
uint64_t trailing_row_volume(const ttnn::Shape& shape, int32_t repeat_dim) {
    const auto rank = static_cast<int32_t>(shape.rank());
    if (repeat_dim < 0 || repeat_dim >= rank - 1) {
        return 1;
    }
    uint64_t v = 1;
    for (int32_t i = repeat_dim + 1; i < rank - 1; ++i) {
        v *= static_cast<uint64_t>(shape[i]);
    }
    return v;
}

}  // namespace

bool is_replication_locally_contained(
    const ShardSpec& input_shard_spec,
    const ttnn::Shape& input_padded_shape,
    int32_t repeat_dim,
    uint32_t num_repeats) {
    const auto rank = static_cast<int32_t>(input_padded_shape.rank());
    if (rank < 2) {
        return false;
    }
    // Last-dim: replicas stay on same core along W.
    if (repeat_dim == rank - 1) {
        return true;
    }
    if (num_repeats <= 1) {
        return true;
    }
    // Higher-dim: per-core rows must hold whole replica groups (shape[repeat_dim] * trailing rows).
    const uint64_t per_core_rows = input_shard_spec.shape[0];
    if (per_core_rows == 0) {
        return false;
    }
    const uint64_t trv = trailing_row_volume(input_padded_shape, repeat_dim);
    const uint64_t group_size = trv * static_cast<uint64_t>(input_padded_shape[repeat_dim]);
    return group_size != 0 && (per_core_rows % group_size) == 0;
}

bool is_native_repeat_sharding(
    const TensorSpec& input_spec,
    const std::optional<MemoryConfig>& output_memory_config,
    int32_t repeat_dim,
    uint32_t num_repeats) {
    if (!side_native(input_spec.memory_config(), input_spec.layout())) {
        return false;
    }
    if (is_unevenly_sharded(input_spec)) {
        return false;
    }
    // TILE input: reject non-tile-aligned H/W (native tile path requires alignment).
    if (input_spec.layout() == tt::tt_metal::Layout::TILE) {
        const auto& lshape = input_spec.logical_shape();
        if (lshape.rank() < 2 || (lshape[-1] % tt::constants::TILE_WIDTH) != 0 ||
            (lshape[-2] % tt::constants::TILE_HEIGHT) != 0) {
            return false;
        }
    }
    // RM WIDTH/BLOCK last-dim: reject (ttnn::view breaks shard width).
    if (input_spec.layout() == tt::tt_metal::Layout::ROW_MAJOR && repeat_dim >= 0 &&
        repeat_dim == static_cast<int32_t>(input_spec.logical_shape().rank()) - 1) {
        const auto in_layout = input_spec.memory_config().memory_layout();
        if (in_layout == TensorMemoryLayout::WIDTH_SHARDED || in_layout == TensorMemoryLayout::BLOCK_SHARDED) {
            return false;
        }
    }
    if (output_memory_config.has_value()) {
        if (output_memory_config->is_sharded()) {
            if (output_memory_config->buffer_type() == BufferType::DRAM) {
                return false;
            }
            // Cross-layout sharded->sharded needs composite reshard.
            if (input_spec.memory_config().memory_layout() != output_memory_config->memory_layout()) {
                return false;
            }
            // Grids must match when both specs are set.
            const auto& in_ss = input_spec.memory_config().shard_spec();
            const auto& out_ss = output_memory_config->shard_spec();
            if (in_ss.has_value() && out_ss.has_value() && in_ss->grid != out_ss->grid) {
                return false;
            }
        }
    }
    // repeat_dim < 0: skip containment check until repeat axis is known.
    if (repeat_dim < 0) {
        return true;
    }
    const auto& in_ss = input_spec.memory_config().shard_spec();
    if (!in_ss.has_value()) {
        return false;
    }
    return is_replication_locally_contained(*in_ss, input_spec.padded_shape(), repeat_dim, num_repeats);
}

std::optional<ShardSpec> adjust_repeat_shard_spec_to_shape(
    const ShardSpec& shard_spec,
    const ttnn::Shape& from_shape,
    const ttnn::Shape& to_shape,
    int32_t repeat_dim,
    uint32_t num_repeats) {
    TT_FATAL(
        from_shape.rank() == to_shape.rank(),
        "adjust_repeat_shard_spec_to_shape: rank mismatch ({} vs {})",
        from_shape.rank(),
        to_shape.rank());
    if (num_repeats == 0) {
        return std::nullopt;
    }
    const auto rank = static_cast<int32_t>(from_shape.rank());
    if (rank < 2 || repeat_dim < 0 || repeat_dim >= rank) {
        return std::nullopt;
    }

    auto ret = shard_spec;
    if (repeat_dim == rank - 1) {
        // Last-dim: scale shard width by num_repeats.
        if (to_shape[-1] % num_repeats != 0 || from_shape[-1] != to_shape[-1] / num_repeats) {
            return std::nullopt;
        }
        const uint64_t scaled = static_cast<uint64_t>(shard_spec.shape[1]) * static_cast<uint64_t>(num_repeats);
        ret.shape[1] = static_cast<uint32_t>(scaled);
        return ret;
    }

    // Higher-dim: scale per-core row count by num_repeats; width unchanged.
    if (from_shape[-1] != to_shape[-1]) {
        return std::nullopt;
    }
    const uint64_t scaled_h = static_cast<uint64_t>(shard_spec.shape[0]) * static_cast<uint64_t>(num_repeats);
    ret.shape[0] = static_cast<uint32_t>(scaled_h);
    return ret;
}

std::optional<ShardSpec> generate_repeat_shard_spec(
    const Tensor& input_tensor,
    const ttnn::Shape& padded_out_shape,
    TensorMemoryLayout memory_layout,
    std::optional<ShardOrientation> orientation_hint) {
    auto* device = input_tensor.device();
    auto compute_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
    uint32_t num_cores = all_cores.num_cores();
    if (num_cores == 0) {
        return std::nullopt;
    }

    uint64_t tensor_height = 1;
    for (int32_t i = 0; i < static_cast<int32_t>(padded_out_shape.rank()) - 1; ++i) {
        tensor_height *= static_cast<uint64_t>(padded_out_shape[i]);
    }
    uint64_t tensor_width = padded_out_shape[-1];
    if (tensor_height == 0 || tensor_width == 0) {
        return std::nullopt;
    }

    std::array<uint32_t, 2> shard_shape = {0, 0};
    if (memory_layout == TensorMemoryLayout::HEIGHT_SHARDED) {
        auto height_padded = tt::round_up(tensor_height, static_cast<uint64_t>(num_cores) * tt::constants::TILE_HEIGHT);
        auto shard_height =
            tt::round_up(tt::div_up(height_padded, static_cast<uint64_t>(num_cores)), tt::constants::TILE_HEIGHT);
        shard_shape = {static_cast<uint32_t>(shard_height), static_cast<uint32_t>(tensor_width)};
    } else if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        auto shard_width =
            tt::round_up(tt::div_up(tensor_width, static_cast<uint64_t>(num_cores)), tt::constants::TILE_WIDTH);
        shard_shape = {static_cast<uint32_t>(tensor_height), static_cast<uint32_t>(shard_width)};
    } else if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        CoreCoord grid_size = all_cores.bounding_box().grid_size();
        if (grid_size.x == 0 || grid_size.y == 0) {
            return std::nullopt;
        }
        auto height_padded =
            tt::round_up(tensor_height, static_cast<uint64_t>(grid_size.y) * tt::constants::TILE_HEIGHT);
        auto shard_height =
            tt::round_up(tt::div_up(height_padded, static_cast<uint64_t>(grid_size.y)), tt::constants::TILE_HEIGHT);
        auto shard_width =
            tt::round_up(tt::div_up(tensor_width, static_cast<uint64_t>(grid_size.x)), tt::constants::TILE_WIDTH);
        shard_shape = {static_cast<uint32_t>(shard_height), static_cast<uint32_t>(shard_width)};
    } else {
        return std::nullopt;  // INTERLEAVED / unsupported — caller handles.
    }

    // RM: reject if page_size not L1-aligned (16 bytes).
    if (input_tensor.layout() == tt::tt_metal::Layout::ROW_MAJOR) {
        const uint64_t page_size_bytes =
            static_cast<uint64_t>(shard_shape[1]) * static_cast<uint64_t>(input_tensor.element_size());
        constexpr uint64_t kL1Alignment = 16;
        if (page_size_bytes == 0 || (page_size_bytes % kL1Alignment) != 0) {
            return std::nullopt;
        }
        if (tensor_width % shard_shape[1] != 0) {
            return std::nullopt;
        }
    }

    log_debug(
        tt::LogOp, "Repeat: synthesised shard spec ({}, {}) over {} cores", shard_shape[0], shard_shape[1], num_cores);
    // Prefer explicit hint, then input's orientation, else ROW_MAJOR.
    ShardOrientation orientation = ShardOrientation::ROW_MAJOR;
    if (orientation_hint.has_value()) {
        orientation = *orientation_hint;
    } else if (input_tensor.shard_spec().has_value()) {
        orientation = input_tensor.shard_spec()->orientation;
    }

    // For BLOCK sharding, place the shards on exactly the sub-grid they occupy instead of the whole
    // compute grid. Reporting the full grid would leave the tensor's shard_spec inconsistent with its
    // physical buffer (tt-metal trims the grid to the used cores at allocation), which hides the real
    // layout from the op-constraints query and can corrupt consumers that trust the reported grid.
    CoreRangeSet shard_grid = all_cores;
    if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
        const uint32_t num_shards_along_height =
            static_cast<uint32_t>(tt::div_up(tensor_height, static_cast<uint64_t>(shard_shape[0])));
        const uint32_t num_shards_along_width =
            static_cast<uint32_t>(tt::div_up(tensor_width, static_cast<uint64_t>(shard_shape[1])));
        // Row-major maps width-shards to grid columns (x) and height-shards to rows (y); column-major swaps.
        const bool row_major = orientation == ShardOrientation::ROW_MAJOR;
        const uint32_t grid_x = row_major ? num_shards_along_width : num_shards_along_height;
        const uint32_t grid_y = row_major ? num_shards_along_height : num_shards_along_width;
        if (grid_x == 0 || grid_y == 0 || grid_x > compute_grid_size.x || grid_y > compute_grid_size.y) {
            return std::nullopt;
        }
        shard_grid = CoreRangeSet(CoreRange({0, 0}, {grid_x - 1, grid_y - 1}));
    }
    return ShardSpec(shard_grid, shard_shape, orientation);
}

}  // namespace ttnn::operations::data_movement::repeat
