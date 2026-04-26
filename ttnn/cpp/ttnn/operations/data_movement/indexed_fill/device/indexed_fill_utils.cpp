// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_utils.hpp"

#include <algorithm>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/tensor/tensor_utils.hpp"

namespace ttnn::operations::data_movement::indexed_fill {

namespace {

const std::optional<tt::tt_metal::ShardSpec>& get_shard_spec(const TensorSpec& tensor_spec) {
    return tensor_spec.memory_config().shard_spec();
}

}  // namespace

bool is_uneven(const TensorSpec& t) {
    if (!t.memory_config().is_sharded()) {
        return false;
    }
    // Guard against sharded MemoryConfigs that have no explicit shard_spec.  Without this
    // an empty std::optional dereference (->shape) would be UB.  In practice we only
    // reach this with a derived/explicit shard_spec, but the predicate is also called
    // from path-selection code that should not crash on partially-specified configs.
    const auto& shard_spec_opt = get_shard_spec(t);
    TT_FATAL(
        shard_spec_opt.has_value(),
        "indexed_fill: is_uneven() called on a sharded MemoryConfig with no explicit shard_spec; "
        "the caller must derive a shard_spec first (see compute_output_specs).");
    const auto& shard = shard_spec_opt->shape;
    const auto& shape = t.padded_shape();
    const auto rank = shape.rank();
    TT_FATAL(rank >= 2, "Rank must be at least 2");
    uint64_t volume_except_last = 1;
    for (int i = 0; i < static_cast<int>(rank) - 1; ++i) {
        volume_except_last *= shape[i];
    }
    return (volume_except_last % shard[0]) != 0 || (shape[-1] % shard[1]) != 0;
}

bool is_native_indexed_fill_sharding(
    const TensorSpec& input_a_spec,
    const TensorSpec& /*input_b_spec*/,
    const TensorSpec& batch_id_spec,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::TensorMemoryLayout;

    // Both input_a and output must be sharded L1.
    if (!input_a_spec.memory_config().is_sharded() || !output_memory_config.is_sharded()) {
        return false;
    }
    if (input_a_spec.memory_config().buffer_type() != BufferType::L1 ||
        output_memory_config.buffer_type() != BufferType::L1) {
        return false;
    }

    // batch_id must also live in L1 (the native reader still NoC-reads it).
    if (batch_id_spec.memory_config().buffer_type() != BufferType::L1) {
        return false;
    }

    // Native fast path is HEIGHT_SHARDED only (one full per-batch slab per core).
    if (input_a_spec.memory_config().memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        return false;
    }
    if (output_memory_config.memory_layout() != TensorMemoryLayout::HEIGHT_SHARDED) {
        return false;
    }

    // Even sharding required (no leftover rows / cols).
    if (is_uneven(input_a_spec)) {
        return false;
    }

    // Both shards must have an explicit ShardSpec we can compare.
    if (!input_a_spec.memory_config().shard_spec().has_value() || !output_memory_config.shard_spec().has_value()) {
        return false;
    }

    const auto& in_shard = *input_a_spec.memory_config().shard_spec();
    const auto& out_shard = *output_memory_config.shard_spec();
    if (in_shard.grid != out_shard.grid) {
        return false;
    }
    if (in_shard.shape != out_shard.shape) {
        return false;
    }

    // One batch per core: shard grid must cover exactly B = padded_shape()[0] cores,
    // and each shard must hold exactly H*W rows (= one whole batch slab).
    const auto& padded = input_a_spec.padded_shape();
    if (padded.rank() < 4) {
        return false;
    }
    const uint32_t B = padded[0];
    const uint32_t batch_height_rows = padded[1] * padded[2];
    if (in_shard.grid.num_cores() != B) {
        return false;
    }
    if (in_shard.shape[0] != batch_height_rows) {
        return false;
    }
    if (in_shard.shape[1] != padded[-1]) {
        return false;
    }

    return true;
}

bool is_shard_local_indexed_fill(
    const TensorSpec& input_a_spec,
    const TensorSpec& input_b_spec,
    const tt::tt_metal::MemoryConfig& output_memory_config) {
    using tt::tt_metal::BufferType;
    using tt::tt_metal::TensorMemoryLayout;

    const auto& a_mem = input_a_spec.memory_config();
    if (!a_mem.is_sharded() || a_mem.buffer_type() != BufferType::L1) {
        return false;
    }
    // Only WIDTH_SHARDED and BLOCK_SHARDED (HEIGHT_SHARDED is handled by the native path or the
    // generic HEIGHT path with the worker-grid fix).
    const auto layout = a_mem.memory_layout();
    if (layout != TensorMemoryLayout::WIDTH_SHARDED && layout != TensorMemoryLayout::BLOCK_SHARDED) {
        return false;
    }

    // Output must be the same layout, L1, same grid, same shard shape.
    if (!output_memory_config.is_sharded() || output_memory_config.buffer_type() != BufferType::L1) {
        return false;
    }
    if (output_memory_config.memory_layout() != layout) {
        return false;
    }
    if (!a_mem.shard_spec().has_value() || !output_memory_config.shard_spec().has_value()) {
        return false;
    }
    const auto& a_shard = *a_mem.shard_spec();
    const auto& out_shard = *output_memory_config.shard_spec();
    if (a_shard.grid != out_shard.grid || a_shard.shape != out_shard.shape) {
        return false;
    }

    // input_b: must be interleaved OR the same WIDTH_SHARDED layout (same grid, same shard
    // width). Direct L1 arithmetic works for WIDTH_SHARDED because every core has all `b`
    // input_b batches locally, so `replace_src` (a global index in [0, b)) always resolves
    // to the correct L1 offset on the current core.
    //
    // BLOCK_SHARDED input_b is intentionally rejected: in that layout each core holds only
    // b/n_y of the `b` input_b batches, so a global `replace_src` index may refer to a batch
    // on a DIFFERENT shard-row core, making direct L1 arithmetic incorrect. Support for that
    // combination (which requires a remote NOC read or TensorAccessor re-indexing) is deferred.
    const auto& b_mem = input_b_spec.memory_config();
    if (b_mem.is_sharded()) {
        // Only WIDTH_SHARDED same-sharded input_b is supported.
        if (layout != TensorMemoryLayout::WIDTH_SHARDED) {
            return false;
        }
        if (b_mem.buffer_type() != BufferType::L1) {
            return false;
        }
        if (b_mem.memory_layout() != TensorMemoryLayout::WIDTH_SHARDED) {
            return false;
        }
        if (!b_mem.shard_spec().has_value()) {
            return false;
        }
        if (b_mem.shard_spec()->grid != a_shard.grid) {
            return false;
        }
        if (b_mem.shard_spec()->shape[1] != a_shard.shape[1]) {
            return false;
        }
    }
    // INTERLEAVED input_b (L1 or DRAM) is always acceptable.

    return true;
}

CoreRangeSet get_indexed_fill_worker_grid(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const Tensor& batch_id,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config) {
    auto* device = input_tensor_a.device();

    // 1. Explicit output shard grid takes precedence — use exactly those cores.
    if (memory_config.has_value() && memory_config->is_sharded() && memory_config->shard_spec().has_value()) {
        return memory_config->shard_spec()->grid;
    }

    // 2. Any sharded input_a drives the worker grid. We use the shard grid directly (not the
    //    full sub-device worker set)
    if (input_tensor_a.is_sharded()) {
        return input_tensor_a.shard_spec()->grid;
    }
    if (input_tensor_b.is_sharded()) {
        return input_tensor_b.shard_spec()->grid;
    }
    if (batch_id.is_sharded()) {
        return batch_id.shard_spec()->grid;
    }

    // 3. Default: all worker cores of the first sub-device.
    return device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, device->get_sub_device_ids().front());
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

    // The div_up + round_up approach distributes pages as uniformly as possible.
    // When tensor_height (or tensor_width for WIDTH_SHARDED) is less than num_cores,
    // some cores receive shards that map entirely to padding and are handled by
    // early-return guards in the reader/writer kernels.  Effective parallelism is
    // therefore min(total_pages, num_cores), not num_cores.
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

}  // namespace ttnn::operations::data_movement::indexed_fill
