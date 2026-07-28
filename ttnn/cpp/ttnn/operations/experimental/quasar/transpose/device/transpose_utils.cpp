// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_utils.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::quasar::transpose_op {

using namespace tt::tt_metal;

namespace {

// True if padded shape doesn't divide evenly into shard, or if the sharded config has no spec.
// Conservatively true for rank < 2 so callers fall back to interleaved.
bool is_unevenly_sharded(const tt::tt_metal::TensorSpec& t) {
    if (!t.memory_config().is_sharded()) {
        return false;
    }
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
        volume_except_last *= shape[i];
    }
    return (volume_except_last % shard[0]) != 0 || (shape[-1] % shard[1]) != 0;
}

// RM shard element count not a tile multiple → native kernels can't use it (whole-tile pages
// only); the interleaved TensorAccessor path handles such shards.
bool rm_shard_elements_not_tile_aligned(const MemoryConfig& mc) {
    if (!mc.shard_spec().has_value()) {
        return false;
    }
    constexpr uint64_t tile_hw =
        static_cast<uint64_t>(tt::constants::TILE_HEIGHT) * static_cast<uint64_t>(tt::constants::TILE_WIDTH);
    const auto& s = mc.shard_spec()->shape;
    const uint64_t elems = static_cast<uint64_t>(s[0]) * static_cast<uint64_t>(s[1]);
    return elems % tile_hw != 0;
}

// Per-side native eligibility: sharded, non-DRAM, non-BLOCK, and for RM: shard elements are a
// tile_hw multiple.
bool side_native(const MemoryConfig& mc, Layout layout) {
    if (!mc.is_sharded()) {
        return false;
    }
    if (mc.buffer_type() == BufferType::DRAM) {
        return false;
    }
    if (mc.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        return false;
    }
    if (layout == Layout::ROW_MAJOR && rm_shard_elements_not_tile_aligned(mc)) {
        return false;
    }
    return true;
}

}  // namespace

bool is_native_transpose_sharding(
    const tt::tt_metal::TensorSpec& input_spec, const std::optional<MemoryConfig>& output_memory_config) {
    if (!side_native(input_spec.memory_config(), input_spec.layout())) {
        return false;
    }
    if (is_unevenly_sharded(input_spec)) {
        return false;
    }
    if (!output_memory_config.has_value()) {
        // Pre-derivation: output spec will be synthesized from input — input eligibility suffices.
        return true;
    }
    if (!side_native(*output_memory_config, input_spec.layout())) {
        return false;
    }
    // Sharded WH/HC factories require a single shared grid; only enforce when both specs concrete
    // (a missing output spec implicitly inherits the input grid).
    const auto& in_ss = input_spec.memory_config().shard_spec();
    const auto& out_ss = output_memory_config->shard_spec();
    return !(in_ss.has_value() && out_ss.has_value() && in_ss->grid != out_ss->grid);
}

std::optional<ShardSpec> adjust_shard_spec_to_shape(
    const ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape) {
    // uint64 accumulators avoid overflow on large tensors; nullopt on non-exact division lets
    // callers fall back gracefully. Transpose preserves rank — mismatched ranks would yield
    // inconsistent volume math, so enforce equality.
    TT_FATAL(
        from_shape.rank() == to_shape.rank(),
        "adjust_shard_spec_to_shape: from_shape rank ({}) and to_shape rank ({}) must match.",
        from_shape.rank(),
        to_shape.rank());
    uint64_t from_volume_except_width = 1;
    uint64_t to_volume_except_width = 1;
    const auto rank = static_cast<int>(from_shape.rank());
    for (int i = 0; i < rank - 1; ++i) {
        from_volume_except_width *= static_cast<uint64_t>(from_shape[i]);
        to_volume_except_width *= static_cast<uint64_t>(to_shape[i]);
    }
    const uint64_t from_width = static_cast<uint64_t>(from_shape[-1]);
    const uint64_t to_width = static_cast<uint64_t>(to_shape[-1]);
    if (from_volume_except_width == 0 || from_width == 0) {
        return std::nullopt;
    }

    const uint64_t h_num = static_cast<uint64_t>(shard_spec.shape[0]) * to_volume_except_width;
    const uint64_t w_num = static_cast<uint64_t>(shard_spec.shape[1]) * to_width;
    if (h_num % from_volume_except_width != 0 || w_num % from_width != 0) {
        return std::nullopt;
    }

    // Exact ratio scale, no tile-size clamp: clamping oversizes shards when transpose legitimately
    // shrinks a dim sub-tile, causing silent correctness bugs. Callers that need tile alignment
    // post-check shape[i] % TILE_* and fall back; RM callers tolerate sub-tile shards.
    auto ret = shard_spec;
    ret.shape[0] = static_cast<uint32_t>(h_num / from_volume_except_width);
    ret.shape[1] = static_cast<uint32_t>(w_num / from_width);
    return ret;
}

// Build a sharded spec over the full compute grid (used when no input shard_spec is available
// to scale from, e.g. interleaved input + sharded output request).
ShardSpec generate_transpose_shard_spec(
    const Tensor& input_tensor, const ttnn::Shape& padded_out_shape, TensorMemoryLayout memory_layout) {
    auto* device = input_tensor.device();
    auto compute_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores(CoreRange({0, 0}, {compute_grid_size.x - 1, compute_grid_size.y - 1}));
    uint32_t num_cores = all_cores.num_cores();

    // uint64 intermediates avoid overflow when leading-dim product exceeds 2^32; final shard
    // dims are still uint32 (the hardware representation).
    uint64_t tensor_height = 1;
    for (int i = 0; i < static_cast<int>(padded_out_shape.rank()) - 1; ++i) {
        tensor_height *= static_cast<uint64_t>(padded_out_shape[i]);
    }
    uint64_t tensor_width = padded_out_shape[-1];

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
    } else {
        CoreCoord grid_size = all_cores.bounding_box().grid_size();
        auto height_padded =
            tt::round_up(tensor_height, static_cast<uint64_t>(grid_size.y) * tt::constants::TILE_HEIGHT);
        auto shard_height =
            tt::round_up(tt::div_up(height_padded, static_cast<uint64_t>(grid_size.y)), tt::constants::TILE_HEIGHT);
        auto shard_width =
            tt::round_up(tt::div_up(tensor_width, static_cast<uint64_t>(grid_size.x)), tt::constants::TILE_WIDTH);
        shard_shape = {static_cast<uint32_t>(shard_height), static_cast<uint32_t>(shard_width)};
    }
    log_debug(tt::LogOp, "Transpose: generated shard spec over full compute grid ({} cores)", num_cores);
    return ShardSpec(all_cores, shard_shape, ShardOrientation::ROW_MAJOR);
}

}  // namespace ttnn::operations::experimental::quasar::transpose_op
