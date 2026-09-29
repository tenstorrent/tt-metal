// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_utils.hpp"

#include <algorithm>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/data_movement/common/synthesize_output_shard_spec.hpp"

namespace ttnn::operations::data_movement::transpose {

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

// Size CoreRangeSet to populated shards (avoids L1 waste + misleading .grid.num_cores()).
ShardSpec generate_transpose_shard_spec(
    const Tensor& input_tensor,
    const ttnn::Shape& padded_out_shape,
    TensorMemoryLayout memory_layout,
    std::optional<ShardOrientation> orientation_hint) {
    auto* device = input_tensor.device();
    auto spec = common::synthesize_output_shard_spec(
        device->compute_with_storage_grid_size(),
        padded_out_shape,
        memory_layout,
        {.is_tile = true,
         .orientation_hint = orientation_hint,
         .input_orientation = input_tensor.shard_spec().has_value()
                                  ? std::optional{input_tensor.shard_spec()->orientation}
                                  : std::nullopt,
         .caller_tag = "Transpose"});
    log_debug(
        tt::LogOp,
        "Transpose: generated shard spec ({}, {}) over {} populated cores",
        spec.shape[0],
        spec.shape[1],
        spec.grid.num_cores());
    return spec;
}

}  // namespace ttnn::operations::data_movement::transpose
