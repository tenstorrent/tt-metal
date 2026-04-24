// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_utils.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::data_movement::transpose {

using namespace tt::tt_metal;

namespace {

// Returns true if the tensor's padded shape does not divide evenly into its shard shape,
// or if the sharded memory config has no concrete shard_spec. Conservatively returns true
// for pathological inputs (rank < 2) so callers fall back to interleaved paths.
bool is_unevenly_sharded(const TensorSpec& t) {
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

}  // namespace

bool is_native_transpose_sharding(const TensorSpec& input_spec, const MemoryConfig& output_memory_config) {
    const auto& in_cfg = input_spec.memory_config();
    if (!output_memory_config.is_sharded() || !in_cfg.is_sharded()) {
        return false;
    }
    if (is_unevenly_sharded(input_spec)) {
        return false;
    }
    if (in_cfg.buffer_type() == BufferType::DRAM || output_memory_config.buffer_type() == BufferType::DRAM) {
        return false;
    }
    if (in_cfg.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
        output_memory_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
        return false;
    }
    // ROW_MAJOR sharded shards whose total element count is not a multiple of the tile footprint
    // (TILE_HEIGHT * TILE_WIDTH) cannot use the specialized native kernels — those assume whole-tile
    // pages. Mirrors the `is_shard_tile_aligned` guard used by `unary_ng`. Returning false routes
    // such cases to the interleaved factories, which use TensorAccessorArgs to read/write sharded
    // buffers directly over NOC without an explicit reshard hop.
    if (input_spec.layout() == Layout::ROW_MAJOR) {
        constexpr uint64_t tile_hw =
            static_cast<uint64_t>(tt::constants::TILE_HEIGHT) * static_cast<uint64_t>(tt::constants::TILE_WIDTH);
        auto shard_elements_not_tile_aligned = [](const MemoryConfig& mc) {
            if (!mc.shard_spec().has_value()) {
                return false;
            }
            const auto& s = mc.shard_spec()->shape;
            const uint64_t elems = static_cast<uint64_t>(s[0]) * static_cast<uint64_t>(s[1]);
            return elems % tile_hw != 0;
        };
        if (shard_elements_not_tile_aligned(in_cfg) || shard_elements_not_tile_aligned(output_memory_config)) {
            return false;
        }
    }
    // When either spec is missing we're in the pre-derivation path (callers like transpose.cpp and
    // compute_output_specs use this predicate as an eligibility probe before deriving the output
    // shard_spec). Skip the grid equality check in that case — the derived grid will match the input's.
    if (output_memory_config.shard_spec().has_value() && in_cfg.shard_spec().has_value()) {
        if (in_cfg.shard_spec()->grid != output_memory_config.shard_spec()->grid) {
            return false;
        }
    }
    return true;
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

    // Require exact division so we never silently truncate the scaled shard dimensions. Callers
    // must only invoke this helper when the to/from shape ratios evenly divide the source shard.
    const uint64_t h_num = static_cast<uint64_t>(ret.shape[0]) * to_volume_except_width;
    const uint64_t w_num = static_cast<uint64_t>(ret.shape[1]) * to_width;
    TT_FATAL(
        h_num % from_volume_except_width == 0,
        "adjust_shard_spec_to_shape: height scaling not exact ({} * {} not divisible by {}).",
        ret.shape[0],
        to_volume_except_width,
        from_volume_except_width);
    TT_FATAL(
        w_num % from_width == 0,
        "adjust_shard_spec_to_shape: width scaling not exact ({} * {} not divisible by {}).",
        ret.shape[1],
        to_width,
        from_width);
    uint32_t scaled_h = static_cast<uint32_t>(h_num / from_volume_except_width);
    uint32_t scaled_w = static_cast<uint32_t>(w_num / from_width);

    // Only clamp to tile dimensions when the source shard was already tile-aligned. For sub-tile
    // ROW_MAJOR shards the caller is responsible for legality and we must not over-size the shard.
    const bool source_tile_aligned =
        shard_spec.shape[0] % tt::constants::TILE_HEIGHT == 0 && shard_spec.shape[1] % tt::constants::TILE_WIDTH == 0;
    if (source_tile_aligned) {
        scaled_h = std::max(scaled_h, tt::constants::TILE_HEIGHT);
        scaled_w = std::max(scaled_w, tt::constants::TILE_WIDTH);
    }

    ret.shape[0] = scaled_h;
    ret.shape[1] = scaled_w;
    return ret;
}

// When the user requests a sharded output but no shard_spec, and the input is interleaved, we derive
// a grid and shard shape using the full device compute grid, matching the shard math used elsewhere
// for height/width/block cases.
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
    log_debug(tt::LogOp, "Transpose: generated shard spec over full compute grid ({} cores)", num_cores);
    return ShardSpec(all_cores, shard_shape, ShardOrientation::ROW_MAJOR);
}

// Refreshes the runtime-tensor-shape common args on program cache hits. The destination span is
// bounds-checked against the element count produced by the fresh TensorAccessorArgs so any layout
// change between program creation and the cache hit triggers a clear assertion instead of a silent
// buffer overrun.
void copy_transpose_common_runtime_args(const Buffer& buffer, std::span<std::uint32_t> dst) {
    const auto src =
        TensorAccessorArgs(buffer, tensor_accessor::ArgConfig::RuntimeTensorShape).get_common_runtime_args();
    TT_FATAL(
        dst.size() >= src.size(),
        "copy_transpose_common_runtime_args: destination span ({} elems) too small for common args ({} elems).",
        dst.size(),
        src.size());
    std::copy(src.begin(), src.end(), dst.begin());
}

void refresh_transpose_common_runtime_args(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle writer_kernel_id,
    const Buffer& input_buffer,
    const Buffer& output_buffer) {
    auto& reader_args = GetCommonRuntimeArgs(program, reader_kernel_id);
    auto& writer_args = GetCommonRuntimeArgs(program, writer_kernel_id);
    copy_transpose_common_runtime_args(input_buffer, std::span<std::uint32_t>(reader_args.data(), reader_args.size()));
    copy_transpose_common_runtime_args(output_buffer, std::span<std::uint32_t>(writer_args.data(), writer_args.size()));
}

}  // namespace ttnn::operations::data_movement::transpose
