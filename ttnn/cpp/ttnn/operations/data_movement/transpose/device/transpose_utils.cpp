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

// True when a RM MemoryConfig's shard element count is not a multiple of the tile footprint.
// Such shards cannot use the specialized native kernels (which assume whole-tile pages); the
// interleaved factories handle them via TensorAccessorArgs instead.
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

// Side-level native eligibility: sharded, non-DRAM, non-BLOCK, and for ROW_MAJOR: shard-element
// count is a whole multiple of tile_hw. Shared by the input and output checks below.
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
    const TensorSpec& input_spec, const std::optional<MemoryConfig>& output_memory_config) {
    if (!side_native(input_spec.memory_config(), input_spec.layout())) {
        return false;
    }
    if (is_unevenly_sharded(input_spec)) {
        return false;
    }
    if (!output_memory_config.has_value()) {
        // Pre-derivation path: the output shard_spec is about to be synthesized from the input's,
        // so there's nothing to compare on the output side yet. Input-only eligibility is enough.
        return true;
    }
    if (!side_native(*output_memory_config, input_spec.layout())) {
        return false;
    }
    // The sharded WH/HC program factories assume a single shared grid; only enforce when both
    // shard_specs are concrete. During derive_effective_output_memory_config the output_mem_config
    // may still carry no shard_spec, and its grid is implicitly the input's.
    const auto& in_ss = input_spec.memory_config().shard_spec();
    const auto& out_ss = output_memory_config->shard_spec();
    return !(in_ss.has_value() && out_ss.has_value() && in_ss->grid != out_ss->grid);
}

std::optional<ShardSpec> adjust_shard_spec_to_shape(
    const ShardSpec& shard_spec, const ttnn::Shape& from_shape, const ttnn::Shape& to_shape) {
    // Volumes are accumulated in uint64_t to prevent overflow on large tensors (e.g. N*C*H
    // products can exceed 2^32). Returning nullopt on non-exact division lets callers fall back
    // gracefully (generate_transpose_shard_spec or interleaved) instead of crashing a valid user
    // call, and avoids the silent-truncation pitfall of blind uint division.
    // Transpose preserves rank, so callers must pass equal-rank shapes. Enforce this explicitly:
    // mismatched ranks would silently produce inconsistent volume math (the two accumulation loops
    // below run for different counts) and yield a valid-looking but wrong shard.
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

    // Return the exact ratio-scaled shard without tile-size clamping. A naive
    // `std::max(scaled, TILE_*)` clamp oversizes the shard whenever the target dim legitimately
    // shrinks below a tile (e.g. WH on a tile-aligned height-sharded input where the new width
    // becomes sub-tile) and then fills the grid with capacity for data that doesn't exist,
    // producing silent correctness regressions. Callers that need tile-aligned shards post-check
    // `shape[i] % TILE_*` and fall back to interleaved; callers targeting RM layouts tolerate
    // sub-tile shards. This mirrors `unary_ng`/`binary_ng` semantics but avoids their latent
    // clamp bug — harmless there because their shape ratios never shrink a dim.
    auto ret = shard_spec;
    ret.shape[0] = static_cast<uint32_t>(h_num / from_volume_except_width);
    ret.shape[1] = static_cast<uint32_t>(w_num / from_width);
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

    // Accumulate in uint64 to match adjust_shard_spec_to_shape and avoid overflow on tensors whose
    // product of leading dims exceeds 2^32 (e.g. large batch x seq_len x head_dim attention shapes).
    // The final per-shard dims we hand back are still uint32 (the hardware/shard-spec representation),
    // but the intermediate height computation uses the wider type.
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

// Refreshes the runtime-tensor-shape common args on program cache hits. Strict equality between
// destination and source lengths catches any drift in the TensorAccessorArgs footprint between
// program creation and the cache hit (a >= check would leave stale trailing elements in dst).
void copy_transpose_common_runtime_args(const Buffer& buffer, std::span<std::uint32_t> dst) {
    const auto src =
        TensorAccessorArgs(buffer, tensor_accessor::ArgConfig::RuntimeTensorShape).get_common_runtime_args();
    TT_FATAL(
        dst.size() == src.size(),
        "copy_transpose_common_runtime_args: destination span ({} elems) must match common args ({} elems).",
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
