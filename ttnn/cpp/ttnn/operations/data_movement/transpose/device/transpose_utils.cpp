// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_utils.hpp"

#include <algorithm>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/experimental/per_core_allocation/memory_config.hpp>
#include <tt-metalium/experimental/range_lockstep_allocation/memory_config.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/mesh_device.hpp>

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

void copy_experimental_allocation_flags(const MemoryConfig& source, MemoryConfig& dest) {
    experimental::per_core_allocation::set_per_core_allocation(
        dest, experimental::per_core_allocation::is_per_core_allocation(source));
    experimental::range_lockstep_allocation::set_range_lockstep_allocation(
        dest, experimental::range_lockstep_allocation::is_range_lockstep_allocation(source));
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
    std::optional<ShardOrientation> orientation_hint,
    OutputTiling output_tiling) {
    auto* device = input_tensor.device();
    const auto compute_grid_size = device->compute_with_storage_grid_size();
    const auto input_orientation = input_tensor.shard_spec().has_value()
                                       ? std::optional{input_tensor.shard_spec()->orientation}
                                       : std::nullopt;
    // Default preserves pre-existing tile-aligned synth; only RowMajor unlocks RM synth.
    const bool is_tile = output_tiling != OutputTiling::RowMajor;
    auto spec = common::synthesize_output_shard_spec(
        compute_grid_size,
        padded_out_shape,
        memory_layout,
        {.is_tile = is_tile,
         .orientation_hint = orientation_hint,
         .input_orientation = input_orientation,
         .caller_tag = "Transpose"});
    // Shrink only when we actually produced an RM synth; Default/Tile keep the pre-existing geometry.
    if (!is_tile) {
        spec = *common::shrink_shard_for_rm_page_alignment(
            spec,
            input_tensor.layout(),
            input_tensor.element_size(),
            static_cast<uint64_t>(padded_out_shape[-1]),
            compute_grid_size,
            memory_layout,
            common::RmPageAlignmentMode::Lenient);
    }
    log_debug(
        tt::LogOp,
        "Transpose: generated shard spec ({}, {}) over {} populated cores",
        spec.shape[0],
        spec.shape[1],
        spec.grid.num_cores());
    return spec;
}

// The sharded row-major WH factory handles an input height that is not a tile multiple through the narrow-row
// untilize, which packs H % TILE_HEIGHT datums per output row. Four shape classes come out wrong (all four measured
// on Blackhole, the first three also on Wormhole; bf16 and fp32), so they are rejected here before any program is
// built:
// - H > TILE_HEIGHT: the kernel packs every tile column at H % TILE_HEIGHT datums, so each full tile column is
//   clipped and the output row is ht * (H % TILE_HEIGHT) datums instead of H;
// - a row wider than one face: 16-byte alignment is necessary but not sufficient (the LLK narrow-row
//   static_assert in llk_pack_untilize.h records the same limit on both architectures);
// - a row byte width that is not a multiple of the L1 alignment: the sharded output stores each H-datum row at
//   the aligned page pitch, but Wormhole packs the rows at their byte width and Blackhole at a 16-byte-floored
//   pitch (its per-row L1 advance is in 16-byte units; the LLK init asserts the same). The factory needs native
//   sharding, which is L1-only, so the L1 alignment is the page pitch;
// - W % TILE_WIDTH != 0: the narrow pack always writes TILE_HEIGHT rows of H % TILE_HEIGHT datums per tile
//   column, but the factory sizes the sharded output page as (W % TILE_WIDTH) * (H % TILE_HEIGHT) datums and
//   advances the last column by one page, i.e. W % TILE_WIDTH rows. Within a shard the next H x W slab
//   overwrites the surplus rows, but the last slab writes them past the end of the output buffer (bf16
//   [1, 4, 16, 48]: 512 bytes past a 6144-byte shard; measured through a canary buffer allocated right above
//   the output). When the remainders differ the other columns' page counts disagree with the packed bytes as
//   well (bf16 16x40: 4851 of 5120 datums wrong; 8x48 overfills the output CB and hangs), and when the tile
//   does not hold a whole number of H % TILE_HEIGHT-row pages the full columns advance short too (fp32 H 12,
//   W 44: 2 pages of 576 bytes advanced per 1536 bytes packed).
void validate_sharded_rm_transpose_narrow_row(const Tensor& input_tensor) {
    const uint32_t logical_h = input_tensor.logical_shape()[-2];
    const uint32_t logical_w = input_tensor.logical_shape()[-1];
    const uint32_t row_num_datums = logical_h % tt::constants::TILE_HEIGHT;
    const uint32_t w_remainder = logical_w % tt::constants::TILE_WIDTH;
    TT_FATAL(
        logical_h < tt::constants::TILE_HEIGHT,
        "Sharded row-major transpose: H {} is above one tile ({}) and not a multiple of it; every tile column of the "
        "output row would be clipped to {} datums. Pad H to a multiple of {} or use an interleaved input.",
        logical_h,
        tt::constants::TILE_HEIGHT,
        row_num_datums,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        row_num_datums <= tt::constants::FACE_WIDTH,
        "Sharded row-major transpose: output rows of {} datums (H {}) are wider than one face ({} datums), which the "
        "narrow-row untilize does not pack correctly. Pad H to a multiple of {} or use an interleaved input.",
        row_num_datums,
        logical_h,
        tt::constants::FACE_WIDTH,
        tt::constants::TILE_HEIGHT);
    const uint32_t row_bytes = row_num_datums * input_tensor.element_size();
    const uint32_t alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t page_pitch = (row_bytes + alignment - 1) / alignment * alignment;
    TT_FATAL(
        row_bytes % alignment == 0,
        "Sharded row-major transpose: output rows of {} datums ({} bytes, H {}) cannot be packed at the {}-byte page "
        "pitch of the sharded output; the row byte width must be a multiple of the {}-byte buffer alignment. Pad H to "
        "a multiple of {} or use an interleaved input.",
        row_num_datums,
        row_bytes,
        logical_h,
        page_pitch,
        alignment,
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        w_remainder == 0,
        "Sharded row-major transpose: W {} is not a multiple of {} while H {} is below one tile; the narrow-row "
        "untilize packs {} rows of {} datums per tile column but the sharded output pages hold {} rows for the last "
        "column, so every pack of that column writes {} bytes the output pages do not account for (past the end of "
        "the output for the last slab of a shard). Pad W to a multiple of {} or use an interleaved input.",
        logical_w,
        tt::constants::TILE_WIDTH,
        logical_h,
        tt::constants::TILE_HEIGHT,
        row_num_datums,
        w_remainder,
        (tt::constants::TILE_HEIGHT - w_remainder) * row_bytes,
        tt::constants::TILE_WIDTH);
}

}  // namespace ttnn::operations::data_movement::transpose
