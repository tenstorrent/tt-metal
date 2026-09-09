// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_codegen_supported.hpp"

#include <algorithm>
#include <numeric>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/tensor/tensor_types.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/codegen/reshape_codegen_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::reshape_codegen {

namespace {

constexpr uint32_t kL1Align = 16;
constexpr uint32_t kNonalignedScratchMargin = 64;

uint32_t round_up_u32(uint32_t v, uint32_t align) { return ((v + align - 1) / align) * align; }

// Whether the RM main-path CBs (and, for the non-coalescing branch, the scratch
// CB) fit in one core's L1. Mirrors build_reshape_rm_factory's l1_fits gate
// exactly: `mx % mn != 0` (indivisible sticks) is rejected first, then the
// same core split and CB-sizing arithmetic the factory itself performs (the
// worst case is the busiest core, i.e. max(work_per_core_1, work_per_core_2)),
// and finally the projected CB byte totals are checked against this core's L1
// budget. Every device resource the RM factory allocates is bounded here, at
// this predicate's only leaf (RM has no further dispatch split).
bool rm_supported(const Tensor& input, const ttnn::Shape& out_padded_shape) {
    const auto& in_shape = input.padded_shape();
    if (in_shape.rank() < 1 || out_padded_shape.rank() < 1) {
        return false;
    }
    if (input.storage_type() != ttnn::StorageType::DEVICE) {
        // Not yet on device; nothing concrete to bound against, defer to native/host paths.
        return true;
    }

    uint32_t num_old_sticks = 1;
    for (uint32_t i = 0; i + 1 < in_shape.rank(); ++i) {
        num_old_sticks *= in_shape[i];
    }
    uint32_t num_new_sticks = 1;
    for (uint32_t i = 0; i + 1 < out_padded_shape.rank(); ++i) {
        num_new_sticks *= out_padded_shape[i];
    }

    const uint32_t old_stick_size = in_shape[-1] * input.element_size();
    const uint32_t new_stick_size = out_padded_shape[-1] * input.element_size();
    const uint32_t mx = std::max(old_stick_size, new_stick_size);
    const uint32_t mn = std::min(old_stick_size, new_stick_size);
    if (mn == 0 || mx % mn != 0) {
        return false;
    }
    const uint32_t ratio = mx / mn;

    Buffer* src_buffer = input.buffer();
    if (src_buffer == nullptr) {
        return false;
    }
    const uint32_t old_aligned = static_cast<uint32_t>(src_buffer->aligned_page_size());
    // The output buffer doesn't exist yet at routing time; interleaved page
    // pitch depends only on stick size and placement alignment, both known
    // here, so recompute it the same way TensorAccessorArgs would.
    const auto& allocator = input.device()->allocator();
    const uint32_t out_alignment = allocator->get_alignment(tt::tt_metal::BufferType::DRAM);
    const uint32_t new_aligned = round_up_u32(new_stick_size, out_alignment);

    const bool can_coalesce = (old_stick_size == old_aligned) && (new_stick_size == new_aligned);
    const bool split_by_old = old_stick_size > new_stick_size;
    const uint32_t split_sticks = split_by_old ? num_old_sticks : num_new_sticks;
    if (split_sticks == 0) {
        return false;
    }

    // Same core split the factory performs, to bound against the busiest core.
    IDevice* device = input.device();
    auto grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core_1, work_per_core_2] =
        tt::tt_metal::split_work_to_cores(grid_size, split_sticks, /*row_wise=*/false);
    const uint32_t max_sticks_per_core = std::max(work_per_core_1, work_per_core_2);
    const bool split_nonaligned = split_by_old && !can_coalesce;

    uint32_t cb_page_size = 0;
    uint32_t cb_total = 0;
    if (split_nonaligned) {
        cb_page_size = round_up_u32(new_stick_size, kL1Align);
        cb_total = max_sticks_per_core * ratio * cb_page_size * 2;
    } else if (!can_coalesce) {
        cb_page_size = round_up_u32(new_stick_size, kL1Align);
        cb_total = max_sticks_per_core * cb_page_size * 2;
    } else if (split_by_old) {
        cb_page_size = new_stick_size;
        cb_total = max_sticks_per_core * old_stick_size;
    } else {
        cb_page_size = new_stick_size;
        cb_total = max_sticks_per_core * new_stick_size;
    }
    const uint32_t scratch_size =
        can_coalesce ? 0 : ttnn::prim::kReshapeNabatch * (old_aligned + kNonalignedScratchMargin);

    const uint64_t max_l1 = get_max_l1_space(input);
    return (static_cast<uint64_t>(cb_total) <= max_l1) && (static_cast<uint64_t>(scratch_size) <= max_l1);
}

// Whether the TILE compute-reshape chunking (build_reshape_tile_factory) is
// legal and its per-chunk CB budget fits L1. The chunk size (and hence CB
// footprint) is fixed by W_in/W_out alone -- it does not shrink with the core
// split the way the RM path's per-core CB does -- so this is the single leaf
// to bound for the TILE branch.
bool tile_supported(const Tensor& input, const ttnn::Shape& out_padded_shape) {
    const auto& in_shape = input.padded_shape();
    if (in_shape.rank() < 1 || out_padded_shape.rank() < 1) {
        return false;
    }
    const uint32_t W_in = in_shape[-1];
    const uint32_t W_out = out_padded_shape[-1];
    if (W_in % tt::constants::TILE_WIDTH != 0 || W_out % tt::constants::TILE_WIDTH != 0) {
        return false;
    }
    if (input.storage_type() != ttnn::StorageType::DEVICE) {
        return true;
    }

    const uint32_t Wt_in = W_in / tt::constants::TILE_WIDTH;
    const uint32_t Wt_out = W_out / tt::constants::TILE_WIDTH;
    const uint32_t lcm_w = std::lcm(W_in, W_out);
    const uint32_t in_tile_rows_per_chunk = lcm_w / W_in;
    const uint32_t out_tile_rows_per_chunk = lcm_w / W_out;
    const uint32_t in_tiles_per_chunk = in_tile_rows_per_chunk * Wt_in;
    const uint32_t out_tiles_per_chunk = out_tile_rows_per_chunk * Wt_out;

    const uint64_t padded_volume = input.physical_volume();
    if (padded_volume % (static_cast<uint64_t>(tt::constants::TILE_HEIGHT) * W_in) != 0) {
        return false;
    }
    const uint32_t Ht_in = static_cast<uint32_t>(padded_volume / (tt::constants::TILE_HEIGHT * W_in));
    if (Ht_in % in_tile_rows_per_chunk != 0) {
        return false;
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_bytes = tt::tile_size(cb_data_format);
    const uint32_t chunk_tiles =
        in_tiles_per_chunk + in_tiles_per_chunk + std::max(out_tiles_per_chunk, ttnn::prim::kReshapeTileWriteBatch * 2);
    const uint64_t chunk_l1_bytes = static_cast<uint64_t>(chunk_tiles) * tile_bytes;
    return chunk_l1_bytes <= get_max_l1_space(input);
}

}  // namespace

bool supported_by_codegen(
    const Tensor& input,
    const ttnn::Shape& /*output_logical_shape*/,
    const ttnn::Shape& output_padded_shape,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    // Both codegen program factories emit an interleaved reader/writer pair
    // over the tensor's own buffer; neither handles a sharded input or output.
    if (input.memory_config().is_sharded() || output_mem_config.is_sharded()) {
        return false;
    }
    // Codegen's writer places the output in the same buffer type it reads
    // the input from (its TensorAccessorArgs sizing assumes matching
    // placement alignment on both sides for the coalescing fast path);
    // cross-placement reshape stays on native, which derives each side's
    // pitch independently.
    if (output_mem_config.buffer_type() != input.memory_config().buffer_type()) {
        return false;
    }
    // Block-float packs a shared exponent across a 16-elem sub-block; the
    // codegen kernels move raw bytes with no notion of that structure, so a
    // reshape that changes which elements land in a sub-block (any W change)
    // would corrupt data. Non-W-changing bf8_b/bf4_b reshapes are just as
    // unsupported here since ROW_MAJOR block-float doesn't exist above the
    // host boundary and TILE block-float still repacks tiles.
    if (is_block_float(input.dtype())) {
        return false;
    }

    if (input.layout() == ttnn::ROW_MAJOR_LAYOUT) {
        return rm_supported(input, output_padded_shape);
    }
    if (input.layout() == ttnn::TILE_LAYOUT) {
        return tile_supported(input, output_padded_shape);
    }
    return false;
}

bool is_demoted(
    const Tensor& /*input*/,
    const ttnn::Shape& /*output_logical_shape*/,
    const ttnn::Shape& /*output_padded_shape*/,
    const tt::tt_metal::MemoryConfig& /*output_mem_config*/) {
    // Start conservative: nothing is demoted until verify's performance band
    // identifies a general class of shapes the generated path measurably
    // loses on.
    return false;
}

}  // namespace ttnn::operations::data_movement::reshape_codegen
