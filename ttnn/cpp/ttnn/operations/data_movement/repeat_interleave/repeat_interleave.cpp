// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_interleave.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/data_movement/concat/concat.hpp"
#include "ttnn/operations/data_movement/reshape_on_device/reshape.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_device_operation.hpp"
#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_supported.hpp"
#include "ttnn/operations/data_movement/unsqueeze/unsqueeze.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include "repeat_interleave_force.hpp"

namespace ttnn::operations::data_movement::detail {
namespace {

// repeat interleave supports repeats as 1 to inf, and any dim in [-rank, rank) for a tensor of
// arbitrary rank >= 2. `dim` is resolved via get_normalized_index (so negative dims are supported),
// and the last dim is handled by transposing it to the second-to-last position and recursing.
//
// Named (not anonymous-call-site) so the recursive branches below can recurse into it directly:
// recursing through the public ttnn::repeat_interleave() would re-enter the routing gate and let a
// call already routed here escalate to codegen mid-recursion.
//
// NOTE: a sharded input with no `output_mem_config` returns interleaved DRAM, because the repeated
// dim's size changes and there is no cheap way to derive a valid sharded spec for the new size.
Tensor repeat_interleave_native(
    const Tensor& input_a, uint32_t repeat, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    if (input_a.logical_shape().rank() == 1) {
        // A rank-1 tensor's only dim is inherently the "last dim", which the general path below can't
        // handle directly (it would attempt an invalid transpose(-1, -2) on a rank-1 tensor). Add a
        // trailing dummy dim so the target dim (now 0 of 2) is no longer the last dim, recurse through
        // the normal path, then collapse the dummy dim back out.
        const uint32_t size = input_a.logical_shape()[0];
        ttnn::Tensor unsqueezed = ttnn::unsqueeze(input_a, 1);
        ttnn::Tensor result = repeat_interleave_native(unsqueezed, repeat, 0, output_mem_config);
        return ttnn::reshape(result, ttnn::Shape({size * repeat}));
    }

    // Sharded memory configs.
    if (input_a.memory_config().is_sharded() && repeat > 1) {
        // TODO(nuked-op upsample): the native sharded fast path (ROW_MAJOR 4D via ttnn::upsample "nearest")
        // was removed with upsample; every sharded input now takes this interleaved round-trip.
        // Run the op in interleaved and restore the requested sharded config on the output (mirroring
        // ttnn::sort). The general implementation below concats on `normalized_dim + 1`, an *interior*
        // dim of the unsqueezed tensor, which ttnn::concat does not support sharded (it only handles rank-1 width /
        // rank-2 height concats), so concat would auto-unshard to interleaved internally anyway. The
        // output shape differs from the input (the repeated dim grows by `repeat`), so the input's shard
        // spec cannot be reused; a sharded input with no explicit output_mem_config defaults to an
        // interleaved DRAM output. Always convert to `requested` at the end (a no-op if it already
        // matches DRAM) so an explicitly-requested interleaved L1 output isn't silently left in DRAM.
        const MemoryConfig interleaved = ttnn::DRAM_MEMORY_CONFIG;
        ttnn::Tensor interleaved_input = ttnn::to_memory_config(input_a, interleaved);
        ttnn::Tensor interleaved_result = repeat_interleave_native(interleaved_input, repeat, dim, interleaved);
        const MemoryConfig requested = output_mem_config.value_or(interleaved);
        return ttnn::to_memory_config(interleaved_result, requested);
    }

    std::vector<Tensor> combined_tensors;
    combined_tensors.reserve(repeat);
    MemoryConfig mem_config = output_mem_config.value_or(input_a.memory_config());
    if (repeat == 1) {
        return ttnn::to_memory_config(input_a, mem_config);
    }
    const auto& input_a_shape = input_a.logical_shape();
    uint32_t input_rank = input_a_shape.rank();
    uint32_t normalized_dim = input_a_shape.get_normalized_index(dim);
    const bool typecast = is_block_float(input_a.dtype()) || input_a.dtype() == DataType::UINT8;
    if (normalized_dim == input_rank - 1) {
        ttnn::Tensor transpose_input = input_a;
        if (typecast) {
            transpose_input = ttnn::typecast(transpose_input, DataType::BFLOAT16, mem_config);
        }
        auto transposed_input = ttnn::transpose(transpose_input, -1, -2, mem_config);
        // Recurse into the native implementation, not the public entry point: the latter re-enters
        // the routing gate, which would let this inner call dispatch to codegen.
        auto repeated_input = repeat_interleave_native(transposed_input, repeat, -2, mem_config);
        auto result = ttnn::transpose(repeated_input, -1, -2, mem_config);
        return typecast ? ttnn::typecast(result, input_a.dtype(), mem_config) : result;
    }

    // This op is composed of unsqueeze -> concat -> reshape, which are run in ROW_MAJOR layout below so that
    // the concat dimension is not subject to tile (32x32) padding. BFLOAT8_B/BFLOAT4_B/UINT8 cannot be laid out
    // in ROW_MAJOR (block-float formats are TILE-only; UINT8 reshape is unsupported), so they are typecast up to
    // BFLOAT16 for the duration of the op and cast back to the original dtype at the end. The bf16 round-trip is
    // numerically lossless for these formats. A pure TILE-preserving path works for BFLOAT16/BFLOAT8_B but fails
    // for BFLOAT4_B (tensor_spec TILE constraint) and UINT8 (reshape), so the typecast path is kept for them.
    ttnn::Tensor rm_input = input_a;
    if (typecast) {
        rm_input = ttnn::typecast(rm_input, DataType::BFLOAT16, mem_config);
    }

    rm_input = ttnn::to_layout(rm_input, Layout::ROW_MAJOR);
    const auto& rm_input_shape = rm_input.logical_shape();
    ttsl::SmallVector<uint32_t> final_shape;
    final_shape.reserve(input_rank);
    for (uint32_t i = 0; i < rm_input_shape.rank(); i++) {
        final_shape.push_back(rm_input_shape[i]);
    }

    final_shape[normalized_dim] *= repeat;

    auto unsqueezed_tensor = ttnn::unsqueeze(rm_input, normalized_dim + 1);
    std::vector<Tensor> combined_tensors_batch;
    constexpr uint32_t repeats_batched = 32;
    combined_tensors_batch.reserve(std::min(repeat, repeats_batched));
    for (uint32_t i = 0; i < repeat; i++) {
        combined_tensors_batch.push_back(unsqueezed_tensor);

        // Concatenate every 32 tensors or at the end of the loop
        if (combined_tensors_batch.size() == repeats_batched || i == repeat - 1) {
            auto batch_concat = ttnn::concat(combined_tensors_batch, normalized_dim + 1);
            combined_tensors.push_back(batch_concat);
            combined_tensors_batch.clear();
        }
    }

    auto concatenated_tensor = ttnn::concat(combined_tensors, normalized_dim + 1);
    auto reshaped_tensor = ttnn::reshape(concatenated_tensor, ttnn::Shape(final_shape));
    auto original_layout = ttnn::to_layout(reshaped_tensor, input_a.layout());
    return typecast ? ttnn::typecast(original_layout, input_a.dtype(), mem_config)
                    : ttnn::to_memory_config(original_layout, mem_config);
}

// Rank-general TILE page geometry for the (outer, non-sub-tile) repeated axis, in the same page
// space prim::repeat_codegen's kernels index.
struct PageMap {
    uint32_t lower_pages;
    uint32_t rep_dim_pages;
    uint32_t total_out_pages;
    uint32_t stick_size;
};

PageMap tile_page_map(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats) {
    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    const uint32_t ht = (shape[ndim - 2] + tt::constants::TILE_HEIGHT - 1) / tt::constants::TILE_HEIGHT;
    const uint32_t wt = (shape[ndim - 1] + tt::constants::TILE_WIDTH - 1) / tt::constants::TILE_WIDTH;
    std::vector<uint32_t> dim_pages;
    dim_pages.reserve(ndim);
    for (uint32_t i = 0; i + 2 < ndim; ++i) {
        dim_pages.push_back(shape[i]);
    }
    dim_pages.push_back(ht);
    dim_pages.push_back(wt);

    uint32_t lower_pages = 1;
    for (uint32_t d = rep_dim + 1; d < dim_pages.size(); ++d) {
        lower_pages *= dim_pages[d];
    }
    uint32_t volume_tiles = 1;
    for (uint32_t pages : dim_pages) {
        volume_tiles *= pages;
    }
    return {lower_pages, dim_pages[rep_dim], volume_tiles * num_repeats, /*stick_size=*/0};
}

// Rank-general RM (stick) page geometry for a whole-stick (outer/H) repeated axis: one page per
// stick, so the last dim contributes width rather than a page count.
PageMap rm_page_map(const Tensor& input, uint32_t rep_dim, uint32_t num_repeats) {
    const auto& shape = input.logical_shape();
    const uint32_t ndim = shape.rank();
    std::vector<uint32_t> dim_pages;
    dim_pages.reserve(ndim);
    for (uint32_t i = 0; i + 1 < ndim; ++i) {
        dim_pages.push_back(shape[i]);
    }
    dim_pages.push_back(1);

    uint32_t lower_pages = 1;
    for (uint32_t d = rep_dim + 1; d < dim_pages.size(); ++d) {
        lower_pages *= dim_pages[d];
    }
    uint32_t total_src_pages = 1;
    for (uint32_t pages : dim_pages) {
        total_src_pages *= pages;
    }
    return {lower_pages, dim_pages[rep_dim], total_src_pages * num_repeats, shape[ndim - 1] * input.element_size()};
}

Tensor repeat_interleave_via_codegen(
    const Tensor& input, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    const uint32_t normalized_dim = input.logical_shape().get_normalized_index(dim);
    const MemoryConfig mem_config = output_mem_config.value_or(input.memory_config());
    const uint32_t ndim = input.logical_shape().rank();
    // Sole writer of operation_attributes_t.rep_dim; the inverse is ttnn::prim::recover_rep_dim.
    const uint32_t padded_dim = normalized_dim + (ttnn::prim::kRepDimPadRank - ndim);
    const PageMap page_map = input.layout() == Layout::TILE ? tile_page_map(input, normalized_dim, repeats)
                                                            : rm_page_map(input, normalized_dim, repeats);
    return ttnn::prim::repeat_interleave_codegen(
        input,
        padded_dim,
        repeats,
        page_map.lower_pages,
        page_map.rep_dim_pages,
        page_map.total_out_pages,
        page_map.stick_size,
        /*stick_size_out=*/0,
        mem_config);
}

}  // namespace

ttnn::Tensor repeat_interleave_force_native(
    const ttnn::Tensor& input_a, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    return repeat_interleave_native(input_a, repeats, dim, output_mem_config);
}

ttnn::Tensor repeat_interleave_force_codegen(
    const ttnn::Tensor& input_a, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    TT_FATAL(
        repeat_interleave_codegen::supported_by_codegen(input_a, repeats, dim, output_mem_config),
        "repeat_interleave_force_codegen invoked for a case the codegen path does not support "
        "(requires a device-resident, unsharded rank-2..4 input of bfloat16/float32/int32, or of "
        "bfloat8_b in TILE layout, an interleaved output, repeats > 1, and a repeated dim outside "
        "the pages the layout "
        "subdivides -- TILE defers the two sub-tile dims and requires the default 32x32 tile and "
        "a padded shape that is the logical shape rounded up to whole tiles, "
        "ROW_MAJOR defers the within-stick last dim and needs a stick narrow enough for two CB "
        "slots in one core's L1). This entry never "
        "falls back to native, because a forced leg that quietly served native would make any "
        "comparison against native vacuous. Use ttnn::repeat_interleave if you want the case "
        "routed.");
    return repeat_interleave_via_codegen(input_a, repeats, dim, output_mem_config);
}

}  // namespace ttnn::operations::data_movement::detail

namespace ttnn {

Tensor repeat_interleave(
    const Tensor& input_a, uint32_t repeats, int32_t dim, const std::optional<MemoryConfig>& output_mem_config) {
    namespace detail = operations::data_movement::detail;
    namespace repeat_interleave_codegen = operations::data_movement::repeat_interleave_codegen;

    if (operations::data_movement::codegen_supported_on_arch(input_a) &&
        repeat_interleave_codegen::supported_by_codegen(input_a, repeats, dim, output_mem_config) &&
        !repeat_interleave_codegen::is_demoted(input_a, repeats, dim, output_mem_config)) {
        return detail::repeat_interleave_via_codegen(input_a, repeats, dim, output_mem_config);
    }
    return detail::repeat_interleave_native(input_a, repeats, dim, output_mem_config);
}

}  // namespace ttnn
