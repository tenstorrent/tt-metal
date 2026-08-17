// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/topk.hpp"

#include "tt-metalium/work_split.hpp"
#include "tt_stl/assert.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"
#include "ttnn/operations/core/core.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/tensor/shape/shape.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/transpose/transpose.hpp"
#include "ttnn/operations/data_movement/fill_pad/fill_pad.hpp"
#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/experimental/topk_large_indices/topk_large_indices.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"
#include "ttnn/operations/reduction/topk/device/topk_constants.hpp"

#include <cstdint>
#include <limits>

namespace ttnn::operations::reduction::topk {
namespace {
namespace CMAKE_UNIQUE_NAMESPACE {

/**
 * @brief Rounds up K value to nearest tile-aligned boundary
 *
 * @param k The requested number of top elements to find
 * @return The tile-aligned K value (rounded up to next multiple of 32)
 *
 * Example: K=50 -> returns 64 (2 tiles of 32 elements each)
 *          K=32 -> returns 32 (exactly 1 tile)
 *          K=33 -> returns 64 (2 tiles needed)
 */
uint32_t get_nearest_supported_k_value(const uint32_t k, const uint32_t tile_width) {
    return tile_width * tt::div_up(k, tile_width);
}

/**
 * @brief Prepares a tensor for TopK operation by applying required preprocessing transformations
 *
 * The TopK operation has specific input format requirements that necessitate preprocessing:
 * 1. Target dimension must be the last dimension for efficient computation
 * 2. Tensor must be in 4D format for consistent kernel execution
 *
 * This function applies these transformations in the correct sequence to ensure
 * the input tensor is properly formatted for the TopK device operation.
 *
 * Transformation sequence:
 * 1. **Dimension reordering**: If the target dimension is not already last, transpose
 *    the tensor to move the target dimension to the last position (-1 index)
 * 2. **Rank normalization**: Convert tensor to 4D format if it's not already,
 *    either by padding with unit dimensions or reshaping higher-rank tensors
 *
 * @param input_tensor The tensor to be preprocessed for TopK operation
 * @param dim The target dimension along which TopK will be performed (original index)
 * @param is_dim_last_idx Whether the target dimension is already the last dimension
 * @param is_rank_le_4d Whether the original tensor rank is less than or equal to 4D
 * @return Preprocessed tensor ready for TopK device operation
 */
Tensor pre_topk_transform_tensor(
    const Tensor& input_tensor, const int8_t dim, const bool is_dim_last_idx, const bool is_rank_le_4d) {
    auto transposed_tensor = reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);
    return reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);
}

/**
 * @brief Applies post-processing transformations after TopK execution
 *
 * After the TopK operation executes, several transformations are needed
 * to restore the output tensors to the expected format:
 *
 * 1. Slice adjustment: If K was rounded up for tile alignment, slice to actual K
 * 2. Rank restoration: Convert from 4D back to original tensor rank
 * 3. Dimension reordering: Transpose back if target dimension wasn't last
 * 4. Shape correction: Final slicing to match exact logical output shape
 *
 * This function consolidates all these transformations in correct order to ensure
 * the output matches user expectations while handling OP constraints.
 *
 * @param input_tensor Original input tensor (for memory config reference)
 * @param result Vector containing [values_tensor, indices_tensor] from TopK op
 * @param dim The dimension along which TopK was performed
 * @param is_dim_last_idx Whether the target dimension was already last
 * @param k The actual requested K value
 * @param adjusted_k The tile-aligned K value used by OP
 * @param original_lshape The original logical shape before transformations
 * @param input_memory_config Memory configuration to use for operations
 * @param sub_core_grids Core grid configuration (unused currently)
 * @param indices_tensor Optional indices tensor (unused currently)
 * @return Vector of [transformed_values, transformed_indices] tensors
 */
std::vector<Tensor> post_topk_transform_tensor(
    const Tensor& input_tensor,
    std::vector<Tensor>& result,
    const int8_t dim,
    const bool is_dim_last_idx,
    const uint32_t k,
    const uint32_t adjusted_k,
    const Shape& original_lshape,
    const MemoryConfig& input_memory_config) {
    const auto& input_shape = input_tensor.logical_shape();
    const auto orig_rank = input_shape.rank();

    // Calculate the expected final logical shape after all transformations
    Shape final_lshape = original_lshape;
    final_lshape[dim] = std::min(original_lshape[dim], k);

    // Slice adjustment for tile-aligned K values
    // OP requires K to be tile-aligned (multiples of 32), but user wants exact K
    // If we had to round up K for op, slice down to the requested K value
    if (adjusted_k != k) {
        const auto output_shape = result[0].logical_shape();
        ttsl::SmallVector<uint32_t> step = {1, 1, 1, 1};
        ttsl::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttsl::SmallVector<uint32_t> end_index = {output_shape[0], output_shape[1], output_shape[2], k};

        // Slice both values and indices tensors to remove extra elements beyond requested K
        result[0] = ttnn::slice(result[0], start_index, end_index, step, input_memory_config);
        result[1] = ttnn::slice(result[1], start_index, end_index, step, input_memory_config);
    }

    // Rank restoration - convert from op-required 4D back to original rank
    if (orig_rank < 4 && orig_rank > 1) {
        // For tensors originally < 4D, squeeze out the extra dimensions that were added
        // This doesn't work for 1D tensors in tile layout, because their padded shape is
        // (1, 1, 32, 32) in tile layout, which cannot be squeezed to 1D since only
        // dimensions of size 1 can be squeezed. We handle that case separately below
        result[0] = ttnn::squeeze_from_4D(result[0], orig_rank);
        result[1] = ttnn::squeeze_from_4D(result[1], orig_rank);
    } else if (orig_rank != 4) {
        // For tensors originally > 4D, or 1D, reshape back to original higher-dimensional structure
        ttsl::SmallVector<uint32_t> result_shape(input_shape.cbegin(), input_shape.cend());
        result_shape[result_shape.size() - 1] = k;  // Update last dimension to K
        result[0] = ttnn::reshape(result[0], ttnn::Shape{result_shape});
        result[1] = ttnn::reshape(result[1], ttnn::Shape{result_shape});
    }

    // Dimension reordering - restore original dimension order
    // If we transposed the target dimension to be last for op processing,
    // transpose it back to its original position
    if (!is_dim_last_idx) {
        result[0] = ttnn::transpose(result[0], dim, -1, input_tensor.memory_config());
        result[1] = ttnn::transpose(result[1], dim, -1, input_tensor.memory_config());
    }

    // Final shape correction - ensure exact logical shape match
    // After all rank and dimension transformations, the logical shape might still not
    // match exactly due to padding or other op-imposed constraints.
    // This final slice ensures the output has the exact expected logical shape.
    if (result[0].logical_shape() != final_lshape) {
        int rank = final_lshape.rank();

        // Build slice parameters to extract exactly the expected shape
        ttsl::SmallVector<uint32_t> step;
        ttsl::SmallVector<uint32_t> start_index;
        ttsl::SmallVector<uint32_t> end_index;

        for (int i = 0; i < rank; i++) {
            step.push_back(1);                     // No skipping, take every element
            start_index.push_back(0);              // Start from beginning in each dimension
            end_index.push_back(final_lshape[i]);  // End at desired dimension size
        }

        // Apply final corrective slice to both values and indices tensors
        result[0] = ttnn::slice(result[0], start_index, end_index, step, input_memory_config);
        result[1] = ttnn::slice(result[1], start_index, end_index, step, input_memory_config);
    }

    // Verify that all transformations resulted in the correct output shape
    TT_FATAL(
        result[0].logical_shape() == final_lshape,
        "Output tensor transformation failed to create correct output shape!");

    return result;
}

// ---------------------------------------------------------------------------
// Large-k routing onto ttnn::experimental::topk_large_indices (Blackhole)
// ---------------------------------------------------------------------------
//
// The device op's multi-core bitonic path is gated at k <= 64 (plus pow2 width
// and width < 65536), so every larger k runs the single-core factory: k=512 at
// W=65536 measures ~158 ms on one core. topk_large_indices is a Blackhole-only
// multi-core top-k that has NONE of those gates (arbitrary width up to 2^30,
// k up to 2048), but returns only UINT32 row-major indices, sorted by value
// descending. This composite routes eligible calls through it:
//
//   TILE bf16 -> [untilize only when the input has >1 flattened row]
//   -> topk_large_indices(return_values, tile_output, index_dtype) emitting
//   BOTH the indices (UINT16 when the padded width fits 16 bits, else UINT32)
//   and the BFLOAT16 values (sorted descending, straight from the op's DST —
//   no gather), natively in TILE layout for k_rounded <= 1024 -> the shared
//   post_topk_transform (slice to user k, rank/dim restore). Single-row
//   inputs feed the op their TILE tensor directly (its tile reader pulls
//   just the row's face runs). k_rounded == 2048 keeps the RM output +
//   tilize (+ typecast) tail — measured faster there; see
//   large_k_route_tile_output_max_k.
//
// Routing lives here at the composite level, NOT in the device op's
// select_program_factory, so the device op's program hash is untouched.
//
// Contract matching vs the device op:
//   * values: input dtype (bf16), TILE, sorted descending — satisfies both
//     sorted=true and sorted=false callers (torch.topk(largest=True) order).
//   * indices: the device op emits UINT16 when the (tile-padded) width fits
//     16 bits and UINT32 otherwise (see compute_output_specs); the routed
//     path passes index_dtype=UINT16 at that exact boundary so the op's
//     writer narrows natively (bit-identical to the former typecast).
//   * -inf lanes: topk_large_indices marks lanes whose value is exactly bf16
//     -inf with the sentinel index 0xFFFFFFFF instead of a real position,
//     and its values output carries exact bf16 -inf on those lanes (matching
//     torch's values). The INDEX for a -inf lane stays the sentinel (0xFFFF
//     under the UINT16 contract) — a known, benign divergence: the stock path
//     is itself loose there (it can return indices pointing into its own
//     -inf padding beyond the logical width).
//
// Expected op count and cost shape (bench note):
//   [untilize, multi-row only] + topk_large_indices
//   [+ 2x slice when k was rounded]  ~= 1-4 dispatches. (History: the op's
//   values output first replaced a gather + sentinel eq/where chain; the
//   tile_output/index_dtype opt-ins then deleted the tilize-values,
//   tilize-indices, and typecast stages — measured 60-120 us of single-core
//   TilizeWithValPadding at k <= 1024 — and single-row calls also dropped
//   the untilize (18 us at W=65536: it read the full 32-row tile padding for
//   one logical row).) The multi-row untilize reads real rows, grid-parallel
//   and DRAM-bandwidth-bound: no extra width gating is warranted. Host
//   dispatch dominates only when the row is small — which the k > 64 gate
//   already bounds.

// k <= 64 keeps the device op's fast multi-core bitonic path when that path
// is eligible (padded width in [8192, 65535), power of two, cost check).
// When it is NOT eligible the stock op falls to the single-core factory,
// which is linear in width (~137 ns/elem measured on p150a: 695 us at
// W=5000, 1.38 ms at 10000, 6.87 ms at 50000, 9.49 ms at 65536) while the
// routed composite is tens of microseconds. The small-k arm below therefore
// routes exactly the structurally-ineligible cells; eligible pow2 cells keep
// the bitonic path unchanged. A second, disjoint measured region — the
// MoE-gate arm (see gate_route_* constants below) — additionally claims the
// decode-class expert-gate shapes (k <= 16, padded width 128..512, <= 32
// rows), which are below the bitonic's 8192 width floor and therefore also
// single-core-bound in stock form.
constexpr uint32_t large_k_route_min_k_exclusive = 64;
// Small-k arm floor (on the tile-padded width): below this the single-core
// fallback is already sub-ms and the routed composite's fixed envelope is
// not an empirically proven win. Measured win region starts at W=5000
// (695 us stock vs tens of us routed); 4096 keeps a conservative floor.
constexpr uint32_t small_k_route_min_padded_width = 4096;
// topk_large_indices LLK ceiling.
constexpr uint32_t large_k_route_max_k = 2048;
// The routed pipeline asks the op for TILE-layout outputs (tile_output=true),
// which requires k to be a multiple of 32 (the op itself accepts multiples of
// 16). post_topk_transform slices back down to the user's k either way.
constexpr uint32_t large_k_route_k_multiple = 32;
// Conservative routed-width envelope: topk_large_indices itself allows up to
// 2^30 columns, but the routed composite is silicon-validated to 131072;
// 2^19 keeps a margin without excluding realistic vocab/logit widths. Wider
// rows fall back to the stock path.
constexpr uint32_t large_k_route_max_width = 1u << 19;
// MoE-gate arm (second measured region of the small-k arm): decode-class MoE
// expert gates select k <= 16 experts out of a few hundred, on <= 1 tile row
// of tokens. Measured cells (p150a, bare-op ceilings vs stock single-core):
//   gpt-oss gate  32x128  k=4  -> 4.07 us op vs 24.2 us stock (5.9-6.3x)
//   qwen3.5 gate  32x512  k=10 -> 4.04 us op vs 77.5 us stock (19.2-20.2x)
// Every padded width in [128, 512] is below the multi-core bitonic's
// multi_core_min_width = 8192 (topk_constants.hpp), so this arm can only ever
// replace the linear single-core factory, never the bitonic. The bounds are a
// measured-contiguous-region gate (both endpoints measured, everything
// between is the same shape class), NOT a hardware limit: k <= 16 keeps
// k_rounded at exactly one LLK window (16), and the <= 32-row cap pins the
// decode-token class the measurement covered. Unmeasured neighbours (k in
// (16, 64], widths in (512, 4096), > 32 rows e.g. MoE prefill) keep today's
// engine; widen only with new measurements.
constexpr uint32_t gate_route_max_k = 16;
constexpr uint32_t gate_route_min_padded_width = 128;
constexpr uint32_t gate_route_max_padded_width = 512;
constexpr uint32_t gate_route_max_flattened_rows = 32;

bool should_route_to_topk_large_indices(
    const Tensor& transformed_tensor,
    const uint32_t k,
    const bool largest,
    const bool stable,
    const bool is_dim_last_idx,
    const bool has_user_indices_tensor,
    const bool has_preallocated_outputs,
    const bool has_sub_core_grids) {
    // topk_large_indices only produces largest-first (descending) results.
    if (!largest) {
        return false;
    }
    // stable=true promises lowest-index tie-breaking; topk_large_indices tie
    // order is deterministic but unspecified.
    if (stable) {
        return false;
    }
    // The routed pipeline creates its own index tensor and fresh outputs, and
    // ignores custom core grids; keep the stock path for all three.
    if (has_user_indices_tensor || has_preallocated_outputs || has_sub_core_grids) {
        return false;
    }
    // Conservative: only route reductions that were already on the last dim.
    if (!is_dim_last_idx) {
        return false;
    }
    if (k > large_k_route_max_k) {
        return false;
    }
    if (k <= large_k_route_min_k_exclusive) {
        const uint32_t padded_width = transformed_tensor.padded_shape()[-1];
        // Wide arm: route only cells the device op's multi-core bitonic
        // cannot take (mirrors select_program_factory requirements #1-#2,
        // topk_device_operation.cpp:66-72 — padded width must be < 65535 and
        // a power of two), where stock otherwise falls to the linear
        // single-core factory. Cells that fail only verify_multi_core_cost
        // stay on the stock path (unchanged, conservative).
        const bool is_pow2 = padded_width != 0 && (padded_width & (padded_width - 1)) == 0;
        const bool multicore_structurally_ineligible = padded_width >= std::numeric_limits<uint16_t>::max() || !is_pow2;
        const bool wide_arm = multicore_structurally_ineligible && padded_width >= small_k_route_min_padded_width;
        // MoE-gate arm: k <= 16 (one LLK window), padded width in [128, 512],
        // <= 32 flattened rows — the measured decode-gate region (see the
        // gate_route_* constants above). Everything here is below the
        // bitonic's 8192 width floor, so only the single-core factory is ever
        // displaced. transformed_tensor is 4D with the target dim last, so
        // logical_volume / width is exactly the flattened row count (zero
        // volume already returned earlier in ttnn::topk).
        const uint64_t flattened_rows = transformed_tensor.logical_volume() / transformed_tensor.logical_shape()[-1];
        const bool gate_arm = k <= gate_route_max_k && padded_width >= gate_route_min_padded_width &&
                              padded_width <= gate_route_max_padded_width &&
                              flattened_rows <= gate_route_max_flattened_rows;
        if (!wide_arm && !gate_arm) {
            return false;
        }
    }
    if (transformed_tensor.dtype() != DataType::BFLOAT16) {
        return false;
    }
    if (transformed_tensor.layout() != Layout::TILE) {
        return false;
    }
    if (transformed_tensor.memory_config().is_sharded()) {
        return false;
    }
    if (transformed_tensor.device()->arch() != tt::ARCH::BLACKHOLE) {
        return false;
    }
    const uint32_t width = transformed_tensor.logical_shape()[-1];
    const uint32_t k_rounded = large_k_route_k_multiple * tt::div_up(k, large_k_route_k_multiple);
    // topk_large_indices needs width >= its (rounded) k; the conservative
    // envelope bounds the width from above. No pow2 / 16-bit width
    // requirements here — that is the point of the route.
    return width >= k_rounded && width <= large_k_route_max_width;
}

// TILE-native output ceiling (measured policy, p150a stage profile 2026-08-17):
// tilize of a [rows, k_rounded] RM output lands on TilizeWithValPadding's
// pathological single-core factory whenever the padded output has <= 32 tiles
// (k_rounded <= 1024) — 20-78 us per tilize there, which the op's native
// tile-scatter writer replaces for ~2-5 us of scattered slice writes. At
// k_rounded == 2048 the tilize is multi-core and cheap (3-4 us) while the
// native scatter costs ~10 us of small writes per stream, so the RM-output +
// tilize chain stays the faster arm and is kept for k > 1024.
constexpr uint32_t large_k_route_tile_output_max_k = 1024;

// TILE-native input floor (measured policy, p150a stage profile 2026-08-17):
// feeding the op its TILE input directly (deleting the untilize) pays only
// when untilize's 32x tile-padding read amplification outweighs the tile
// reader's staged 64-byte scatter reads. Measured on single-row cells: at
// padded W=65536 untilize costs ~18 us vs ~7 us of reader overhead (win);
// at W=2048 untilize is ~1.6-1.9 us vs ~5.5 us of reader overhead (loss).
// The crossover sits in the low tens of thousands of columns; 32768 keeps a
// safe margin. Narrower (or multi-row) inputs keep the untilize stage.
constexpr uint32_t large_k_route_tile_input_min_width = 32768;

// Runs the routed pipeline on the 4D, last-dim-target TILE bf16 tensor.
// Returns {values, indices} in TILE layout with last dim k_rounded, matching
// what ttnn::prim::topk would have produced for adjusted_k == k_rounded, so
// the shared post_topk_transform_tensor handles the rest.
//
// Input side: for a single (flattened) row at padded width >=
// large_k_route_tile_input_min_width the op consumes the TILE input directly
// (its tile reader pulls only the row's own face runs), deleting the untilize
// stage and its 32x tile-padding read amplification; narrower single rows and
// multi-row inputs keep the untilize (grid-parallel and cheap there, while
// the tile reader's staged scatter reads are not).
//
// Output side, k_rounded <= 1024 (see large_k_route_tile_output_max_k): the
// op emits TILE-layout outputs natively (tile_output=true; zero-filled tile
// padding, matching what tilize_with_val_padding produced here before) and,
// when the device op's index-dtype contract calls for UINT16, emits UINT16
// indices natively too — the former tilize-values, tilize-indices, and
// typecast stages are gone. k_rounded == 2048 keeps the RM-output + tilize
// (+ typecast) chain, which measures faster there.
std::vector<Tensor> run_topk_large_indices_route(const Tensor& transformed_tensor, const uint32_t k_rounded) {
    const auto& lshape = transformed_tensor.logical_shape();
    const uint32_t flattened_rows =
        ttnn::operations::experimental::topk_large_indices::flattened_rows_excluding_last_dim(lshape);
    const uint32_t padded_width = transformed_tensor.padded_shape()[-1];
    const bool tile_native_input = flattened_rows == 1 && padded_width >= large_k_route_tile_input_min_width;
    const Tensor op_input =
        tile_native_input ? transformed_tensor : ttnn::to_layout(transformed_tensor, Layout::ROW_MAJOR);

    // Multi-row rectangle trees: when every row can own a concurrent P-core
    // tree (rows <= grid tiling capacity) and the op's cost model says the
    // tree wins, pass num_slices explicitly. allow_multi_row=true is
    // ROUTING'S opt-in, not the op's: routed calls already changed engine vs
    // stock (tie order re-audited at the I5 relaxation), so the rect engine's
    // different-but-equal tie identity is within the same acceptance class.
    // rows > capacity come back disabled and keep nullopt — the op's internal
    // hybrid row split handles rows > grid on its own. The MoE-gate shapes
    // (single-chunk rows) also come back disabled. Trade-off: rects are
    // ROW_MAJOR-output only, so k_rounded <= 1024 cells give up the native
    // TILE/u16 writer and pay the to_layout(+typecast) tail below — measured
    // net-positive at the sampling shapes (32 x 65536-class rows).
    const auto rect_cfg = ttnn::operations::experimental::topk_large_indices::program::compute_column_split_config(
        k_rounded,
        lshape[-1],
        flattened_rows,
        transformed_tensor.device()->compute_with_storage_grid_size(),
        std::nullopt,
        /*allow_multi_row=*/true);
    const bool use_rects = flattened_rows > 1 && rect_cfg.enabled && rect_cfg.num_slices >= 2;

    // Match the device op's index dtype contract: UINT16 iff the tile-padded
    // width fits 16 bits (compute_output_specs compares the padded shape).
    const bool emit_u16 = padded_width <= std::numeric_limits<uint16_t>::max();
    const bool tile_native_output = !use_rects && k_rounded <= large_k_route_tile_output_max_k;

    // BFLOAT16 values + (UINT32 or UINT16) indices of the top k_rounded per
    // row, both sorted by value descending; -inf value lanes carry exact
    // bf16 -inf values and the sentinel index (0xFFFF under the UINT16
    // contract, exactly what a UINT32 -> UINT16 typecast produces).
    auto [values, indices] = ttnn::experimental::topk_large_indices_with_values(
        op_input,
        k_rounded,
        /*valid_length=*/std::nullopt,
        use_rects ? std::optional<uint32_t>(rect_cfg.num_slices) : std::nullopt,
        /*tile_output=*/tile_native_output,
        (tile_native_output && emit_u16) ? std::optional<DataType>(DataType::UINT16) : std::nullopt);

    if (!tile_native_output) {
        values = ttnn::to_layout(values, Layout::TILE);
        indices = ttnn::to_layout(indices, Layout::TILE);
        if (emit_u16) {
            indices = ttnn::typecast(indices, DataType::UINT16);
        }
    }

    std::vector<Tensor> result;
    result.reserve(2);
    result.push_back(std::move(values));
    result.push_back(std::move(indices));
    return result;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

}  // namespace ttnn::operations::reduction::topk

namespace ttnn {

std::vector<Tensor> topk(
    const Tensor& input_tensor,
    const uint32_t k,
    const int8_t dim,
    const bool largest,
    const bool sorted,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Tensor>& indices_tensor,
    std::optional<std::tuple<Tensor&, Tensor&>> preallocated_output_tensors,
    const bool stable) {
    // Store original shape for final output validation
    const ttnn::Shape& original_lshape = input_tensor.logical_shape();

    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");

    // Analyze input tensor properties to determine required transformations
    std::size_t rank_st = input_tensor.logical_shape().rank();
    TT_FATAL(rank_st <= std::numeric_limits<int8_t>::max(), "Rank is too large to convert to int8_t");
    const int8_t rank = static_cast<int8_t>(rank_st);

    TT_FATAL(
        (dim >= -rank && dim <= (rank - 1)) || (rank == 0 && (dim == 0 || dim == -1)),
        "Dimension for topk is out of range (expected to be in range of [{}, {}])",
        -rank,
        rank - 1);

    const bool is_dim_last_idx = (dim == -1 || dim == rank - 1 || (rank == 0 && dim == 0));
    const bool is_rank_le_4d = rank <= 4;

    // Normalize negative dimension index and validate K parameter
    // Rank 0 means scalar, so the only valid dimension is 0.
    const int8_t adjusted_dim = (rank == 0) ? 0 : ((dim < 0) ? (dim + rank) : dim);

    TT_FATAL(
        (rank == 0 && k <= 1) || input_tensor.logical_shape()[adjusted_dim] >= k,
        "K cannot be larger than the dimension size! K={}, dimension size={}",
        k,
        (rank == 0) ? 1 : input_tensor.logical_shape()[adjusted_dim]);
    ttnn::Shape desired_final_shape = original_lshape;
    if (rank != 0) {
        desired_final_shape[adjusted_dim] = k;
    }

    if (preallocated_output_tensors.has_value()) {
        const Tensor& preallocated_values = std::get<0>(preallocated_output_tensors.value());
        const Tensor& preallocated_indices = std::get<1>(preallocated_output_tensors.value());
        TT_FATAL(is_device_tensor(preallocated_values), "Preallocated output values tensor must be on device");
        TT_FATAL(is_device_tensor(preallocated_indices), "Preallocated output indices tensor must be on device");

        TT_FATAL(
            preallocated_values.logical_shape() == desired_final_shape,
            "Preallocated values tensor has incorrect shape! Got : {}, expected: {}",
            preallocated_values.logical_shape(),
            desired_final_shape);
        TT_FATAL(
            preallocated_indices.logical_shape() == desired_final_shape,
            "Preallocated indices tensor has incorrect shape! Got : {}, expected: {}",
            preallocated_indices.logical_shape(),
            desired_final_shape);
    }

    // When input is a scalar, PyTorch returns the copy of the input tensor (that is the 1 top element)
    // as values. This happens when k = 1 (which makes sense), but also when k = 0 (which is unusual,
    // but for now we simply match its behavior).
    if (rank == 0 && (k == 1 || k == 0)) {
        // Caller requested top 1 elements from a scalar tensor.
        // We need to return a copy of the input tensor as is (that is the 1 top element), and
        // a scalar tensor containing a single 0 as the index (matches what PyTorch does).
        if (!preallocated_output_tensors.has_value()) {
            return {
                ttnn::clone(
                    input_tensor, /*dtype=*/std::nullopt, memory_config, /*compute_kernel_config=*/std::nullopt),
                ttnn::full(
                    desired_final_shape,
                    0,
                    tt::tt_metal::DataType::UINT16,
                    input_tensor.layout(),
                    *(input_tensor.device()),
                    memory_config.value_or(input_tensor.memory_config()))};
        }

        // If the output tensors were preallocated, they should already have
        // the correct shapes (validated above), so fill the values with the input tensor
        // and a zero as the index and return.
        auto& [values, indices] = preallocated_output_tensors.value();
        ttnn::copy(input_tensor, values);

        // Creating indices tensor on host and copying to device (there is no direct way to write
        // to a device tensor with a scalar value).
        const tt::tt_metal::TensorSpec& indices_spec = indices.tensor_spec();
        TT_FATAL(indices_spec.data_type() == DataType::UINT16, "Indices tensor must be UINT16 for rank 0 input tensor");
        // Although we only need to store one value, have to account for extra padding
        // in the tile layout. So host buffer size needs to match device buffer size.
        auto indices_vec =
            std::vector<uint16_t>(indices_spec.physical_shape().height() * indices_spec.physical_shape().width(), 0);
        Tensor host_indices(
            tt::tt_metal::HostBuffer(std::move(indices_vec)), desired_final_shape, DataType::UINT16, indices.layout());
        copy_to_device(host_indices, indices);

        return {values, indices};
    }

    // For a zero volume input tensor, return a zero volume tensor with the shape adjusted for k.
    // Same if k is 0 (i.e. top 0 elements were requested).
    if (input_tensor.logical_volume() == 0 || k == 0) {
        if (!preallocated_output_tensors.has_value()) {
            auto make_zero_volume_tensor = [&] {
                return ttnn::full(
                    desired_final_shape,
                    0.0f,  // Value doesn't matter, since it is a 0-volume tensor.
                    input_tensor.dtype(),
                    input_tensor.layout(),
                    *input_tensor.device(),
                    memory_config.value_or(input_tensor.memory_config()));
            };

            return {make_zero_volume_tensor(), make_zero_volume_tensor()};
        }

        // If the output tensors were preallocated, they should already have
        // the correct shapes (validated above), so just return them as is.
        auto& [values, indices] = preallocated_output_tensors.value();
        return {values, indices};
    }

    // Set up memory and execution configurations with defaults if not provided
    const auto input_memory_config = memory_config.value_or(input_tensor.memory_config());
    const auto compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    const uint32_t max_number_of_cores = compute_with_storage_grid_size.y * compute_with_storage_grid_size.x;
    const auto full_core_grids =
        tt::tt_metal::num_cores_to_corerangeset(max_number_of_cores, compute_with_storage_grid_size, true);
    const auto used_sub_core_grids = sub_core_grids.value_or(full_core_grids);

    // OP constraint: K must be tile-aligned (multiple of 32 elements)
    // Round up to nearest supported value for OP execution
    const uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const uint32_t adjusted_k =
        operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::get_nearest_supported_k_value(k, tile_width);

    // Dimension reordering - move target dimension to last position
    Tensor transposed_tensor = ::reduction_common::perform_transpose(input_tensor, is_dim_last_idx, dim, -1);

    // Rank normalization - convert to 4D tensor format
    Tensor transformed_tensor = ::reduction_common::transform_to_4d_tensor(transposed_tensor, is_rank_le_4d);

    // Blackhole routing onto ttnn::experimental::topk_large_indices covers
    // three regions that otherwise land on the linear single-core factory:
    // k in (64, 2048] (off the multi-core k <= 64 gate); k <= 64 rows whose
    // padded width is structurally ineligible for the multi-core bitonic
    // (>= 65535 or non-pow2); and the MoE-gate region (k <= 16, padded width
    // in [128, 512], <= 32 rows). See the comment block on
    // should_route_to_topk_large_indices for the full predicate and contract.
    if (operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::should_route_to_topk_large_indices(
            transformed_tensor,
            k,
            largest,
            stable,
            is_dim_last_idx,
            indices_tensor.has_value(),
            preallocated_output_tensors.has_value(),
            sub_core_grids.has_value())) {
        const uint32_t k_rounded =
            operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::large_k_route_k_multiple *
            tt::div_up(k, operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::large_k_route_k_multiple);
        auto routed_result = operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::run_topk_large_indices_route(
            transformed_tensor, k_rounded);
        return operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::post_topk_transform_tensor(
            transposed_tensor, routed_result, dim, is_dim_last_idx, k, k_rounded, original_lshape, input_memory_config);
    }

    // Dimension size padding - ensure minimum dimension size for efficient processing
    auto padded_tensor = transformed_tensor;
    const auto current_dim_size = static_cast<int>(transformed_tensor.logical_shape()[-1]);
    const auto min_required_size = static_cast<int>(ttnn::prim::constants::min_dim_per_core);
    const auto pad_amount = std::max(min_required_size - current_dim_size, 0);

    // Choose padding value based on whether we want largest or smallest values
    const auto pad_val = largest ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();

    if (pad_amount > 0) {
        ttsl::SmallVector<std::array<uint32_t, 2>> padding = {{0, 0}, {0, 0}, {0, 0}, {0, pad_amount}};

        // Use multicore padding for BFLOAT16 tensors not in L1 memory for better performance
        const bool pad_multicore = transformed_tensor.dtype() == DataType::BFLOAT16 &&
                                   transformed_tensor.memory_config().buffer_type() != BufferType::L1;
        padded_tensor = ttnn::pad(transformed_tensor, padding, pad_val, pad_multicore);
    }

    // Fill any implicit tile padding with appropriate values
    padded_tensor = ttnn::fill_implicit_tile_padding(padded_tensor, pad_val);

    // Preprocess optional output tensors if provided
    std::optional<std::tuple<Tensor, Tensor>> output_tensors = std::nullopt;
    if (preallocated_output_tensors.has_value()) {
        const auto preallocated_values = operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::pre_topk_transform_tensor(
            std::get<0>(preallocated_output_tensors.value()), dim, is_dim_last_idx, is_rank_le_4d);
        const auto preallocated_indices =
            operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::pre_topk_transform_tensor(
                std::get<1>(preallocated_output_tensors.value()), dim, is_dim_last_idx, is_rank_le_4d);
        output_tensors = {preallocated_values, preallocated_indices};
    } else {
        output_tensors = std::optional<std::tuple<Tensor, Tensor>>{std::nullopt};
    }

    // Execute TopK operation
    auto [output_value_tensor, output_index_tensor] = ttnn::prim::topk(
        padded_tensor,
        adjusted_k,
        -1,
        largest,
        sorted,
        stable,
        input_memory_config,
        used_sub_core_grids,
        indices_tensor,
        output_tensors);

    // Package results into vector format expected by post-processing
    std::vector<Tensor> output_tensor_vec;
    output_tensor_vec.reserve(2);
    output_tensor_vec.push_back(std::move(output_value_tensor));
    output_tensor_vec.push_back(std::move(output_index_tensor));

    // Save pre-transformed tensor buffers for comparison (needed to check if device op changed buffers)
    auto* pre_transform_buffer_0 =
        preallocated_output_tensors.has_value() ? std::get<0>(output_tensors.value()).buffer() : nullptr;
    auto* pre_transform_buffer_1 =
        preallocated_output_tensors.has_value() ? std::get<1>(output_tensors.value()).buffer() : nullptr;

    // Apply post-processing transformations to restore original format
    auto post_transform_output_tensors =
        operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::post_topk_transform_tensor(
            transposed_tensor,
            output_tensor_vec,
            dim,
            is_dim_last_idx,
            k,
            adjusted_k,
            original_lshape,
            input_memory_config);

    // Check if padding or dtype conversion changed buffer address
    if (preallocated_output_tensors.has_value()) {
        if (std::get<0>(preallocated_output_tensors.value()).buffer() != pre_transform_buffer_0) {
            std::get<0>(preallocated_output_tensors.value()) = post_transform_output_tensors.at(0);
        }
        if (std::get<1>(preallocated_output_tensors.value()).buffer() != pre_transform_buffer_1) {
            std::get<1>(preallocated_output_tensors.value()) = post_transform_output_tensors.at(1);
        }
    }
    return post_transform_output_tensors;
}

}  // namespace ttnn
