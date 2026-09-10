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
#include "ttnn/operations/experimental/topk_large_indices/topk_large_indices.hpp"
#include "ttnn/operations/reduction/reduction_common/reduction_common.hpp"
#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"
#include "ttnn/operations/reduction/topk/device/topk_constants.hpp"
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"
#include "ttnn/operations/reduction/topk/device/topk_route_prep_device_operation.hpp"
#include "ttnn/operations/reduction/topk/device/topk_route_finish_device_operation.hpp"

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
// k up to 2048; internally it splits rows across the grid when rows exceed it
// and fuses segment merges inside a row), but returns only UINT32 row-major
// indices of a ROW_MAJOR input, sorted by value descending. This composite
// routes eligible calls through it:
//
//   TILE bf16 -> fused untilize+clamp to the lowest finite bf16 (the private
//   topk_route_prep device op; -inf parity, see below) -> topk_large_indices
//   on the CLAMPED tensor (k rounded up to a multiple of 16, its own
//   constraint) -> fused gather of the ORIGINAL (unclamped) values straight
//   from the TILE source + TILE assembly of both outputs + index dtype emit
//   (the private topk_route_finish device op) -> the shared
//   post_topk_transform (slice to user k, rank/dim restore).
//
// Routing lives here at the composite level, NOT in the device op's
// select_program_factory, so the device op's program hash is untouched.
//
// Contract matching vs the device op:
//   * values: input dtype (bf16), TILE, sorted descending — satisfies both
//     sorted=true and sorted=false callers (torch.topk(largest=True) order).
//   * indices: the device op emits UINT16 when the (tile-padded) width fits
//     16 bits and UINT32 otherwise (see compute_output_specs);
//     topk_route_finish emits that exact boundary directly.
//   * -inf lanes (the clamp trick): topk_large_indices marks lanes whose
//     value is exactly bf16 -inf with the sentinel index 0xFFFFFFFF instead
//     of a real position. The routed path never lets the op see a -inf: the
//     op's input is first clamped to the lowest FINITE bf16 (bit pattern
//     0xFF7F, ~-3.39e38), so every lane keeps a REAL stamped source position
//     and the sentinel never fires. The gather then runs against the
//     ORIGINAL tensor, so -inf VALUES come back bit-exact. Net: stock/torch
//     parity for both values and indices on -inf rows, and every returned
//     index is in-range (the op's own -inf width padding can never outrank a
//     clamped-finite lane, and width >= k_rounded is a predicate invariant).
//     One documented caveat: a genuine 0xFF7F input element ties with
//     clamped -inf lanes, so their relative order (and which of them makes
//     the k cut) follows the op's deterministic-but-unspecified tie order —
//     inside ttnn.topk's documented non-stable contract (stable=true never
//     routes).
//
// Cost shape: only topk_route_prep and topk_large_indices touch the full
// input width; every other stage works on [rows, k_rounded <= 2048] tensors
// (topk_route_finish reads 64 B per selected element, not whole rows).

// k <= 64 keeps the device op's fast multi-core bitonic path for pow2 widths
// at or above multi_core_min_width (plus the grid/L1 cost check). Below that
// floor the stock op either falls to the single-core factory — linear in
// width (~137 ns/elem measured on p150a: 695 us at W=5000, 1.38 ms at 10000,
// 6.87 ms at 50000, 9.49 ms at 65536) while the routed composite is tens of
// microseconds — or, for pow2 low-tile-row cells eligible via the Ht-aware
// gate, runs a multi-core bitonic that the composite still beats (measured
// at W=4096: 70–166 us stock multi-core vs 47–48 us composite; numbers at
// the wide-arm predicate below). The small-k arm therefore routes every
// >= small_k_route_min_padded_width cell below the plain
// multi_core_min_width floor, pow2 or not; eligible pow2 cells at or above
// the floor keep the stock bitonic path.
constexpr uint32_t large_k_route_min_k_exclusive = 64;
// Small-k arm floor (on the tile-padded width): below this the single-core
// fallback is already sub-ms and the routed composite's fixed envelope is
// not an empirically proven win. Measured win region starts at W=5000
// (695 us stock vs tens of us routed); 4096 keeps a conservative floor.
constexpr uint32_t small_k_route_min_padded_width = 4096;
// topk_large_indices LLK ceiling.
constexpr uint32_t large_k_route_max_k = 2048;
// topk_large_indices requires k to be a multiple of 16. post_topk_transform
// slices back down to the user's k when it was rounded.
constexpr uint32_t large_k_route_k_multiple = 16;
// Routed width ceiling: 2^19 is the conservative, silicon-validated routing
// envelope (the widest routed cells, 262144/524288, are pinned in
// test_topk.py; topk_large_indices itself allows up to 2^30 columns).
// Lifting it is a separate, perf-validated change.
constexpr uint32_t large_k_route_max_width = 1u << 19;

// Python sampling queries this exact predicate through the nanobind helper
// ttnn._ttnn.operations.reduction._sampling_topk_would_route_to_large_indices
// before relaxing its call shape (the binding lives on the reduction
// submodule; only registered ttnn operations are hoisted to top-level ttnn).
// Keep that helper wired to this implementation so policy changes cannot
// drift.
bool should_route_to_topk_large_indices(
    const Tensor& transformed_tensor,
    const uint32_t k,
    const bool largest,
    const bool stable,
    const bool is_dim_last_idx,
    const bool has_user_indices_tensor,
    const bool has_preallocated_outputs,
    const bool has_sub_core_grids,
    const std::optional<MemoryConfig>& user_memory_config) {
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
    // The routed composite produces interleaved outputs; the stock path raises
    // on sharded output configs. Keep the stock (loud) behavior.
    if (user_memory_config.has_value() && user_memory_config->is_sharded()) {
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
        // Wide arm: route the cells where the composite measured faster than
        // whatever the stock op would run, at padded width >=
        // small_k_route_min_padded_width:
        //  * structurally ineligible widths (non-pow2, >= 65535, or a pow2
        //    below multi_core_min_width with more than
        //    multi_core_low_ht_max_tile_rows tile rows): stock falls to the
        //    linear single-core factory (~137 ns/elem) while the composite is
        //    tens of microseconds.
        //  * the pow2 [small_k_route_min_padded_width, multi_core_min_width)
        //    low-tile-row cell: structurally eligible for the stock
        //    multi-core bitonic via the Ht-aware relaxation, but the
        //    composite still wins it — measured on p150a (bf16 largest W=4096, 3 Tracy
        //    trials, <0.2% spread): 70.3 -> 46.8 us (k=32, 1 tile row) up to
        //    165.5 -> 48.0 us (k=64, 2 tile rows).
        // Both collapse to evaluating the SHARED
        // ttnn::prim::topk_multicore_structurally_eligible helper with the
        // low-Ht relaxation disabled (num_tile_rows pinned above
        // multi_core_low_ht_max_tile_rows), i.e. the plain
        // multi_core_min_width floor — which also keeps this router aligned
        // with the models/common/sampling/_utils.py mirror
        // (topk_would_route_to_large_indices). Eligible cells at width >=
        // multi_core_min_width keep the stock bitonic; cells that fail only
        // verify_multi_core_cost (grid/L1 feasibility) stay on the stock path
        // (conservative). When routing is declined for other
        // reasons (sub_core_grids, preallocated outputs, non-bf16, smallest,
        // stable), the device op's own Ht-aware eligibility still upgrades
        // low-tile-row cells from single-core to multi-core.
        const bool multicore_ineligible_ignoring_low_ht_arm = !ttnn::prim::topk_multicore_structurally_eligible(
            padded_width, ttnn::prim::constants::multi_core_low_ht_max_tile_rows + 1, k);
        const bool wide_arm =
            multicore_ineligible_ignoring_low_ht_arm && padded_width >= small_k_route_min_padded_width;
        if (!wide_arm) {
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
    // routed envelope (see large_k_route_max_width) bounds the width from
    // above. No pow2 / 16-bit width requirements here — that is the point of
    // the route.
    return width >= k_rounded && width <= large_k_route_max_width;
}

// Runs the routed pipeline on the 4D, last-dim-target TILE bf16 tensor.
// Returns {values, indices} in TILE layout with last dim k_rounded, matching
// what ttnn::prim::topk would have produced for adjusted_k == k_rounded, so
// the shared post_topk_transform_tensor handles the rest.
std::vector<Tensor> run_topk_large_indices_route(const Tensor& transformed_tensor, const uint32_t k_rounded) {
    // Fused untilize + floor-at-lowest-finite-bf16 (the clamp trick — see the
    // contract block above). A fused max, like the ttnn::maximum it replaced
    // (and unlike ttnn::clamp, whose implicit FLT_MAX upper bound has bf16
    // rounding behavior at +inf we would rather not depend on), cannot
    // perturb +inf lanes. The routing predicate guarantees a TILE-layout
    // input here; the op asserts it (no ROW_MAJOR branch).
    const Tensor clamped_rm = topk_route_prep(transformed_tensor);

    // UINT32 row-major indices of the top k_rounded per row, sorted by value
    // descending; every lane carries a real in-range position.
    const Tensor indices_rm = ttnn::experimental::topk_large_indices(clamped_rm, k_rounded);

    // Gather values by index from the ORIGINAL, unclamped TILE tensor and
    // assemble both TILE outputs (index dtype per the stock op's u16/u32
    // width boundary). Canonicalization note: a tilize datacopy pass would
    // canonicalize bf16 through DEST (-0 -> +0, NaN -> +/-inf);
    // topk_route_finish is pure data movement and copies values BIT-EXACT
    // from the source instead. NaN inputs are outside ttnn.topk's contract
    // on every path.
    return topk_route_finish(transformed_tensor, indices_rm);
}

}  // namespace CMAKE_UNIQUE_NAMESPACE
}  // namespace

}  // namespace ttnn::operations::reduction::topk

namespace ttnn::operations::reduction::topk::detail {

bool sampling_topk_would_route_to_large_indices(const Tensor& input_tensor, const uint32_t k) {
    TT_FATAL(is_device_tensor(input_tensor), "Sampling top-k route query requires an on-device tensor");
    return CMAKE_UNIQUE_NAMESPACE::should_route_to_topk_large_indices(
        input_tensor,
        k,
        /*largest=*/true,
        /*stable=*/false,
        /*is_dim_last_idx=*/true,
        /*has_user_indices_tensor=*/false,
        /*has_preallocated_outputs=*/false,
        /*has_sub_core_grids=*/false,
        /*user_memory_config=*/std::nullopt);
}

}  // namespace ttnn::operations::reduction::topk::detail

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

    if (indices_tensor.has_value()) {
        // The reader streams a caller-supplied indices tensor with the input's page index, so the
        // two have to describe the same grid. Only the dtype was checked, and a narrower indices
        // tensor read at the input's stride returned indices from the wrong pages rather than
        // failing.
        TT_FATAL(
            indices_tensor->logical_shape() == input_tensor.logical_shape(),
            "Indices tensor has incorrect shape! Got : {}, expected: {}",
            indices_tensor->logical_shape(),
            input_tensor.logical_shape());
        // Matching shapes are not enough when the reduced dim is not already last: the transpose
        // below is applied to the input and not to the indices tensor, so the reader pages the
        // indices in the input's post-transpose layout. At [1, 1, 8192, 32] with dim=2 that returns
        // every index rounded down to a multiple of the tile height.
        TT_FATAL(
            is_dim_last_idx,
            "Indices tensor is only supported for a reduction on the last dimension, got dim={} for rank {}",
            dim,
            rank);
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
    // two regions where the composite measures faster than the device op:
    // k in (64, 2048] (above the device op's multi-core k <= 64 gate), and
    // k <= 64 calls whose padded width is >= small_k_route_min_padded_width
    // but fails the plain multi_core_min_width bitonic gate (>= 65535,
    // non-pow2, or pow2 below multi_core_min_width). The router evaluates
    // that gate without the device op's Ht-aware low-tile-row relaxation:
    // the composite also beats the stock multi-core bitonic on those
    // low-tile-row cells. See the comment block on
    // should_route_to_topk_large_indices for the full predicate and contract.
    if (operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::should_route_to_topk_large_indices(
            transformed_tensor,
            k,
            largest,
            stable,
            is_dim_last_idx,
            indices_tensor.has_value(),
            preallocated_output_tensors.has_value(),
            sub_core_grids.has_value(),
            memory_config)) {
        const uint32_t k_rounded =
            operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::large_k_route_k_multiple *
            tt::div_up(k, operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::large_k_route_k_multiple);
        auto routed_result = operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::run_topk_large_indices_route(
            transformed_tensor, k_rounded);
        auto routed_final = operations::reduction::topk::CMAKE_UNIQUE_NAMESPACE::post_topk_transform_tensor(
            transposed_tensor, routed_result, dim, is_dim_last_idx, k, k_rounded, original_lshape, input_memory_config);
        // Stock parity: the device op places outputs in
        // memory_config.value_or(input.memory_config()); the routed composite
        // produces default interleaved outputs and post-transform only applies
        // the config on its slice branches, so conform here.
        for (auto& t : routed_final) {
            if (t.memory_config() != input_memory_config) {
                t = ttnn::to_memory_config(t, input_memory_config);
            }
        }
        return routed_final;
    }

    // Dimension size padding - ensure minimum dimension size for efficient processing
    auto padded_tensor = transformed_tensor;
    const auto current_dim_size = static_cast<int>(transformed_tensor.logical_shape()[-1]);
    const auto min_required_size = static_cast<int>(ttnn::prim::constants::min_dim_per_core);
    const auto pad_amount = std::max(min_required_size - current_dim_size, 0);

    // Choose padding value based on whether we want largest or smallest values
    const auto pad_val = largest ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();

    // A caller-supplied indices tensor must keep describing the same tile grid as the input the
    // device op actually sees, so it is padded in lockstep. The pad slots hold +/-inf values and
    // can never be selected, so the index value used to pad is irrelevant.
    auto padded_indices_tensor = indices_tensor;
    if (pad_amount > 0) {
        ttsl::SmallVector<std::array<uint32_t, 2>> padding = {{0, 0}, {0, 0}, {0, 0}, {0, pad_amount}};

        // Use multicore padding for BFLOAT16 tensors not in L1 memory for better performance
        const bool pad_multicore = transformed_tensor.dtype() == DataType::BFLOAT16 &&
                                   transformed_tensor.memory_config().buffer_type() != BufferType::L1;
        padded_tensor = ttnn::pad(transformed_tensor, padding, pad_val, pad_multicore);
        if (padded_indices_tensor.has_value()) {
            padded_indices_tensor = ttnn::pad(padded_indices_tensor.value(), padding, 0.0f, false);
        }
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
        padded_indices_tensor,
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
