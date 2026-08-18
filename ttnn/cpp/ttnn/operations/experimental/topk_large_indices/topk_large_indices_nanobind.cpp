// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_large_indices_nanobind.hpp"

#include <optional>
#include <tuple>
#include <variant>

#include <nanobind/nanobind.h>
#include <nanobind/stl/variant.h>

#include "ttnn-nanobind/bind_function.hpp"
#include "topk_large_indices.hpp"

namespace ttnn::operations::experimental::topk_large_indices::detail {

namespace {

// Dispatches on return_values so the default call keeps returning a single
// indices tensor (backward compatible) while opting in returns a
// (values, indices) tuple, torch-style.
std::variant<ttnn::Tensor, std::tuple<ttnn::Tensor, ttnn::Tensor>> topk_large_indices_py(
    const ttnn::Tensor& input_tensor,
    uint32_t k,
    std::optional<uint32_t> valid_length,
    bool return_values,
    std::optional<uint32_t> num_slices,
    bool tile_output,
    std::optional<ttnn::DataType> index_dtype) {
    if (return_values) {
        return ttnn::experimental::topk_large_indices_with_values(
            input_tensor, k, valid_length, num_slices, tile_output, index_dtype);
    }
    return ttnn::experimental::topk_large_indices(input_tensor, k, valid_length, num_slices, tile_output, index_dtype);
}

}  // namespace

void bind_topk_large_indices(nb::module_& mod) {
    ttnn::bind_function<"topk_large_indices", "ttnn.experimental.">(
        mod,
        R"doc(
        Experimental Top-K over the last dimension of a BFLOAT16 tensor (ROW_MAJOR or
        TILE layout). This op is Blackhole-only.

        Returns a ROW_MAJOR UINT32 tensor containing sorted descending top-k indices,
        or, with ``return_values=True``, a ``(values, indices)`` tuple where values is
        a ROW_MAJOR BFLOAT16 tensor sorted descending to match the indices.
        The output shape matches the input shape except that the last dimension is k.
        ``tile_output=True`` emits the output tensor(s) in TILE layout instead
        (requires k to be a multiple of 32; the tile padding rows are zero-filled), and
        ``index_dtype=ttnn.uint16`` narrows the indices output (requires the searched
        width to be <= 65535; the -inf sentinel becomes 0xFFFF).

        This op is intended for large row-major rows. Internally it snaps k to the
        nearest supported LLK size and streams each input row in LLK-sized windows.
        Input values equal to -inf produce the sentinel index 0xFFFFFFFF when they
        survive into the final top-k result; with ``return_values=True`` those lanes
        carry exact bf16 -inf values.

        K constraints:
            * k must be in [16, 2048];
            * k must be a multiple of 16;
            * the internal LLK window is snapped to 512, 1024, or 2048 elements.

        Input tensor constraints:
            * the input tensor must be allocated on a Blackhole device;
            * rank must be >= 1;
            * all leading dimensions are flattened into independent rows;
            * the flattened leading-dimension row count must fit in uint32_t;
            * the last dimension is the input row length;
            * the flattened row count must be > 0;
            * the last dimension must be >= k and <= 1,073,741,824 elements.

        valid_length (optional):
            * restricts the search to the first ``valid_length`` columns of each row;
            * the remaining columns are ignored -- neither read nor ranked -- so an
              over-allocated row whose tail is stale can be searched without slicing it;
            * must be in (0, last dimension]; defaults to the full last dimension. A
              prefix shorter than k is allowed: the lanes past its capacity emit the
              sentinel index 0xFFFFFFFF (and -inf values with return_values=True);
            * applied at runtime (no recompile), so a loop growing valid_length reuses one program.

        Args:
            input_tensor: device tensor with ROW_MAJOR or TILE layout and BFLOAT16 dtype.
            k: required number of indices to return.
            valid_length: optional number of leading columns to search (default: full width).
            return_values: also return the top-k values; the result becomes a
                (values, indices) tuple (default: False, indices tensor only).
            num_slices: optional column-parallel slice-count (core count) override.
                Only valid when the column-parallel (single-row) factory is selected;
                must be in [2, 128] and at most the row's LLK-window chunk count
                (loud error otherwise); clamped only against the physical core grid,
                with a warning. Default: the built-in cost model's pick.
            tile_output: emit the output tensor(s) in TILE layout (default: False,
                ROW_MAJOR). Requires k to be a multiple of 32. Tile padding rows are
                zero-filled.
            index_dtype: dtype of the indices output, ttnn.uint32 or ttnn.uint16
                (default: None = ttnn.uint32). UINT16 requires the searched width
                (valid_length if set, else the last dimension) to be <= 65535.
        )doc",
        &topk_large_indices_py,
        nb::arg("input_tensor"),
        nb::kw_only(),
        nb::arg("k"),
        nb::arg("valid_length") = std::nullopt,
        nb::arg("return_values") = false,
        nb::arg("num_slices") = std::nullopt,
        nb::arg("tile_output") = false,
        nb::arg("index_dtype") = std::nullopt);
}

}  // namespace ttnn::operations::experimental::topk_large_indices::detail
