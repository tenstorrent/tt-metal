// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <optional>

#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::topk_large_indices {

inline uint32_t flattened_rows_excluding_last_dim(const ttnn::Shape& shape) {
    uint64_t rows = 1;
    for (uint32_t i = 0; i + 1 < shape.rank(); ++i) {
        const auto dim = shape[i];
        TT_FATAL(
            dim == 0 || rows <= std::numeric_limits<uint32_t>::max() / dim,
            "topk_large_indices flattened leading dimensions must fit in uint32_t rows; got shape {}",
            shape);
        rows *= dim;
    }
    return static_cast<uint32_t>(rows);
}

struct operation_attributes_t {
    uint32_t k{};
    // Restrict the search to the first `valid_length` columns of each row instead of the full last
    // dimension. Lets top-k run over the real prefix of an over-allocated row (whose tail may be stale)
    // without physically slicing the input. nullopt = search the full width. Runtime-only (hash-excluded,
    // validated on cache hit) so a serving loop growing valid_length reuses one program.
    std::optional<uint32_t> valid_length{};
    // INTERNAL (not exposed through the public API): override the column-parallel slice count P
    // (tree cores splitting each row). Single row: the classic column-parallel tree. Multiple
    // rows: the multi-rectangle variant — one P-core tree per rectangle, rows split contiguously
    // over as many rectangles as tile the worker grid, all concurrent (the cost model can also
    // auto-select this form when it models a win; the override bypasses the model and pins an
    // exact P — the hybrid wrapper's remainder window sets it to a searched-width-modeled P).
    // Values outside [2, 128] or above the row's chunk count are loud errors; P is clamped only
    // against the physical core grid (with a warning). Changes the program structure, so it is
    // part of the program hash. nullopt = the built-in cost model.
    std::optional<uint32_t> num_slices{};
    // Composite-internal row window [row_start, row_start + row_count): the op reads only these
    // input rows and emits a row_count-row output. Set by the hybrid wrapper (ttnn::experimental::
    // topk_large_indices) to run row-parallel full waves and a multi-rectangle remainder wave as
    // two launches over one un-sliced input; not exposed through the public bindings. Requires the
    // canonical [1.., R, W] shape (leading dims 1). Runtime-only for the program itself (rows are
    // runtime args); the effective row count feeds the derived split config, whose hashed fields
    // already capture any structural difference.
    std::optional<uint32_t> row_start{};
    std::optional<uint32_t> row_count{};
};

struct tensor_args_t {
    Tensor input_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttnn::operations::experimental::topk_large_indices
