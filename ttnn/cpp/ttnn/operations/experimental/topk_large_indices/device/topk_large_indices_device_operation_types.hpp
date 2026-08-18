// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <limits>
#include <optional>
#include <vector>

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
    // Also emit the top-k VALUES (ROW_MAJOR BFLOAT16, sorted descending to match the indices;
    // exact bf16 -inf on the sentinel-index lanes). Changes kernel selection, CBs, and output
    // specs, so it is part of the program hash. Default off: indices-only, byte-identical
    // program to before this option existed.
    bool return_values{false};
    // Override the column-parallel slice count P (number of local cores splitting the row).
    // Only meaningful when the column-parallel (single-row) factory is selected: setting it on a
    // row-parallel shape is a loud error, as are values outside [2, 64] or above the row's chunk
    // count; it is clamped only against the physical core grid (with a warning). Changes the
    // program structure, so it is part of the program hash. nullopt = the built-in cost model.
    std::optional<uint32_t> num_slices{};
    // Emit the outputs (indices, and values when return_values) in TILE layout instead of
    // ROW_MAJOR. The writer scatters the 16-element result slices straight into their tile
    // positions and zero-fills the tile padding rows, so callers that need TILE tensors skip
    // the tilize ops entirely. Requires k % 32 == 0 (no partial output tile columns). Swaps in
    // dedicated writer kernels and output CBs, so it is part of the program hash. Default off:
    // ROW_MAJOR outputs, byte-identical program to before this option existed.
    bool tile_output{false};
    // Output dtype of the indices tensor. nullopt/UINT32 = today's UINT32 output. UINT16 is an
    // opt-in narrowing for callers that know every winner index fits 16 bits: it requires the
    // searched width (valid_length if set, else the last dimension) to be <= 65535 so real
    // indices are < 0xFFFF and the -inf sentinel truncates to the unambiguous 0xFFFF (the same
    // value a UINT32 -> UINT16 typecast of the sentinel produces). Changes the writer kernel and
    // output spec, so it is part of the program hash.
    std::optional<tt::tt_metal::DataType> index_dtype{};
};

struct tensor_args_t {
    Tensor input_tensor;
};

// [0] = UINT32 indices (always). [1] = BFLOAT16 values (only when return_values).
using tensor_return_value_t = std::vector<Tensor>;
using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;

}  // namespace ttnn::operations::experimental::topk_large_indices
