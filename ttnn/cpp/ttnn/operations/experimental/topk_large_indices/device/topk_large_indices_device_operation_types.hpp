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
    // Trace-safe form of `valid_length`: when tensor_args.valid_length_tensor is set the kernel computes
    // valid_length = tensor[0] + valid_length_offset on-device. The offset exists because the indexer's
    // bound is chunk_start + chunk_global, and chunk_global is structural (hashed) while only chunk_start
    // varies per chunk -- so one metadata tensor drives BOTH this op and ring_indexer_score_dsa's kv_len,
    // and the two provably cannot disagree. Hashed: it shapes nothing but must not be silently ignored.
    uint32_t valid_length_offset{0};
};

struct tensor_args_t {
    Tensor input_tensor;
    // 1-element uint32 ROW_MAJOR DRAM tensor holding the per-chunk base of valid_length. `valid_length` is
    // a host runtime arg that a ttnn trace replay freezes at its capture-time value; reading it on-device
    // is what makes top-k replayable across chunks.
    std::optional<Tensor> valid_length_tensor{std::nullopt};
    bool has_valid_length_metadata() const { return valid_length_tensor.has_value(); }
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

}  // namespace ttnn::operations::experimental::topk_large_indices
