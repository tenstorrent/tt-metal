// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::common {

// Program-cache keys for the row-free ops. The row count (shape[-2]: the token rows a dynamic-length call changes) is
// the one shape entry those ops' kernels take as a runtime argument, so it stays out of the hash; everything else
// (the other dims, padding, alignment, dtype, layout, memory config) stays in. With the rows in the key every served
// sequence length would compile and cache a program of its own. Ops never need to tell their keys apart here: the
// framework folds the op's type hash into every custom program hash.

namespace detail {
// Folds every entry of `dims` except the one at `rows_dim`, counted from the end (-2 is the row dim of a
// row-major-tiles tensor, -1 the sequence dim of a transposed [.., D, S] one); a rank too small to have that dim
// skips nothing. The size goes in so [a,_,c] and [a,c] stay distinct.
inline ttsl::hash::hash_t hash_dims_without_rows(
    ttsl::hash::hash_t seed, ttsl::Span<const uint32_t> dims, int rows_dim) {
    TT_FATAL(rows_dim < 0, "rows_dim counts from the end of the shape (-2 for the rows), got {}", rows_dim);
    seed = ttsl::hash::hash_objects(seed, dims.size());
    const auto skip = static_cast<std::ptrdiff_t>(dims.size()) + rows_dim;  // negative: nothing to skip
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (static_cast<std::ptrdiff_t>(i) == skip) {
            continue;
        }
        seed = ttsl::hash::hash_objects(seed, dims[i]);
    }
    return seed;
}
}  // namespace detail

// Hashes the spec except shape[rows_dim] (default [-2]).
inline ttsl::hash::hash_t hash_spec_without_rows(const tt::tt_metal::TensorSpec& spec, int rows_dim = -2) {
    const auto& layout = spec.tensor_layout();
    auto h = ttsl::hash::hash_objects_with_default_seed(
        layout.get_data_type(), layout.get_page_config(), layout.get_memory_config());
    h = detail::hash_dims_without_rows(h, spec.logical_shape().view(), rows_dim);
    h = detail::hash_dims_without_rows(h, spec.padded_shape().view(), rows_dim);
    return detail::hash_dims_without_rows(h, layout.get_alignment().view(), rows_dim);
}

// The optional overloads hash the presence bit, then wrap the value ones rather than replace them: binding a
// TensorSpec to a `const std::optional<TensorSpec>&` would copy it into a temporary on every key computation.
inline ttsl::hash::hash_t hash_spec_without_rows(
    const std::optional<tt::tt_metal::TensorSpec>& spec, int rows_dim = -2) {
    return spec.has_value() ? ttsl::hash::hash_objects_with_default_seed(true, hash_spec_without_rows(*spec, rows_dim))
                            : ttsl::hash::hash_objects_with_default_seed(false);
}

inline ttsl::hash::hash_t hash_spec_without_rows(const Tensor& tensor, int rows_dim = -2) {
    return hash_spec_without_rows(tensor.tensor_spec(), rows_dim);
}

inline ttsl::hash::hash_t hash_spec_without_rows(const std::optional<Tensor>& tensor, int rows_dim = -2) {
    return tensor.has_value() ? ttsl::hash::hash_objects_with_default_seed(
                                    true, hash_spec_without_rows(tensor->tensor_spec(), rows_dim))
                              : ttsl::hash::hash_objects_with_default_seed(false);
}

// Rows of a tensor with its upper dims folded in: the count the row-free kernels iterate. The spec's cached physical
// height: tile-padded for a TILE layout, the logical row count for a ROW_MAJOR one.
inline uint32_t folded_rows(const tt::tt_metal::TensorSpec& spec) {
    return static_cast<uint32_t>(spec.physical_shape().height());
}

inline uint32_t folded_rows(const Tensor& tensor) { return folded_rows(tensor.tensor_spec()); }

}  // namespace ttnn::operations::common
