// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>

#include <tt_stl/reflection.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::common {

// Program-cache keys for the row-free ops. The row count (shape[-2]: the token rows a dynamic-length call changes) is
// the one shape entry those ops' kernels take as a runtime argument, so it stays out of the hash; everything else
// (the other dims, padding, alignment, dtype, layout, memory config) stays in. With the rows in the key every served
// sequence length would compile and cache a program of its own.

namespace detail {
// Folds every entry of `dims` except the one at `rows_dim` (counted from the end, so -2 is the row dim of a
// row-major-tiles tensor and -1 the sequence dim of a transposed [.., D, S] one); the size goes in so
// [a,_,c] and [a,c] stay distinct.
inline ttsl::hash::hash_t hash_dims_without_rows(
    ttsl::hash::hash_t seed, ttsl::Span<const uint32_t> dims, int rows_dim = -2) {
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

// Hashes the spec except shape[rows_dim] (default [-2]); `salt` separates ops that share fields, since a
// custom hash drops the framework's collision guard.
inline ttsl::hash::hash_t hash_spec_without_rows(
    const tt::tt_metal::TensorSpec& spec, uint64_t salt = 0, int rows_dim = -2) {
    const auto& layout = spec.tensor_layout();
    auto h = ttsl::hash::hash_objects_with_default_seed(
        salt, layout.get_data_type(), layout.get_page_config(), layout.get_memory_config());
    h = detail::hash_dims_without_rows(h, spec.logical_shape().view(), rows_dim);
    h = detail::hash_dims_without_rows(h, spec.padded_shape().view(), rows_dim);
    return detail::hash_dims_without_rows(h, layout.get_alignment().view(), rows_dim);
}

// The optional overloads wrap the value ones rather than replace them: binding a TensorSpec to a
// `const std::optional<TensorSpec>&` would copy it into a temporary on every key computation.
inline ttsl::hash::hash_t hash_spec_without_rows(
    const std::optional<tt::tt_metal::TensorSpec>& spec, uint64_t salt = 0, int rows_dim = -2) {
    return spec.has_value() ? hash_spec_without_rows(*spec, salt, rows_dim)
                            : ttsl::hash::hash_objects_with_default_seed(salt);
}

inline ttsl::hash::hash_t hash_spec_without_rows(const Tensor& tensor, uint64_t salt = 0, int rows_dim = -2) {
    return hash_spec_without_rows(tensor.tensor_spec(), salt, rows_dim);
}

inline ttsl::hash::hash_t hash_spec_without_rows(
    const std::optional<Tensor>& tensor, uint64_t salt = 0, int rows_dim = -2) {
    return tensor.has_value() ? hash_spec_without_rows(tensor->tensor_spec(), salt, rows_dim)
                              : ttsl::hash::hash_objects_with_default_seed(salt);
}

// Rows of a tensor with its upper dims folded in, tile-padded: the count the row-free kernels iterate.
inline uint32_t folded_rows(const Tensor& tensor) {
    return static_cast<uint32_t>(tensor.physical_volume() / tensor.padded_shape()[-1]);
}

}  // namespace ttnn::operations::common
