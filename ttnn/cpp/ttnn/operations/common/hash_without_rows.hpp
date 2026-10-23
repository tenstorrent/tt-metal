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

namespace detail {
// Folds every entry of `dims` except [-2]; the size goes in so [a,_,c] and [a,c] stay distinct.
inline ttsl::hash::hash_t hash_dims_without_rows(ttsl::hash::hash_t seed, ttsl::Span<const uint32_t> dims) {
    seed = ttsl::hash::hash_objects(seed, dims.size());
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (dims.size() >= 2 && i == dims.size() - 2) {
            continue;
        }
        seed = ttsl::hash::hash_objects(seed, dims[i]);
    }
    return seed;
}
}  // namespace detail

// Hashes the spec except shape[-2]; `salt` separates ops that share fields, since a custom hash drops
// the framework's collision guard.
inline ttsl::hash::hash_t hash_spec_without_rows(const tt::tt_metal::TensorSpec& spec, uint64_t salt = 0) {
    const auto& layout = spec.tensor_layout();
    auto h = ttsl::hash::hash_objects_with_default_seed(
        salt, layout.get_data_type(), layout.get_page_config(), layout.get_memory_config());
    h = detail::hash_dims_without_rows(h, spec.logical_shape().view());
    h = detail::hash_dims_without_rows(h, spec.padded_shape().view());
    return detail::hash_dims_without_rows(h, layout.get_alignment().view());
}

inline ttsl::hash::hash_t hash_spec_without_rows(
    const std::optional<tt::tt_metal::TensorSpec>& spec, uint64_t salt = 0) {
    return spec.has_value() ? hash_spec_without_rows(*spec, salt) : ttsl::hash::hash_objects_with_default_seed(salt);
}

inline ttsl::hash::hash_t hash_spec_without_rows(const Tensor& tensor, uint64_t salt = 0) {
    return hash_spec_without_rows(tensor.tensor_spec(), salt);
}

inline ttsl::hash::hash_t hash_spec_without_rows(const std::optional<Tensor>& tensor, uint64_t salt = 0) {
    return tensor.has_value() ? hash_spec_without_rows(tensor->tensor_spec(), salt)
                              : ttsl::hash::hash_objects_with_default_seed(salt);
}

// Rows of a tensor with its upper dims folded in, tile-padded: the count the row-free kernels iterate.
inline uint32_t folded_rows(const Tensor& tensor) {
    return static_cast<uint32_t>(tensor.physical_volume() / tensor.padded_shape()[-1]);
}

}  // namespace ttnn::operations::common
