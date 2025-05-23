// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <vector>
#include <optional>
#include <cstdint>

namespace reduction_common {

template <class Tuple, class T = std::decay_t<std::tuple_element_t<0, std::decay_t<Tuple>>>>
std::vector<std::optional<T>> tuple_to_vector_optional(Tuple&& tuple) {
    return std::apply(
        [](auto&&... elems) { return std::vector<std::optional<T>>{std::forward<decltype(elems)>(elems)...}; },
        std::forward<Tuple>(tuple));
}

ttnn::Tensor perform_transpose(
    const ttnn::Tensor& input_tensor, const bool is_dim_last_idx, const int8_t dim1 = -1, const int8_t dim2 = -1);

ttnn::Tensor transform_to_4d_tensor(const ttnn::Tensor& input_tensor, const bool is_rank_le_4d);

}  // namespace reduction_common
