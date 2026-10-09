// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/unsqueeze/unsqueeze.hpp"

#include <tt_stl/small_vector.hpp>

#include "ttnn/operations/experimental/quasar/reshape_view/reshape.hpp"

namespace ttnn::operations::experimental::quasar {

ttnn::Tensor unsqueeze(const ttnn::Tensor& input_tensor, const int dim) {
    const auto& tensor_shape = input_tensor.logical_shape();
    const uint32_t rank = tensor_shape.rank();
    const int32_t max_dim = static_cast<int32_t>(rank);
    const int32_t min_dim = -max_dim - 1;

    TT_FATAL(
        (dim >= min_dim) && (dim <= max_dim),
        "Dimension out of range (expected to be in range of [{},{}], but got {})",
        min_dim,
        max_dim,
        dim);
    const int normal_dim = dim < 0 ? static_cast<int>(rank) + 1 + dim : dim;

    ttsl::SmallVector<uint32_t> output_shape_vector;
    for (int i = 0; i < static_cast<int>(rank); ++i) {
        if (i == normal_dim) {
            output_shape_vector.push_back(1);
        }
        output_shape_vector.push_back(tensor_shape[i]);
    }
    if (normal_dim == static_cast<int>(rank)) {
        output_shape_vector.push_back(1);
    }

    return ttnn::operations::experimental::quasar::reshape(input_tensor, ttnn::Shape(std::move(output_shape_vector)));
}

}  // namespace ttnn::operations::experimental::quasar
