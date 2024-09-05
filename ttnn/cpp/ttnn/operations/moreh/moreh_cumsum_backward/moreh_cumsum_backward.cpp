// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum_backward.hpp"
namespace ttnn::operations::moreh::moreh_cumsum_backward {
Tensor MorehCumsumBackward::invoke(const Tensor& input, const int64_t dim) {
    return ttnn::prim::moreh_cumsum(input, dim, true);
}
}  // namespace ttnn::operations::moreh::moreh_cumsum
