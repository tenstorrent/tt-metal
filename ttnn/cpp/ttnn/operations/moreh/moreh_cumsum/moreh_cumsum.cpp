// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_cumsum.hpp"
namespace ttnn::operations::moreh::moreh_cumsum {
Tensor MorehCumsum::invoke(const Tensor& input, const Tensor& output, const int64_t dim) {
    return ttnn::prim::moreh_cumsum(input, output, dim, false);
}

Tensor MorehCumsumBackward::invoke(const Tensor& input, const Tensor& output, const int64_t dim) {
    return ttnn::prim::moreh_cumsum(input, output, dim, true);
}
}  // namespace ttnn::operations::moreh::moreh_cumsum
