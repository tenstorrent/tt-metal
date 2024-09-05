// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/operations/moreh/moreh_cumsum/device/moreh_cumsum_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_cumsum {
struct MorehCumsum {
    static Tensor invoke(const Tensor& input, const Tensor& output, const int64_t dim);
};

struct MorehCumsumBackward {
    static Tensor invoke(const Tensor& input, const Tensor& output, const int64_t dim);
};
}  // namespace ttnn::operations::moreh::moreh_cumsum

namespace ttnn {
constexpr auto moreh_cumsum =
    ttnn::register_operation<"ttnn::moreh_cumsum", ttnn::operations::moreh::moreh_cumsum::MorehCumsum>();

constexpr auto moreh_cumsum_backward =
    ttnn::register_operation<"ttnn::moreh_cumsum_backward", ttnn::operations::moreh::moreh_cumsum::MorehCumsumBackward>();
}
