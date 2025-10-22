// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::reduction {

struct IntImgOperation {
    static void validate(const Tensor& input_tensor);

    static Tensor invoke(const Tensor& input_tensor);
};

}  // namespace operations::experimental::reduction

namespace experimental {
constexpr auto intimg = ttnn::
    register_operation<"ttnn::experimental::intimg", ttnn::operations::experimental::reduction::IntImgOperation>();
}

}  // namespace ttnn
