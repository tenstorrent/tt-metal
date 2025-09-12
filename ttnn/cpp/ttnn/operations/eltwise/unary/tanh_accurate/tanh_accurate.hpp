// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations {

namespace unary {

struct Tanh_accurate {
    static Tensor invoke(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

struct Tanhshrink_accurate {
    static Tensor invoke(
        const Tensor& input,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};

}  // namespace unary
}  // namespace operations

constexpr auto tanh_accurate =
    ttnn::register_operation<"ttnn::tanh_accurate", ttnn::operations::unary::Tanh_accurate>();

constexpr auto tanhshrink_accurate =
    ttnn::register_operation<"ttnn::tanhshrink_accurate", ttnn::operations::unary::Tanhshrink_accurate>();

}  // namespace ttnn
