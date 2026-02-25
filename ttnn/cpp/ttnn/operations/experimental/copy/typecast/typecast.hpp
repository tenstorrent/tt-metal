// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include <optional>

namespace ttnn {
namespace operations::experimental::copy {

struct TypecastOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const DataType& dtype,
        const std::optional<MemoryConfig>& output_mem_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};
}  // namespace operations::experimental::copy

namespace experimental {
constexpr auto typecast =
    ttnn::register_operation<"ttnn::experimental::typecast", ttnn::operations::experimental::copy::TypecastOperation>();
}  // namespace experimental
}  // namespace ttnn
