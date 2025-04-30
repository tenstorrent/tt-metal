// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "ttnn/decorators.hpp"

namespace ttnn::operations::index_fill {

struct IndexFill {
    static Tensor invoke(
        const Tensor& input,
        const uint32_t dim,
        const Tensor& index,
        const std::variant<float, int> value,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::index_fill

namespace ttnn {
constexpr auto index_fill = ttnn::register_operation<"ttnn::index_fill", ttnn::operations::index_fill::IndexFill>();
}  // namespace ttnn
