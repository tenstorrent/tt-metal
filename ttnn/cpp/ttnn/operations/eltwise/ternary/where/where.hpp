// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/decorators.hpp"
#include "ttnn/common/queue_id.hpp"

namespace ttnn {

namespace operations {

namespace ternary {

struct WhereOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& predicate,
        const std::variant<float, Tensor>& value_true,
        const std::variant<float, Tensor>& value_false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> output_tensor = std::nullopt);
};

}  // namespace ternary
}  // namespace operations

constexpr auto where = ttnn::register_operation<"ttnn::where", operations::ternary::WhereOperation>();

}  // namespace ttnn
