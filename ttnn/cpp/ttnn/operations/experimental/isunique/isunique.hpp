// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn {
namespace operations::experimental {

using namespace tt;

struct IsUniqueOperation {
    static Tensor invoke(
        const QueueId& queue_id,
        const Tensor& input_tensor,
        const std::optional<int>& dim = std::nullopt,
        const bool& invert = false,
        const bool& first_occurrences = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& opt_out = std::nullopt);
};

}  // namespace operations::experimental

namespace experimental {
constexpr auto isunique =
    ttnn::register_operation<"ttnn::experimental::isunique", ttnn::operations::experimental::IsUniqueOperation>();
}  // namespace experimental

}  // namespace ttnn
