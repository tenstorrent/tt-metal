// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn {
namespace operations::experimental {

using namespace tt;

struct UniqueOperation {
    static Tensor invoke(
        const Tensor& input,
        const bool& sorted,
        const bool& return_inverse,
        const bool& return_counts,
        const std::optional<int32_t>& dim,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

namespace experimental {
constexpr auto unique =
    ttnn::register_operation<"ttnn::experimental::unique", ttnn::operations::experimental::UniqueOperation>();
}  // namespace experimental

}  // namespace ttnn
