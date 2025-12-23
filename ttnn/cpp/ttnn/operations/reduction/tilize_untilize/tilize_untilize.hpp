// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <tuple>
#include "ttnn/decorators.hpp"
#include "device/tilize_untilize_device_operation.hpp"

namespace ttnn {
namespace operations {
namespace reduction {
namespace tilize_untilize {

struct ExecuteTilizeUntilize {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input,
        std::optional<MemoryConfig> output_memory_config = std::nullopt,
        std::optional<DataType> output_dtype = std::nullopt,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace tilize_untilize
}  // namespace reduction
}  // namespace operations

// Register the operation
constexpr auto tilize_untilize = ttnn::
    register_operation<"ttnn::tilize_untilize", ttnn::operations::reduction::tilize_untilize::ExecuteTilizeUntilize>();

}  // namespace ttnn
