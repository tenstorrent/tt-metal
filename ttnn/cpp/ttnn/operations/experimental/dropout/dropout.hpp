
// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::experimental {

struct DropoutOperation {
    static Tensor invoke(
        const Tensor& input_tensor,
        float prob,
        float scale,
        uint32_t seed,
        bool use_per_device_seed = true,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        const std::optional<Tensor>& optional_output_tensor = std::nullopt);
};
}  // namespace ttnn::operations::experimental
namespace ttnn::experimental {
constexpr auto dropout =
    ttnn::register_operation<"ttnn::experimental::dropout", ttnn::operations::experimental::DropoutOperation>();
}  // namespace ttnn::experimental
