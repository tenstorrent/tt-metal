// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "device/dit_minimal_binary_device_operation_types.hpp"

#include <optional>
#include <string>

namespace ttnn {
namespace operations::experimental {

struct DitMinimalRmBinaryOperation {
    static ttnn::Tensor invoke(
        const Tensor& input_a,
        const Tensor& input_b,
        const std::string& op = "add",
        const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::experimental

constexpr auto dit_minimal_binary = ttnn::register_operation<
    "ttnn::experimental::dit_minimal_binary",
    ttnn::operations::experimental::DitMinimalRmBinaryOperation>();

}  // namespace ttnn
