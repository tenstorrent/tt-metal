// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::reduction {

struct ArgMaxOperation {
    static ttnn::Tensor invoke(
        uint8_t queue_id,
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const bool use_muticore = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static ttnn::Tensor invoke(
        const Tensor& input_tensor,
        const std::optional<int> dim = std::nullopt,
        const bool use_muticore = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

};

}  // namespace operations::reduction

constexpr auto argmax =
    ttnn::register_operation_with_auto_launch_op<"ttnn::argmax", ttnn::operations::reduction::ArgMaxOperation>();

}  // namespace ttnn
