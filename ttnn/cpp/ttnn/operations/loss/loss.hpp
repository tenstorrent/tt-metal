// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "loss_types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn {

namespace operations::loss {

struct MseLossOperation {
    static Tensor invoke(
        const Tensor& ref,
        const Tensor& prediction,
        LossReductionMode mode = LossReductionMode::NONE,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

struct MaeLossOperation {
    static Tensor invoke(
        const Tensor& ref,
        const Tensor& prediction,
        LossReductionMode mode = LossReductionMode::NONE,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);
};

}  // namespace operations::loss

constexpr auto mse_loss = ttnn::register_operation<"ttnn::mse_loss", operations::loss::MseLossOperation>();

constexpr auto l1_loss = ttnn::register_operation<"ttnn::l1_loss", operations::loss::MaeLossOperation>();

}  // namespace ttnn
