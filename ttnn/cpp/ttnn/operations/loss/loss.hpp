// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "loss_types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {
Tensor mse_loss(
    const Tensor& ref,
    const Tensor& prediction,
    operations::loss::LossReductionMode mode = operations::loss::LossReductionMode::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

Tensor l1_loss(
    const Tensor& ref,
    const Tensor& prediction,
    operations::loss::LossReductionMode mode = operations::loss::LossReductionMode::NONE,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& optional_output_tensor = std::nullopt);

}  // namespace ttnn
