// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

#include "loss_types.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn {

namespace operations::loss {

struct MseLossOperation {

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& ref,
        const Tensor& prediction,
        const LossReductionMode mode = LossReductionMode::NONE,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& ref,
        const Tensor& prediction,
        const LossReductionMode mode = LossReductionMode::NONE,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
            return MseLossOperation::invoke (DefaultQueueId, ref, prediction, mode, memory_config, optional_output_tensor);
    }
};

struct MaeLossOperation {

    static Tensor invoke(
        uint8_t queue_id,
        const Tensor& ref,
        const Tensor& prediction,
        const LossReductionMode mode = LossReductionMode::NONE,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt);

    static Tensor invoke(
        const Tensor& ref,
        const Tensor& prediction,
        const LossReductionMode mode = LossReductionMode::NONE,
        const std::optional<MemoryConfig>& memory_config = std::nullopt,
        std::optional<Tensor> optional_output_tensor = std::nullopt) {
            return MaeLossOperation::invoke (DefaultQueueId, ref, prediction, mode, memory_config, optional_output_tensor);
    }
};

}  // namespace operations::loss


constexpr auto mse_loss = ttnn::register_operation_with_auto_launch_op<"ttnn::mse_loss", operations::loss::MseLossOperation>();

constexpr auto l1_loss = ttnn::register_operation_with_auto_launch_op<"ttnn::l1_loss", operations::loss::MaeLossOperation>();


}  // namespace ttnn
