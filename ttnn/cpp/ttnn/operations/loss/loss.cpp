// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>

#include "loss.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn {

namespace operations::loss {

namespace loss_utils {

using ttnn::operations::loss::LossFunction;
using ttnn::operations::loss::LossReductionMode;
using ttnn::operations::unary::UnaryOpType;
using ttnn::operations::unary::UnaryWithParam;

Tensor loss_function(uint8_t queue_id,
                     const Tensor& ref,
                     const Tensor& prediction,
                     const LossFunction loss_kind,
                     const LossReductionMode reduce_mode,
                     const std::optional<MemoryConfig>& memory_config,
                     std::optional<Tensor> optional_output_tensor) {
    std::vector<UnaryWithParam> fused_ops;
    switch (loss_kind) {
        case LossFunction::MAE: fused_ops.push_back(UnaryWithParam{UnaryOpType::ABS}); break;
        case LossFunction::MSE: fused_ops.push_back(UnaryWithParam{UnaryOpType::SQUARE}); break;
        default: TT_THROW("unsupported loss function {}. Please change.", loss_kind);
    }
    Tensor result =
        ttnn::subtract(queue_id, ref, prediction, std::nullopt, memory_config, optional_output_tensor, fused_ops);

    switch (reduce_mode) {
        case LossReductionMode::SUM:
            return ttnn::sum(result, std::nullopt, true, memory_config.value_or(ref.memory_config()));
        case LossReductionMode::MEAN:
            return ttnn::mean(result, std::nullopt, true, memory_config.value_or(ref.memory_config()));
        case LossReductionMode::NONE:
        default:
            // TODO: old code indicated this path is unsupported, but the all post commit test pipeline uses this path.
            // Need to update test or replace this comment with a throw.
            break;
    }
    return result;
}

}  // namespace loss_utils

Tensor MseLossOperation::invoke(uint8_t queue_id,
                                const Tensor& ref,
                                const Tensor& prediction,
                                const LossReductionMode mode,
                                const std::optional<MemoryConfig>& memory_config,
                                std::optional<Tensor> optional_output_tensor) {
    return loss_utils::loss_function(
        queue_id, ref, prediction, LossFunction::MSE, mode, memory_config, optional_output_tensor);
}

Tensor MaeLossOperation::invoke(uint8_t queue_id,
                                const Tensor& ref,
                                const Tensor& prediction,
                                const LossReductionMode mode,
                                const std::optional<MemoryConfig>& memory_config,
                                std::optional<Tensor> optional_output_tensor) {
    return loss_utils::loss_function(
        queue_id, ref, prediction, LossFunction::MAE, mode, memory_config, optional_output_tensor);
}

}  // namespace operations::loss

}  // namespace ttnn
