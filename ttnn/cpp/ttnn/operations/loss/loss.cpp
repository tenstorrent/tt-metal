// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <utility>
#include <vector>

#include "loss.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include "ttnn/operations/data_movement/copy/copy.hpp"

namespace ttnn::operations::loss::loss_utils {

using ttnn::operations::loss::LossFunction;
using ttnn::operations::loss::LossReductionMode;
using ttnn::operations::unary::EltwiseUnaryWithParam;
using ttnn::operations::unary::UnaryOpType;

Tensor loss_function(
    const Tensor& ref,
    const Tensor& prediction,
    const LossFunction loss_kind,
    const LossReductionMode reduce_mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    std::vector<EltwiseUnaryWithParam> fused_ops;
    switch (loss_kind) {
        case LossFunction::MAE: fused_ops.push_back(EltwiseUnaryWithParam{UnaryOpType::ABS}); break;
        case LossFunction::MSE: fused_ops.push_back(EltwiseUnaryWithParam{UnaryOpType::SQUARE}); break;
        default: TT_THROW("unsupported loss function {}. Please change.", loss_kind);
    }

    if (reduce_mode == LossReductionMode::NONE) {
        return ttnn::subtract(ref, prediction, std::nullopt, memory_config, optional_output_tensor, fused_ops);
    }

    Tensor diff = ttnn::subtract(ref, prediction, std::nullopt, memory_config, std::nullopt, fused_ops);
    Tensor reduced;

    switch (reduce_mode) {
        case LossReductionMode::SUM:
            reduced = ttnn::sum(
                diff, /*dim_arg=*/std::nullopt, /*keepdim=*/false, memory_config.value_or(ref.memory_config()));
            break;
        case LossReductionMode::MEAN:
            reduced = ttnn::mean(
                diff, /*dim_arg=*/std::nullopt, /*keepdim=*/false, memory_config.value_or(ref.memory_config()));
            break;
        default: TT_THROW("unsupported loss reduction mode {}.", reduce_mode);
    }

    if (optional_output_tensor.has_value()) {
        ttnn::copy(reduced, optional_output_tensor.value());
        return optional_output_tensor.value();
    }
    return reduced;
}

}  // namespace ttnn::operations::loss::loss_utils

namespace ttnn {

Tensor mse_loss(
    const Tensor& ref,
    const Tensor& prediction,
    operations::loss::LossReductionMode mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::loss::loss_utils::loss_function(
        ref, prediction, operations::loss::LossFunction::MSE, mode, memory_config, optional_output_tensor);
}

Tensor l1_loss(
    const Tensor& ref,
    const Tensor& prediction,
    operations::loss::LossReductionMode mode,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return operations::loss::loss_utils::loss_function(
        ref, prediction, operations::loss::LossFunction::MAE, mode, memory_config, optional_output_tensor);
}

}  // namespace ttnn
