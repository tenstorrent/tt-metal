// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

/**
 * MSE: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
 * MAE: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
 * CEL: https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
 * NLL: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
*/

enum LossFunction {
    MSE, //Mean Squared Error - squared L2 norm
    MAE, //Mean Absolute Error - L1 norm
    CEL, //Cross Entropy Loss
    NLL, //Negative Log Likelihood -
};

enum LossReductionMode {
    NONE, //no reduction
    MEAN,
    SUM
};

Tensor lossfunction(
    const Tensor& ref,
    const Tensor& prediction,
    const LossFunction loss_kind,
    const LossReductionMode reduce_mode,
    const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor mseloss(
    const Tensor& ref,
    const Tensor& prediction,
    const LossReductionMode mode,
    const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);

Tensor maeloss(
    const Tensor& ref,
    const Tensor& prediction,
    const LossReductionMode mode,
    const MemoryConfig& mem_config = operation::DEFAULT_OUTPUT_MEMORY_CONFIG);


}  // namespace tt_metal
}  // namespace tt
