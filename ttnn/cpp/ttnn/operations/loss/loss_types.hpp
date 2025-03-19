// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::operations::loss {

// /**
//  * MSE: https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss
//  * MAE: https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html#torch.nn.L1Loss
//  * CEL:
//  https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy
//  * NLL: https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss
// */

enum class LossFunction {
    MSE,  // Mean Squared Error - squared L2 norm
    MAE,  // Mean Absolute Error - L1 norm
    CEL,  // Cross Entropy Loss
    NLL,  // Negative Log Likelihood -
};

enum class LossReductionMode {
    NONE,
    MEAN,
    SUM,
};

}  // namespace ttnn::operations::loss
