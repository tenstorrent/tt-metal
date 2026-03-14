// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "device/masked_bincount_device_operation.hpp"
#include "ttnn/operations/experimental/masked_bincount/masked_bincount.hpp"

#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor MaskedBincountOperation::invoke(const Tensor& input_tensor, uint32_t n_routed_experts) {
    return ttnn::prim::masked_bincount(input_tensor, n_routed_experts);
}

}  // namespace ttnn::operations::experimental
