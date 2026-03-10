// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/moe_dispatch_offsets_device_operation.hpp"
#include "ttnn/operations/experimental/moe_dispatch_offsets/moe_dispatch_offsets.hpp"

#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor MoeDispatchOffsetsOperation::invoke(const Tensor& input_tensor, uint32_t n_routed_experts) {
    return ttnn::prim::moe_dispatch_offsets(input_tensor, n_routed_experts);
}

}  // namespace ttnn::operations::experimental
