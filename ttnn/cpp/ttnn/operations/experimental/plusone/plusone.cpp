// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/plusone_op.hpp"
#include "ttnn/operations/experimental/plusone/plusone.hpp"

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn::operations::experimental {

ttnn::Tensor PlusOneOperation::invoke(uint8_t queue_id, const Tensor& input_tensor) {
    return operation::run(PlusOne{}, {input_tensor}, {}, {}, queue_id).at(0);
}

ttnn::Tensor PlusOneOperation::invoke(const Tensor& input_tensor) {
    return invoke(DefaultQueueId, input_tensor);
}

}  // namespace ttnn::operations::experimental
