// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/reduction/cumsum/cumsum.hpp"
#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/experimental/reduction/cumsum/device/cumsum_device_operation.hpp"

namespace ttnn::operations::experimental::reduction {

Tensor CumSumOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const int64_t dim,
    std::optional<ttnn::DataType> dtype,
    std::optional<Tensor> optional_output_tensor) {
    return ttnn::prim::cumsum(queue_id, input_tensor, dim, dtype, optional_output_tensor);
}

}  // namespace ttnn::operations::experimental::reduction
