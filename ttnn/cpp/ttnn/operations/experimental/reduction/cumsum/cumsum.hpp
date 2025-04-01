// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/common/queue_id.hpp"
#include "ttnn/operations/functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::reduction {

struct CumSumOperation {
    static Tensor invoke(
        QueueId queue_id,
        const Tensor& input,
        int64_t dim,
        std::optional<ttnn::DataType> dtype = std::nullopt,
        std::optional<Tensor> preallocated_output = std::nullopt);
};

}  // namespace ttnn::operations::experimental::reduction
