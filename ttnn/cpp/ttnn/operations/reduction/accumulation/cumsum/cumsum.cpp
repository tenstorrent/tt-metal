// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "../accumulation_common.hpp"

#include "../device/accumulation_device_operation_types.hpp"
#include "../device/accumulation_device_operation.hpp"

#include "cumsum.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt_stl/small_vector.hpp>
#include "ttnn/common/queue_id.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::reduction::accumulation {

Tensor CumsumOperation::invoke(
    QueueId queue_id,
    const Tensor& input_tensor,
    const int32_t& dim,
    std::optional<ttnn::DataType> dtype,
    const bool& reverse_order,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config) {
    return common::accumulation_invoke(
        queue_id, input_tensor, dim, dtype, optional_out, reverse_order, memory_config, AccumulationOp::CUMSUM);
}

}  // namespace ttnn::operations::reduction::accumulation
