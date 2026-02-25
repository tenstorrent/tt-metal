// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "../accumulation_common.hpp"

#include "../device/accumulation_device_operation_types.hpp"
#include "../device/accumulation_device_operation.hpp"

#include "cumsum.hpp"

#include <utility>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn {

Tensor cumsum(
    const Tensor& input,
    const int32_t& dim,
    std::optional<DataType> dtype,
    const bool& reverse_order,
    std::optional<Tensor> optional_out,
    const std::optional<MemoryConfig>& memory_config) {
    return operations::reduction::accumulation::common::accumulation_invoke(
        input, dim, dtype, std::move(optional_out), reverse_order, memory_config, ttnn::prim::AccumulationOp::CUMSUM);
}

}  // namespace ttnn
