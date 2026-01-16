// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce.hpp"

#include "device/hc_sum_reduce_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ssm {

ttnn::Tensor ExecuteHCSumReduce::invoke(
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype,
    const std::optional<MathFidelity> math_fidelity) {
    return ttnn::prim::hc_sum_reduce(input, memory_config, dtype, math_fidelity);
}

}  // namespace ttnn::operations::experimental::ssm
