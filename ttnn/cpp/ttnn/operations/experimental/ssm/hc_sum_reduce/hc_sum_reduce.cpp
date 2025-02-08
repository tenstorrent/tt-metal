// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce.hpp"

#include "device/hc_sum_reduce_op.hpp"
#include "ttnn/common/queue_id.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ssm {

ttnn::Tensor ExecuteHCSumReduce::invoke(
    QueueId queue_id,
    const Tensor& input,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype,
    const std::optional<MathFidelity> math_fidelity) {
    auto program = HCSumReduce{
        memory_config.value_or(input.memory_config()),
        dtype.value_or(input.dtype()),
        math_fidelity.value_or(MathFidelity::HiFi4)};
    return operation::run(program, {input}, {}, {}, queue_id).at(0);
}

}  // namespace ttnn::operations::experimental::ssm
