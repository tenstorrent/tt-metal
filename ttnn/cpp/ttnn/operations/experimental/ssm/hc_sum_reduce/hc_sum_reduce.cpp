// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce.hpp"

#include "device/hc_sum_reduce_op.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::experimental::ssm {

ttnn::Tensor ExecuteHCSumReduce::invoke(uint8_t queue_id,
                                        const Tensor& input,
                                        const std::optional<MemoryConfig>& memory_config,
                                        const std::optional<DataType> dtype,
                                        const std::optional<MathFidelity> math_fidelity) {
    auto program = HCSumReduce{memory_config.value_or(input.memory_config()),
                               dtype.value_or(input.dtype()),
                               math_fidelity.value_or(MathFidelity::HiFi4)};
    return operation::run(program, {input}, {}, {}, queue_id).at(0);
}

ttnn::Tensor ExecuteHCSumReduce::invoke(const Tensor& input,
                                        const std::optional<MemoryConfig>& memory_config,
                                        const std::optional<DataType> dtype,
                                        const std::optional<MathFidelity> math_fidelity) {
    return invoke(DefaultQueueId, input, memory_config, dtype, math_fidelity);
}

}  // namespace ttnn::operations::experimental::ssm
