// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan.hpp"

#include "device/prefix_scan_op.hpp"
#include "ttnn/common/constants.hpp"

namespace ttnn::operations::experimental::ssm {

ttnn::Tensor ExecutePrefixScan::invoke(
    uint8_t queue_id,
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h_prev,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype,
    const std::optional<MathFidelity> math_fidelity) {
    auto program = PrefixScan{
        memory_config.value_or(a.memory_config()),
        dtype.value_or(a.dtype()),
        math_fidelity.value_or(MathFidelity::HiFi4)};
    return operation::run(program, {a, bx, h_prev}, {}, {}, queue_id).at(0);
}

ttnn::Tensor ExecutePrefixScan::invoke(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h_prev,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype,
    const std::optional<MathFidelity> math_fidelity) {
    return invoke(DefaultQueueId, a, bx, h_prev, memory_config, dtype, math_fidelity);
}

}  // namespace ttnn::operations::experimental::ssm
