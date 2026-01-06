// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_and_interleave_eltwise_mul.hpp"

#include "device/repeat_and_interleave_eltwise_mul_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ssm {

ttnn::Tensor ExecuteRepeatAndInterleaveEltwiseMul::invoke(
    const Tensor& a,
    const Tensor& b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype,
    const std::optional<MathFidelity> math_fidelity) {
    return ttnn::prim::repeat_and_interleave_eltwise_mul(a, b, memory_config, dtype, math_fidelity);
}

}  // namespace ttnn::operations::experimental::ssm
