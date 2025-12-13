// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "prefix_scan.hpp"

#include "device/prefix_scan_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::experimental::ssm {

ttnn::Tensor ExecutePrefixScan::invoke(
    const Tensor& a,
    const Tensor& bx,
    const Tensor& h_prev,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype,
    const std::optional<MathFidelity> math_fidelity) {
    return ttnn::prim::prefix_scan(a, bx, h_prev, memory_config, dtype, math_fidelity);
}

}  // namespace ttnn::operations::experimental::ssm
