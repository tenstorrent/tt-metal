// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "uniform.hpp"

#include <cstdint>

#include "device/uniform_device_operation.hpp"
#include "uniform_range.hpp"

namespace ttnn {

Tensor uniform(
    const Tensor& input,
    const float from,
    const float to,
    const std::uint32_t seed,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Uniform: Input tensor must be Float32 or Bfloat16");
    const auto output_range = operations::uniform::make_inclusive_output_range(from, to, input.dtype());
    return ttnn::prim::uniform(
        input, output_range.lower_bound, output_range.upper_bound, seed, memory_config, compute_kernel_config);
}

}  // namespace ttnn
