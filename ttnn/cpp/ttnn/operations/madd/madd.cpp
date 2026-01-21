// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "madd.hpp"
#include <algorithm>
#include "tt-metalium/shape.hpp"
#include "ttnn/operations/madd/device/madd_device_operation.hpp"
#include "tt-metalium/buffer_types.hpp"
#include "tt-metalium/work_split.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/tensor/types.hpp"

namespace ttnn::operations::madd {

ttnn::Tensor MAdd::invoke(
    const ttnn::Tensor& a,
    const ttnn::Tensor& b,
    const ttnn::Tensor& c,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    tt::tt_metal::MemoryConfig mem_config = output_mem_config.value_or(a.memory_config());

    ttnn::DeviceComputeKernelConfig config = compute_kernel_config.value_or(
        ttnn::init_device_compute_kernel_config(a.device()->arch(), std::nullopt, MathFidelity::HiFi4));

    ttnn::Tensor output_tensor = ttnn::prim::madd(a, b, c, mem_config, config);
    return output_tensor;
}
}  // namespace ttnn::operations::madd
