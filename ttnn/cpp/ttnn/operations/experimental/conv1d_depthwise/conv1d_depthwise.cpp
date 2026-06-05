// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv1d_depthwise.hpp"

#include "device/conv1d_depthwise_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental {

ttnn::Tensor conv1d_depthwise(
    const ttnn::Tensor& input_tensor,
    const std::vector<float>& taps,
    uint32_t stride,
    const std::optional<DataType>& dtype,
    const std::optional<DeviceComputeKernelConfig>& compute_config,
    const std::optional<MemoryConfig>& memory_config) {
    const DataType out_dtype = dtype.value_or(input_tensor.dtype());
    const MemoryConfig out_mem = memory_config.value_or(input_tensor.memory_config());
    const DeviceComputeKernelConfig cc = compute_config.value_or(init_device_compute_kernel_config(
        input_tensor.device()->arch(),
        /*device_kernel_config=*/std::nullopt,
        /*default_fidelity=*/MathFidelity::HiFi4,
        /*default_approx_mode=*/false,
        /*default_fp32_acc=*/true,
        /*default_l1_acc=*/true));

    return ttnn::prim::conv1d_depthwise(input_tensor, taps, stride, out_dtype, cc, out_mem);
}

}  // namespace ttnn::experimental
