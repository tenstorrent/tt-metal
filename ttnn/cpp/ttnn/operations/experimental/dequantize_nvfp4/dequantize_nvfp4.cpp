// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dequantize_nvfp4.hpp"
#include "device/dequantize_nvfp4_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental {
Tensor dequantize_nvfp4(
    const Tensor& packed,
    const Tensor& scales,
    float global_scale,
    Nvfp4ScaleFormat scale_format,
    DataType dtype,
    Layout output_layout,
    bool fp32_dest_acc_en,
    const std::optional<MemoryConfig>& memory_config,
    bool experimental_fast,
    bool fpu_tensor_scale,
    bool transpose_output) {
    return ttnn::device_operation::launch<ttnn::experimental::prim::Nvfp4DeviceOperation>(
        ttnn::experimental::prim::Nvfp4Params{
            .global_scale = global_scale,
            .scale_format = scale_format,
            .dtype = dtype,
            .output_layout = output_layout,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .memory_config = memory_config.value_or(packed.memory_config()),
            .experimental_fast = experimental_fast,
            .fpu_tensor_scale = fpu_tensor_scale,
            .transpose_output = transpose_output},
        ttnn::experimental::prim::Nvfp4Inputs{.packed = packed, .scales = scales});
}
}  // namespace ttnn::operations::experimental
