// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dequantize_mxfp8.hpp"
#include "device/dequantize_mxfp8_device_operation.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental {
Tensor dequantize_mxfp8(
    const Tensor& packed,
    const Tensor& scales,
    DataType dtype,
    Layout output_layout,
    bool fp32_dest_acc_en,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::device_operation::launch<ttnn::experimental::prim::Mxfp8DeviceOperation>(
        ttnn::experimental::prim::Mxfp8Params{
            .dtype = dtype,
            .output_layout = output_layout,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .memory_config = memory_config.value_or(packed.memory_config())},
        ttnn::experimental::prim::Mxfp8Inputs{.packed = packed, .scales = scales});
}
}  // namespace ttnn::operations::experimental
