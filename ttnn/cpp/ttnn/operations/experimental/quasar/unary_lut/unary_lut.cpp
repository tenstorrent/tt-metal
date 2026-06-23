// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_lut.hpp"

#include "ttnn/operations/experimental/quasar/unary_lut/device/unary_lut_device_operation.hpp"

namespace ttnn::operations::experimental::quasar::unary_lut {

Tensor unary_lut(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& output,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
    const std::optional<LutConfig>& lut_config) {
    return ttnn::prim::qsr::unary_lut(input_tensor, memory_config, output, sub_device_id, lut_config);
}

}  // namespace ttnn::operations::experimental::quasar::unary_lut
