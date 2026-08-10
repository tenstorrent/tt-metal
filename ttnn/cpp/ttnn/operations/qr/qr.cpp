// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qr.hpp"

#include "device/qr_device_operation.hpp"

namespace ttnn {

std::tuple<Tensor, Tensor> qr(
    const Tensor& input, const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::qr(input, memory_config);
}

}  // namespace ttnn
