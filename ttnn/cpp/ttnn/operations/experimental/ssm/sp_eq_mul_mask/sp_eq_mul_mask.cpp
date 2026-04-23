// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sp_eq_mul_mask.hpp"

#include "device/sp_eq_mul_mask_device_operation.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental {

ttnn::Tensor sp_eq_mul_mask(
    const Tensor& a,
    const Tensor& b,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DataType> dtype) {
    return ttnn::prim::sp_eq_mul_mask(a, b, memory_config, dtype);
}

}  // namespace ttnn::experimental
