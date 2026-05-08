// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "device/dropout_new_device_operation.hpp"
#include "dropout_new.hpp"

namespace ttnn::experimental {

Tensor dropout_new(
    const Tensor& input_tensor,
    float prob,
    float scale,
    uint32_t seed,
    bool use_per_device_seed,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::dropout_new(
        input_tensor,
        prob,
        scale,
        seed,
        use_per_device_seed,
        DataType::BFLOAT16,
        memory_config,
        optional_output_tensor);
}

}  // namespace ttnn::experimental
