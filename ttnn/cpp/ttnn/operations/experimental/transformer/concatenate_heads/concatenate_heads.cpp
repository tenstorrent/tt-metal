// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads.hpp"
#include "device/concatenate_heads_device_operation.hpp"

namespace ttnn::experimental {

ttnn::Tensor concatenate_heads(
    const Tensor& input_tensor,
    const CoreCoord& compute_with_storage_grid_size,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    return ttnn::prim::concatenate_heads(
        input_tensor, compute_with_storage_grid_size, memory_config, optional_output_tensor);
}

}  // namespace ttnn::experimental
