// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold.hpp"

#include "device/fold_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_fold {
Tensor MorehFold::invoke(
    const Tensor& input,
    const std::optional<Tensor>& output,
    const std::vector<uint32_t>& output_size,
    const std::vector<uint32_t>& kernel_size,
    const std::vector<uint32_t>& dilation,
    const std::vector<uint32_t>& padding,
    const std::vector<uint32_t>& stride,
    const std::optional<MemoryConfig>& memory_config) {
    return ttnn::prim::moreh_fold(input, output, output_size, kernel_size, dilation, padding, stride, memory_config);
}

}  // namespace ttnn::operations::moreh::moreh_fold
