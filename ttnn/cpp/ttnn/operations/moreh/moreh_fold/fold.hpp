// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {

Tensor moreh_fold(
    const Tensor& input,
    const std::optional<Tensor>& output,
    const std::vector<uint32_t>& output_size,
    const std::vector<uint32_t>& kernel_size,
    const std::vector<uint32_t>& dilation = {1, 1},
    const std::vector<uint32_t>& padding = {0, 0},
    const std::vector<uint32_t>& stride = {1, 1},
    const std::optional<MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn
