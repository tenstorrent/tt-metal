// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn::operations::moreh::moreh_fold {
struct MorehFold {
    static Tensor invoke(
        const Tensor& input,
        const std::optional<Tensor>& output,
        const std::vector<uint32_t>& output_size,
        const std::vector<uint32_t>& kernel_size,
        const std::vector<uint32_t>& dilation,
        const std::vector<uint32_t>& padding,
        const std::vector<uint32_t>& stride,
        const std::optional<MemoryConfig>& memory_config);
};
}  // namespace ttnn::operations::moreh::moreh_fold

namespace ttnn {
constexpr auto moreh_fold =
    ttnn::register_operation<"ttnn::moreh_fold", ttnn::operations::moreh::moreh_fold::MorehFold>();
}
