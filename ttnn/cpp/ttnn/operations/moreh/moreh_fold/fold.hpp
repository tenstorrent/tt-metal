// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <optional>
#include <vector>

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace tt {
namespace tt_metal {
struct MemoryConfig;
}  // namespace tt_metal
}  // namespace tt

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
    ttnn::register_operation_with_auto_launch_op<"ttnn::moreh_fold", ttnn::operations::moreh::moreh_fold::MorehFold>();
}
