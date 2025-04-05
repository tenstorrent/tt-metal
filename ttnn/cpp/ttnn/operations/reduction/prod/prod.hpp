// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <stdint.h>
#include <functional>
#include <optional>

#include <tt-metalium/small_vector.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

namespace operations::reduction {

struct ProdOperation {
    static Tensor invoke(
        const Tensor& input,
        bool all_dimensions = false,
        int64_t dim = 0,
        const bool keepdim = false,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);

    static Tensor invoke(
        const Tensor& input,
        const Tensor& output,
        ttnn::SmallVector<int64_t>& dims,
        const std::optional<MemoryConfig>& memory_config = std::nullopt);
};

}  // namespace operations::reduction

constexpr auto prod =
    ttnn::register_operation_with_auto_launch_op<"ttnn::prod", ttnn::operations::reduction::ProdOperation>();

}  // namespace ttnn
