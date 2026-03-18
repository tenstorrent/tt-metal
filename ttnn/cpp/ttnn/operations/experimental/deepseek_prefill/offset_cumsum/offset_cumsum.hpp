// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>

#include "ttnn/operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"

namespace ttnn {
namespace operations::experimental {

struct OffsetCumsumOperation {
    static std::array<ttnn::Tensor, 2> invoke(
        const Tensor& input_tensor, uint32_t cluster_axis, uint32_t num_links, const ttnn::MemoryConfig& memory_config);
};

}  // namespace operations::experimental

constexpr auto offset_cumsum =
    ttnn::register_operation<"ttnn::offset_cumsum", ttnn::operations::experimental::OffsetCumsumOperation>();

}  // namespace ttnn
