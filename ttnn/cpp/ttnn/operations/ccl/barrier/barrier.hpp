// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_types.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/barrier/device/barrier_op.hpp"

namespace ttnn {
namespace operations::ccl {

struct BarrierOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor,
        const std::optional<ttnn::MemoryConfig>& memory_config = std::nullopt,
        ttnn::ccl::Topology topology = ttnn::ccl::Topology::Ring);
};
}  // namespace operations::ccl
constexpr auto barrier = ttnn::register_operation<"ttnn::barrier", ttnn::operations::ccl::BarrierOperation>();
}  // namespace ttnn
