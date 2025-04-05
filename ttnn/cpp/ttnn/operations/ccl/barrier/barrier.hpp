// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include <tt-metalium/fabric_edm_types.hpp>
#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/types.hpp"
#include "ttnn/types.hpp"

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
