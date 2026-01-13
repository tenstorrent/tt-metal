// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/graph/graph_type_serialization.hpp"
#include "ttnn/graph/graph_registration.hpp"
#include <variant>

namespace ttnn {

// Compute kernel config serialization
TTNN_DEFINE_REFLECTION_SERIALIZATION(GrayskullComputeKernelConfig);
TTNN_DEFINE_REFLECTION_SERIALIZATION(WormholeComputeKernelConfig);

inline std::ostream& operator<<(
    std::ostream& os, const std::variant<GrayskullComputeKernelConfig, WormholeComputeKernelConfig>& value) {
    std::visit([&os](const auto& v) { os << v; }, value);
    return os;
}

}  // namespace ttnn

// Automatic type registration
TTNN_REGISTER_GRAPH_ARG(ttnn::GrayskullComputeKernelConfig);
TTNN_REGISTER_GRAPH_ARG(ttnn::WormholeComputeKernelConfig);
TTNN_REGISTER_GRAPH_ARG(std::variant<ttnn::GrayskullComputeKernelConfig, ttnn::WormholeComputeKernelConfig>);
