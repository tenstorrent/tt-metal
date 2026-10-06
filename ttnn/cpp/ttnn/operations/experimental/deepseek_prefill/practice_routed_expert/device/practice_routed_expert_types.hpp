// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert {

struct PracticeRoutedExpertParams {
    ttnn::DeviceComputeKernelConfig compute_kernel_config;

    static constexpr auto attribute_names = std::forward_as_tuple("compute_kernel_config");

    auto attribute_values() const { return std::forward_as_tuple(compute_kernel_config); }
};

struct PracticeRoutedExpertInputs {
    Tensor x;
    Tensor w_gate;
    Tensor w_up;
    Tensor w_down;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::practice_routed_expert
