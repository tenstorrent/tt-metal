// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/common/thread_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

// Attributes of the Quasar batch_norm (inference only: no training / momentum).
struct BatchNormParams {
    float eps = 1e-5f;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    ThreadPlan threads;
};

struct BatchNormInputs {
    Tensor input;                  // [N, C, H, W]
    Tensor running_mean;           // [1, C, 1, 1]
    Tensor running_var;            // [1, C, 1, 1]
    std::optional<Tensor> weight;  // [1, C, 1, 1]
    std::optional<Tensor> bias;    // [1, C, 1, 1]
};

}  // namespace ttnn::prim::qsr
