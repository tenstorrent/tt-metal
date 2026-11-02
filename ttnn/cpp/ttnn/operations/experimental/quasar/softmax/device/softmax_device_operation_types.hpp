// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/quasar/common/thread_plan.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim::qsr {

// Attributes of the Quasar softmax (last dimension only, so no dim is carried).
struct SoftmaxParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
    bool numeric_stable = true;
    ThreadPlan threads;
};

struct SoftmaxInputs {
    Tensor input;
};

}  // namespace ttnn::prim::qsr
