// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct DitLayernormPostAllGatherParams {
    float eps = 0.0f;
    tt::tt_metal::MemoryConfig memory_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::DataType> dtype;
};

struct DitLayernormPostAllGatherInputs {
    Tensor input;
    Tensor stats;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
};

}  // namespace ttnn::experimental::prim
