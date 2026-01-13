// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::transformer::dit_layernorm {

struct PostAllGatherOperationAttributes {
    float eps;
    tt::tt_metal::MemoryConfig memory_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::DataType> dtype;
};

struct PostAllGatherTensorArgs {
    const Tensor& input;
    const Tensor& stats;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
};

using PostAllGatherTensorReturnValue = Tensor;
using PostAllGatherSpecReturnValue = TensorSpec;

}  // namespace ttnn::operations::experimental::transformer::dit_layernorm
