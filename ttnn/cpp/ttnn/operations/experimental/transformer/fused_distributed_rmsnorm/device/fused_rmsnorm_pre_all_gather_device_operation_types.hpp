// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::transformer::fused_rmsnorm_pre_all_gather {

struct FusedRmsnormPreAllGatherParams {
    tt::tt_metal::DataType dtype;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct FusedRmsnormPreAllGatherInputs {
    Tensor input_tensor;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::experimental::transformer::fused_rmsnorm_pre_all_gather
