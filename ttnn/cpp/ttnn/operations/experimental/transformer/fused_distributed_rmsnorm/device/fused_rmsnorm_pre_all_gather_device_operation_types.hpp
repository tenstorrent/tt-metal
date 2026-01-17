// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct FusedRmsnormPreAllGatherParams {
    tt::tt_metal::DataType dtype;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct FusedRmsnormPreAllGatherInputs {
    Tensor input_tensor;
};

}  // namespace ttnn::experimental::prim
