// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

// Non-tensor configuration for the per-tile triangle-solve op.
struct TriangleSolveParams {
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

// Input tensors. Both TILE, bf16, logical shape [1, 1, 32, 32]:
//   l_neg : negated unit-lower-triangular L (strict-lower entries pre-negated)
//   rhs   : right-hand-side matrix
struct TriangleSolveInputs {
    Tensor l_neg;
    Tensor rhs;
};

}  // namespace ttnn::experimental::prim
