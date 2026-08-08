// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct KdaAffinePrefixParams {
    uint32_t batch_heads;
    uint32_t groups_per_head;
    uint32_t key_dim;
    uint32_t value_dim;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct KdaAffinePrefixInputs {
    Tensor transform_a;
    Tensor transform_b;
    Tensor initial_state;
};

}  // namespace ttnn::prim
