// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct ReduceAffineTransformsParams {
    uint32_t batch_heads;
    uint32_t groups_per_head;
    uint32_t key_dim;
    uint32_t value_dim;
    uint32_t sequence_parallel_axis;
    uint32_t local_rows;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct ReduceAffineTransformsInputs {
    Tensor a;
    Tensor b;
    Tensor actual_start;
};

}  // namespace ttnn::experimental::prim
