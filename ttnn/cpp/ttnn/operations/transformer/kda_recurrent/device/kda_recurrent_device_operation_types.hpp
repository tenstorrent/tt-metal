// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct KDARecurrentParams {
    uint32_t num_heads;
    uint32_t key_dim;
    uint32_t value_dim;
    tt::tt_metal::MemoryConfig output_memory_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct KDARecurrentInputs {
    Tensor q_scaled;
    Tensor k_unit;
    Tensor v;
    Tensor decay;
    Tensor beta;
    Tensor state;
};

}  // namespace ttnn::prim
