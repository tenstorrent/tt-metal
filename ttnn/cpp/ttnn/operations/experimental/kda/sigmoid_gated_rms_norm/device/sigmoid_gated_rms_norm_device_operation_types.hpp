// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct SigmoidGatedRmsNormParams {
    uint32_t batch;
    uint32_t num_heads;
    uint32_t sequence;
    uint32_t value_dim;
    float epsilon;
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct SigmoidGatedRmsNormInputs {
    Tensor input;
    Tensor gate;
    Tensor weight;
};

}  // namespace ttnn::experimental::prim
