// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct QkvCausalConv1dSiluParams {
    uint32_t sequence;
    uint32_t q_width;
    uint32_t k_width;
    uint32_t v_width;
    uint32_t channel_chunk_size;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct QkvCausalConv1dSiluInputs {
    Tensor input;
    Tensor history;
    Tensor tap0;
    Tensor tap1;
    Tensor tap2;
    Tensor tap3;
};

}  // namespace ttnn::experimental::prim
