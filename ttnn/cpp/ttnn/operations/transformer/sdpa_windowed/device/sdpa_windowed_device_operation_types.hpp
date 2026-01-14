// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::transformer::sdpa_windowed {

struct operation_attributes_t {
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct tensor_args_t {
    Tensor q;
    Tensor k;
    Tensor v;
    Tensor cu_window_seqlens;
};

}  // namespace ttnn::operations::transformer::sdpa_windowed
