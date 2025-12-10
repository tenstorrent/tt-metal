// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa_config.hpp"

namespace ttnn::operations::transformer::ring_distributed_sdpa {

struct operation_attributes_t {
    uint32_t ring_size = 0;
    std::optional<uint32_t> ring_id;
    std::optional<float> scale;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<SDPAProgramConfig> program_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct tensor_args_t {
    ttnn::Tensor q;
    ttnn::Tensor k;
    ttnn::Tensor v;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = ttnn::Tensor;

}  // namespace ttnn::operations::transformer::ring_distributed_sdpa
