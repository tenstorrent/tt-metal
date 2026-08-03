// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingHfParams {
    bool is_decode_mode;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct RotaryEmbeddingHfInputs {
    ttnn::Tensor input_tensor;
    ttnn::Tensor cos_cache;
    ttnn::Tensor sin_cache;
};

}  // namespace ttnn::experimental::prim
