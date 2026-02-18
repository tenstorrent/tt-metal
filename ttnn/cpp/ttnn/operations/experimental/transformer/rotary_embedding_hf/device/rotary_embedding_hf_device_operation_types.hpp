// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct RotaryEmbeddingHfParams {
    bool is_decode_mode;
    tt::tt_metal::MemoryConfig output_mem_config;
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
};

struct RotaryEmbeddingHfInputs {
    tt::tt_metal::Tensor input_tensor;
    tt::tt_metal::Tensor cos_cache;
    tt::tt_metal::Tensor sin_cache;
};

}  // namespace ttnn::experimental::prim
