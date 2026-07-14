// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/device/regime_a_matmul_config.hpp"

namespace ttnn::experimental::prim {

struct RegimeAMatmulParams {
    // The whole config drives compile-time kernel args, so it lives in operation_attributes and is
    // keyed by the framework's default reflection-based program hash.
    std::optional<RegimeAMatmulConfig> config;
    std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;

    DeviceComputeKernelConfig compute_kernel_config;
};

struct RegimeAMatmulInputs {
    Tensor input_tensor;   // in0 : [.., M, K], DRAM interleaved, bf16, TILE
    Tensor weight_tensor;  // in1 : [.., K, N], DRAM width-sharded (8 banks), bf16, TILE
};

}  // namespace ttnn::experimental::prim
