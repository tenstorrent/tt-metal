// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::experimental::prim {

struct FastReduceNCParams {
    const int32_t dim;
    const tt::tt_metal::MemoryConfig output_mem_config;
    const ttnn::DeviceComputeKernelConfig compute_kernel_config;
    const std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
    // When set, packer writes this dtype instead of input.dtype() (used by the Sum precision chain).
    const std::optional<tt::tt_metal::DataType> output_dtype;
};

struct FastReduceNCInputs {
    const Tensor input;
    std::optional<Tensor> preallocated_output;
};

}  // namespace ttnn::experimental::prim
