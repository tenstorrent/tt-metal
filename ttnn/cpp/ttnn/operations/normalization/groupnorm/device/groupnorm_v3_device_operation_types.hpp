// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::normalization::group_norm_v3 {

struct operation_attributes_t {
    uint32_t num_groups = 0;
    float eps = 0.0f;
    tt::tt_metal::DataType output_dtype = tt::tt_metal::DataType::INVALID;
    tt::tt_metal::MemoryConfig output_mem_config;
    CoreCoord core_grid;
    bool inplace = false;
    int chunk_size = 0;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct tensor_args_t {
    Tensor input;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
};

using spec_return_value_t = TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::normalization::group_norm_v3
