// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "layernorm_distributed_types.hpp"

namespace ttnn::operations::normalization {

struct operation_attributes_t {
    LayerNormDistributedType norm_type = LayerNormDistributedType::LAYERNORM;
    tt::tt_metal::DataType dtype = tt::tt_metal::DataType::INVALID;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<bool> use_2d_core_grid;
    LayerNormDistributedDefaultProgramConfig program_config;
};

struct tensor_args_t {
    Tensor input;
    std::optional<Tensor> preallocated_output;
};

using tensor_return_value_t = Tensor;

using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::normalization
