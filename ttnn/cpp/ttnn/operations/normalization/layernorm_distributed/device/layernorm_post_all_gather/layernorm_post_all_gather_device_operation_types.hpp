// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/normalization/layernorm_distributed/device/layernorm_distributed_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>

namespace ttnn::operations::normalization::layernorm_post_all_gather {

struct operation_attributes_t {
    LayerNormDistributedType norm_type{};
    float eps{};
    tt::tt_metal::MemoryConfig memory_config;
    DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    std::optional<bool> use_2d_core_grid;
    LayerNormDistributedDefaultProgramConfig program_config;
};

struct tensor_args_t {
    Tensor input;
    Tensor stats;
    std::optional<Tensor> gamma;
    std::optional<Tensor> beta;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = TensorSpec;

}  // namespace ttnn::operations::normalization::layernorm_post_all_gather
