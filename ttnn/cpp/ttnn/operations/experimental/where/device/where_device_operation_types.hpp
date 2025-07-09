// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::experimental::ternary::where_ttt_args {

using tensor_return_value_type = Tensor;

struct operation_attributes_type {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
    const CoreRangeSet worker_grid;

    std::optional<DeviceComputeKernelConfig> compute_kernel_config;

    tt::stl::hash::hash_t to_hash() const {
        // hash has to exclude the scalar value
        return tt::stl::hash::hash_objects_with_default_seed(memory_config, dtype, compute_kernel_config);
    }
};
struct tensor_args_type {
    const Tensor& condition_tensor;
    Tensor true_value_tensor;
    Tensor false_value_tensor;
    std::optional<Tensor> output_tensor;
};
}  // namespace ttnn::operations::experimental::ternary::where_ttt_args
