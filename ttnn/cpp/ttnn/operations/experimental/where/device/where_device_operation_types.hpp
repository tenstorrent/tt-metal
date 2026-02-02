// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include "ttnn/tensor/types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct WhereParams {
    const tt::tt_metal::MemoryConfig memory_config;
    const tt::tt_metal::DataType dtype;
    const CoreRangeSet worker_grid;

    std::optional<DeviceComputeKernelConfig> compute_kernel_config;

    tt::stl::hash::hash_t to_hash() const {
        // hash has to exclude the scalar value
        return tt::stl::hash::hash_objects_with_default_seed(memory_config, dtype, compute_kernel_config);
    }
};

struct WhereInputs {
    const Tensor& condition_tensor;
    Tensor true_value_tensor;
    Tensor false_value_tensor;
    std::optional<Tensor> output_tensor;
};

}  // namespace ttnn::experimental::prim
