// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
// #include "ttnn/operations/eltwise/lazy/expression.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::fused {

struct TensorAttributes {
    DataType data_type;
    // TODO add SmallVector of singleton dimensions for broadcasting
};

struct MaterializeDeviceOperation {
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    struct operation_attributes_t {
        // lazy::Expression expression;
        MemoryConfig memory_config;
        std::optional<DataType> dtype;
        std::optional<DeviceComputeKernelConfig> compute_kernel_config;

        DataType get_dtype() const;
    };

    // struct tensor_args_t {
    //     const Tensor
    // };
};

}  // namespace ttnn::operations::fused
