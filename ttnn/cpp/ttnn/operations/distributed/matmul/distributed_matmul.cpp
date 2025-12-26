// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "distributed_matmul.hpp"

#include <tt-metalium/mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/matmul.hpp"

namespace ttnn::operations::distributed {

ttnn::Tensor ExecuteDistributedMatmul::invoke(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<ttnn::MemoryConfig>& memory_config,
    std::optional<ttnn::DataType> dtype) {
    // Get the mesh device from input tensor
    auto mesh_device = input_tensor_a.device();

    // Basic validation
    TT_FATAL(mesh_device != nullptr, "Input tensor must be on a device");
    TT_FATAL(input_tensor_a.device() == input_tensor_b.device(), "Both input tensors must be on the same device");

    // Log basic info
    log_debug(
        tt::LogOp,
        "Distributed Matmul called with input shapes: A={}, B={}",
        input_tensor_a.logical_shape(),
        input_tensor_b.logical_shape());

    // For now, this is a placeholder implementation that just calls the regular matmul
    // on each device. In the future, this will:
    // 1. Consider TensorTopology for input and output tensors
    // 2. Virtualize device boundaries
    // 3. Implement cross-device communication as needed
    // 4. Support arbitrary input/output global layouts

    auto memory_config_ = memory_config.value_or(input_tensor_a.memory_config());
    auto dtype_ = dtype.value_or(input_tensor_a.dtype());

    // Call the standard matmul operation
    // This will execute on all devices in the mesh
    auto output = ttnn::matmul(
        input_tensor_a,
        input_tensor_b,
        /*transpose_a=*/false,
        /*transpose_b=*/false,
        memory_config_,
        dtype_);

    log_debug(tt::LogOp, "Distributed Matmul output shape: {}", output.logical_shape());

    return output;
}

}  // namespace ttnn::operations::distributed
