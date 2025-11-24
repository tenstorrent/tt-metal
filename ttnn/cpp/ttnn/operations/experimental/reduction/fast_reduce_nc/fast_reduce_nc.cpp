// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/fast_reduce_nc.hpp"
#include "ttnn/operations/experimental/reduction/fast_reduce_nc/device/fast_reduce_nc_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn {
namespace operations::experimental::reduction {

ttnn::Tensor FastReduceNCOperation::invoke(
    const ttnn::Tensor& input,
    tt::stl::Span<const int32_t> dims,
    const std::optional<const Tensor>& output,
    const ttnn::MemoryConfig& memory_config,
    std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config) {
    TT_FATAL(
        input.storage_type() == StorageType::DEVICE,
        "Input tensor storage type must be DEVICE but got {}",
        input.storage_type());

    TT_FATAL(!dims.empty(), "fast_reduce_nc dims should not be empty");

    auto kernel_config_val =
        init_device_compute_kernel_config(input.device()->arch(), compute_kernel_config, MathFidelity::HiFi4);

    ttnn::SmallVector<int32_t> sorted_dims(dims.begin(), dims.end());
    std::sort(sorted_dims.begin(), sorted_dims.end());

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        auto temp_output =
            ttnn::prim::fast_reduce_nc(temp_input, sorted_dims[i], std::nullopt, memory_config, kernel_config_val);
        temp_input = temp_output;
    }
    return ttnn::prim::fast_reduce_nc(temp_input, sorted_dims.front(), output, memory_config, kernel_config_val);
}

}  // namespace operations::experimental::reduction

}  // namespace ttnn
