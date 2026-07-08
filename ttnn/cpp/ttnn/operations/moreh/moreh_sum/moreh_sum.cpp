// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_sum/device/moreh_sum_device_operation.hpp"
#include "ttnn/operations/data_movement/reshape_view/reshape.hpp"

namespace ttnn {

Tensor moreh_sum(
    const Tensor& input,
    const std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>>& dim,
    const bool keepdim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    ttsl::SmallVector<int64_t> dims = operations::get_dim(dim, input.logical_shape().rank());
    std::sort(dims.begin(), dims.end());

    if (input.logical_shape().rank() == 1 && dims.size() == 1 && dims.front() == 0) {
        const auto input_width = input.logical_shape()[0];
        auto rank_2_input = ttnn::reshape(input, ttnn::Shape({1, input_width}));
        auto reduced =
            ttnn::prim::moreh_sum(rank_2_input, /*dim=*/1, keepdim, output, memory_config, compute_kernel_config);
        return output.has_value() ? reduced : ttnn::reshape(reduced, ttnn::Shape({1}));
    }

    auto temp_input = input;
    for (uint32_t i = dims.size() - 1; i > 0; i--) {
        log_debug(tt::LogOp, "{}:{} dim {} keepdim {}", __func__, __LINE__, dims[i], keepdim);
        auto temp_output =
            ttnn::prim::moreh_sum(temp_input, dims[i], keepdim, std::nullopt, memory_config, compute_kernel_config);
        temp_input = temp_output;
    }
    log_debug(tt::LogOp, "{}:{} dim {} keepdim {}", __func__, __LINE__, dims.front(), keepdim);
    return ttnn::prim::moreh_sum(temp_input, dims.front(), keepdim, output, memory_config, compute_kernel_config);
}

}  // namespace ttnn
