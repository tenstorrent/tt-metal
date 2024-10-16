// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_sum.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_sum/device/moreh_sum_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_sum {
Tensor MorehSum::invoke(const Tensor& input,
                        std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
                        const bool keepdim,
                        const std::optional<Tensor>& output,
                        const std::optional<MemoryConfig>& memory_config,
                        const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    std::vector<int64_t> dims = tt::operations::primary::get_dim(dim, input.get_legacy_shape().rank());
    std::sort(dims.begin(), dims.end());

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
}  // namespace ttnn::operations::moreh::moreh_sum
