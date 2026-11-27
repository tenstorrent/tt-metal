// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm.hpp"

#include "device/moreh_norm_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/moreh/moreh_abs_pow/moreh_abs_pow.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn {

Tensor moreh_norm(
    const Tensor& input,
    float p,
    const std::optional<std::variant<int64_t, ttsl::SmallVector<int64_t>>>& dim,
    bool keepdim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    // get_dim() normalizes dim into a concrete list of dims, handling every form uniformly:
    // nullopt (Python dim=None) and an empty list both expand to the full dim range, a single
    // int becomes {d}, and a list is taken as-is. Single-dim cases are then handled by the
    // dims.size() == 1 branch below, so dim is never accessed via dim.value() directly.
    auto dims = operations::get_dim(dim, input.logical_shape().rank());
    auto INF = std::numeric_limits<float>::infinity();

    if (dims.size() == 1) {
        if (p == 0.0 || p == INF || p == -INF) {
            return ttnn::prim::moreh_norm(input, p, dims[0], keepdim, output, memory_config, compute_kernel_config);
        }
        auto tmp_output = ttnn::moreh_abs_pow(input, p, std::nullopt, memory_config, compute_kernel_config);
        tmp_output = ttnn::moreh_sum(tmp_output, dims[0], keepdim, std::nullopt, memory_config, compute_kernel_config);
        return ttnn::moreh_abs_pow(tmp_output, 1.0f / p, output, memory_config, compute_kernel_config);
    }
    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());

    if (p == 0) {
        auto tmp_output =
            ttnn::prim::moreh_norm(input, p, dims.front(), keepdim, std::nullopt, memory_config, compute_kernel_config);
        dims.erase(dims.begin());
        return ttnn::moreh_sum(tmp_output, dims, keepdim, output, memory_config, compute_kernel_config);
    }
    if (p == INF || p == -INF) {
        auto tmp_output =
            ttnn::prim::moreh_norm(input, p, dims.front(), keepdim, std::nullopt, memory_config, compute_kernel_config);
        using idx_t = decltype(dims.size());
        for (idx_t idx = 1; idx < dims.size() - 1; ++idx) {
            tmp_output = ttnn::prim::moreh_norm(
                tmp_output, p, dims[idx], keepdim, std::nullopt, memory_config, compute_kernel_config);
        }
        return ttnn::prim::moreh_norm(
            tmp_output, p, dims.back(), keepdim, output, memory_config, compute_kernel_config);
    }
    auto tmp_output = ttnn::moreh_abs_pow(input, p, std::nullopt, memory_config, compute_kernel_config);
    tmp_output = ttnn::moreh_sum(tmp_output, dims, keepdim, std::nullopt, memory_config, compute_kernel_config);
    return ttnn::moreh_abs_pow(tmp_output, 1.0f / p, output, memory_config, compute_kernel_config);
}

}  // namespace ttnn
