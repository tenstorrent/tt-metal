// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm.hpp"

#include "device/moreh_norm_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_abs_pow/moreh_abs_pow.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"

namespace ttnn::operations::moreh::moreh_norm {
Tensor MorehNorm::invoke(
    const Tensor& input,
    float p,
    std::optional<std::variant<int64_t, ttnn::SmallVector<int64_t>>> dim,
    bool keepdim,
    const std::optional<Tensor>& output,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (!dim.has_value()) {
        ttnn::SmallVector<int64_t> dims(input.padded_shape().rank());
        std::iota(dims.begin(), dims.end(), 0);
        dim = std::make_optional(dims);
    }
    auto INF = std::numeric_limits<float>::infinity();
    if (auto single_dim = std::get_if<int64_t>(&dim.value())) {
        if (p == 0.0 || p == INF || p == -INF) {
            return ttnn::prim::moreh_norm(input, p, *single_dim, keepdim, output, memory_config, compute_kernel_config);
        }
        auto tmp_output = ttnn::moreh_abs_pow(input, p, std::nullopt, memory_config, compute_kernel_config);
        tmp_output =
            ttnn::moreh_sum(tmp_output, *single_dim, keepdim, std::nullopt, memory_config, compute_kernel_config);
        return ttnn::moreh_abs_pow(tmp_output, 1.0f / p, output, memory_config, compute_kernel_config);
    }

    auto dims = std::get<ttnn::SmallVector<int64_t>>(dim.value());
    if (dims.empty()) {
        ttnn::SmallVector<int64_t> all_dims(input.padded_shape().rank());
        std::iota(all_dims.begin(), all_dims.end(), 0);
        dims = all_dims;
    }
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
    } else if (p == INF || p == -INF) {
        auto tmp_output =
            ttnn::prim::moreh_norm(input, p, dims.front(), keepdim, std::nullopt, memory_config, compute_kernel_config);
        using idx_t = decltype(dims.size());
        for (idx_t idx = 1; idx < dims.size() - 1; ++idx) {
            tmp_output = ttnn::prim::moreh_norm(
                tmp_output, p, dims[idx], keepdim, std::nullopt, memory_config, compute_kernel_config);
        }
        return ttnn::prim::moreh_norm(
            tmp_output, p, dims.back(), keepdim, output, memory_config, compute_kernel_config);
    } else {
        auto tmp_output = ttnn::moreh_abs_pow(input, p, std::nullopt, memory_config, compute_kernel_config);
        tmp_output = ttnn::moreh_sum(tmp_output, dims, keepdim, std::nullopt, memory_config, compute_kernel_config);
        return ttnn::moreh_abs_pow(tmp_output, 1.0f / p, output, memory_config, compute_kernel_config);
    }
}
}  // namespace ttnn::operations::moreh::moreh_norm
