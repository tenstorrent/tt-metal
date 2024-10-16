// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_norm.hpp"

#include "device/moreh_norm_device_operation.hpp"

namespace ttnn::operations::moreh::moreh_norm {
Tensor MorehNorm::invoke(const Tensor& input,
                         float p,
                         std::optional<std::variant<int64_t, std::vector<int64_t>>> dim,
                         bool keepdim,
                         const std::optional<Tensor>& output,
                         const std::optional<MemoryConfig>& memory_config,
                         const std::optional<DeviceComputeKernelConfig>& compute_kernel_config) {
    if (!dim.has_value()) {
        std::vector<int64_t> dims(input.get_legacy_shape().rank());
        std::iota(dims.begin(), dims.end(), 0);
        dim = std::make_optional(dims);
    }
    if (auto single_dim = std::get_if<int64_t>(&dim.value()))
        return ttnn::prim::moreh_norm(input, p, *single_dim, keepdim, output, memory_config, compute_kernel_config);

    auto dims = std::get<std::vector<int64_t>>(dim.value());
    if (dims.empty()) {
        std::vector<int64_t> all_dims(input.get_legacy_shape().rank());
        std::iota(all_dims.begin(), all_dims.end(), 0);
        dims = all_dims;
    }
    if (dims.size() == 1)
        return ttnn::prim::moreh_norm(input, p, dims[0], keepdim, output, memory_config, compute_kernel_config);

    std::sort(dims.begin(), dims.end(), std::greater<int64_t>());
    auto tmp_output =
        ttnn::prim::moreh_norm(input, p, dims.front(), keepdim, std::nullopt, memory_config, compute_kernel_config);
    using idx_t = decltype(dims.size());
    for (idx_t idx = 1; idx < dims.size() - 1; ++idx)
        tmp_output = ttnn::prim::moreh_norm(
            tmp_output, p, dims[idx], keepdim, std::nullopt, memory_config, compute_kernel_config);
    return ttnn::prim::moreh_norm(tmp_output, p, dims.back(), keepdim, output, memory_config, compute_kernel_config);
}
}  // namespace ttnn::operations::moreh::moreh_norm
