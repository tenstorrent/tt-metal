// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <tuple>

namespace ttnn::prim {

struct EmaParams {
    float alpha{};
    CoreCoord grid_size;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;

    static constexpr auto attribute_names =
        std::forward_as_tuple("alpha", "grid_size", "output_mem_config", "compute_kernel_config");
    auto attribute_values() const {
        return std::forward_as_tuple(alpha, grid_size, output_mem_config, compute_kernel_config);
    }
};

struct EmaInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;

    static constexpr auto attribute_names = std::forward_as_tuple("input", "optional_output_tensor");
    auto attribute_values() const { return std::forward_as_tuple(input, optional_output_tensor); }
};

}  // namespace ttnn::prim
