// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct DitLayernormPreAllGatherParams {
    std::optional<tt::tt_metal::DataType> dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig memory_config;

    static constexpr auto attribute_names = std::forward_as_tuple("dtype", "compute_kernel_config", "memory_config");
    auto attribute_values() const { return std::forward_as_tuple(dtype, compute_kernel_config, memory_config); }
};

}  // namespace ttnn::experimental::prim
