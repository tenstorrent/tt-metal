// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::experimental::prim {

struct DitLayernormPreAllGatherParams {
    std::optional<tt::tt_metal::DataType> dtype;
    DeviceComputeKernelConfig compute_kernel_config;
    tt::tt_metal::MemoryConfig memory_config;
};

}  // namespace ttnn::experimental::prim
