// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim::qsr {

// Quasar clone of ttnn::prim::UpsampleParams (nearest mode only: no bilinear sliding-window config).
struct UpsampleParams {
    float scale_factor_h = 1.0f;
    float scale_factor_w = 1.0f;
    std::string mode = "nearest";
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

}  // namespace ttnn::prim::qsr
