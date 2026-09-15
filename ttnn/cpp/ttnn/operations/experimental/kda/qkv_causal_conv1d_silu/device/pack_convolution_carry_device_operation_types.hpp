// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct PackConvolutionCarryParams {
    uint32_t sequence;
    uint32_t channels;
    uint32_t wrap_row;
    uint32_t history_rows;
    tt::tt_metal::MemoryConfig output_mem_config;
    DeviceComputeKernelConfig compute_kernel_config;
};

struct PackConvolutionCarryInputs {
    Tensor input;
    Tensor wrap_indicator;
};

}  // namespace ttnn::experimental::prim
