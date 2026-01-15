// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "common.hpp"

namespace ttnn::operations::reduction::generic {

struct GenericParams {
    tt::tt_metal::ReduceOpMath math_op{};
    tt::tt_metal::ReduceOpDim dim{};
    float scaler{1.0f};
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype{tt::tt_metal::DataType::INVALID};
    ttnn::DeviceComputeKernelConfig compute_kernel_config;
    std::optional<tt::tt_metal::CoreRangeSet> sub_core_grids;
};

struct GenericInputs {
    Tensor input_tensor;
};

}  // namespace ttnn::operations::reduction::generic
