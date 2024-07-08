// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/experimental/tensor/tensor_utils.hpp"
#include "ttnn/experimental/tt_dnn/op_library/bcast/bcast_op.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/experimental/tt_dnn/op_library/eltwise_unary/eltwise_unary_op.hpp"
#include "tt_metal/common/core_coord.h"
#include "tt_metal/impl/dispatch/command_queue.hpp"

namespace ttnn {

using MatmulMultiCoreReuseProgramConfig = tt::operations::primary::MatmulMultiCoreReuseProgramConfig;
using MatmulMultiCoreReuseMultiCastProgramConfig = tt::operations::primary::MatmulMultiCoreReuseMultiCastProgramConfig;
using MatmulMultiCoreReuseMultiCast1DProgramConfig =
    tt::operations::primary::MatmulMultiCoreReuseMultiCast1DProgramConfig;
// MatmulProgramConfig is the Union of the above types
using MatmulProgramConfig = tt::operations::primary::MatmulProgramConfig;
namespace operations {
namespace matmul {

namespace detail {

inline bool is_input_batched(const ttnn::Shape& shape);

}  // namespace detail

std::optional<UnaryWithParam> get_fused_activation(const std::optional<const std::string>& activation);

ttnn::Tensor matmul(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const std::optional<const ttnn::Tensor>& bias,
    const bool transpose_a = false,
    const bool transpose_b = false,
    const std::optional<const MatmulProgramConfig> program_config = std::nullopt,
    const ttnn::MemoryConfig& memory_config = ttnn::DRAM_MEMORY_CONFIG,
    std::optional<const DataType> dtype = std::nullopt,
    const std::optional<const std::string>& activation = std::nullopt,
    const std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    const std::optional<const ttnn::CoreGrid> core_grid = std::nullopt,
    const bool propagate_is_b_batched = false);

}  // namespace matmul
}  // namespace operations
}  // namespace ttnn
