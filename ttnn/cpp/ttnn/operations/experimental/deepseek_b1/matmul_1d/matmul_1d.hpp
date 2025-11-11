// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"

namespace ttnn::operations::experimental::deepseek_b1::matmul_1d {

using ttnn::operations::matmul::MatmulMultiCoreReuseMultiCast1DProgramConfig;
using ttnn::operations::unary::UnaryWithParam;

struct Matmul1DOperation {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor_a,
        const ttnn::Tensor& input_tensor_b,
        const ttnn::CoreGrid& core_grid,
        const std::size_t in0_block_w,
        const std::size_t out_subblock_h,
        const std::size_t out_subblock_w,
        const std::size_t per_core_M,
        const std::size_t per_core_N,
        const bool fuse_batch = true,
        const bool mcast_in0 = true,
        const std::optional<const ttnn::MemoryConfig>& memory_config = std::nullopt,
        const std::optional<const ttnn::DataType>& dtype = std::nullopt,
        const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt);
};

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d

namespace ttnn::experimental::deepseek_b1 {

constexpr auto matmul_1d = ttnn::register_operation<
    "ttnn::experimental::deepseek_b1::matmul_1d",
    ttnn::operations::experimental::deepseek_b1::matmul_1d::Matmul1DOperation>();

}  // namespace ttnn::experimental::deepseek_b1
