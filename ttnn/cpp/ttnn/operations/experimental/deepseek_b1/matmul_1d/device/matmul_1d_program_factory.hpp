// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/tt_metal.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/circular_buffer.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::operations::experimental::deepseek_b1::matmul_1d {

// MCast 1D program factory
tt::tt_metal::operation::ProgramWithCallbacks deepseek_b1_matmul_multi_core_reuse_mcast_1d_optimized(
    const Tensor& a,
    const Tensor& b,
    const Tensor& output_tensor,
    CoreCoord compute_with_storage_grid_size,
    DeviceComputeKernelConfig compute_kernel_config);

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d
