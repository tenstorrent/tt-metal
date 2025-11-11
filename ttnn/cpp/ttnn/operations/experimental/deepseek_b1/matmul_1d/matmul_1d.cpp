// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_1d.hpp"
#include "device/matmul_1d_device_operation.hpp"

namespace ttnn::operations::experimental::deepseek_b1::matmul_1d {

ttnn::Tensor Matmul1DOperation::invoke(
    const ttnn::Tensor& input_tensor_a,
    const ttnn::Tensor& input_tensor_b,
    const ttnn::CoreGrid& core_grid,
    const std::size_t in0_block_w,
    const std::size_t out_subblock_h,
    const std::size_t out_subblock_w,
    const std::size_t per_core_M,
    const std::size_t per_core_N,
    const bool fuse_batch,
    const bool mcast_in0,
    const std::optional<const ttnn::MemoryConfig>& memory_config,
    const std::optional<const ttnn::DataType>& dtype,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    // Create the 1D mcast program config
    Matmul1DProgramConfig program_config{
        .compute_with_storage_grid_size = CoreCoord(core_grid.x, core_grid.y),
        .in0_block_w = in0_block_w,
        .out_subblock_h = out_subblock_h,
        .out_subblock_w = out_subblock_w,
        .per_core_M = per_core_M,
        .per_core_N = per_core_N,
        .fuse_batch = fuse_batch,
        .mcast_in0 = mcast_in0,
    };

    // Create device operation
    Matmul1DDeviceOperation device_op{
        .program_config = program_config,
        .output_mem_config = memory_config,
        .output_dtype = dtype,
        .compute_kernel_config = compute_kernel_config,
    };

    // Run the operation
    return tt::tt_metal::operation::run(device_op, {input_tensor_a, input_tensor_b}).at(0);
}

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d
