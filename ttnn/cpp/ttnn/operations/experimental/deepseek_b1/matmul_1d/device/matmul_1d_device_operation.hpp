// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_types.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::experimental::deepseek_b1::matmul_1d {

using ttnn::operations::unary::UnaryWithParam;

// Program config for 1D mcast matmul
struct Matmul1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
    std::size_t in0_block_w{};
    std::size_t out_subblock_h{};
    std::size_t out_subblock_w{};
    std::size_t per_core_M{};
    std::size_t per_core_N{};
    bool fuse_batch{};
    bool mcast_in0{};
};

// Device operation for 1D mcast matmul
struct Matmul1DDeviceOperation {
    const Matmul1DProgramConfig program_config;
    const std::optional<MemoryConfig> output_mem_config;
    const std::optional<DataType> output_dtype;
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;

    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d
