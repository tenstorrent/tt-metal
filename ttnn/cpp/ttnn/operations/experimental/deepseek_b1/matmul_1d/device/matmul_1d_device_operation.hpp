// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/run_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include <tt-metalium/core_coord.hpp>

namespace ttnn::operations::experimental::deepseek_b1::matmul_1d {

// Program config for 1D mcast matmul
struct Matmul1DProgramConfig {
    CoreCoord compute_with_storage_grid_size;
};

// Device operation for 1D mcast matmul
struct Matmul1DDeviceOperation {
    const Matmul1DProgramConfig program_config;
    const std::optional<tt::tt_metal::MemoryConfig> output_mem_config;
    const std::optional<tt::tt_metal::DataType> output_dtype;
    const std::optional<DeviceComputeKernelConfig> compute_kernel_config;

    void validate(const std::vector<Tensor>& input_tensors) const;

    std::vector<ttnn::TensorSpec> compute_output_specs(const std::vector<Tensor>& input_tensors) const;

    tt::tt_metal::operation::ProgramWithCallbacks create_program(
        const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const;
};

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d
