// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/matmul_device_operation_types.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"

namespace ttnn::operations::matmul {
namespace bmm_op_utils {
std::tuple<uint32_t, uint32_t> get_matmul_subblock_params(
    const uint32_t per_core_M,
    const uint32_t per_core_N,
    const bool per_core_M_equals_subblock_h_constraint,
    const bool per_core_N_equals_subblock_w_constraint,
    const bool fp32_dest_acc_en);
}

MatmulProgramConfig create_simple_matmul_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    const std::optional<const ttnn::DeviceComputeKernelConfig> compute_kernel_config,
    const CoreCoord& compute_with_storage_grid_size,
    const tt::tt_metal::MemoryConfig& mem_config,
    const tt::tt_metal::DataType output_dtype);

MatmulProgramConfig get_program_config(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const uint32_t bias_single_tile_size,
    const matmul::operation_attributes_t& attributes);

}  // namespace ttnn::operations::matmul
