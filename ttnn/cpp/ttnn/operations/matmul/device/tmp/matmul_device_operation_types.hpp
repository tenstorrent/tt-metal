// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/tmp/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::operations::matmul {

struct operation_attributes_t {
    std::optional<MatmulProgramConfig> program_config;
    std::optional<bool> bcast_batch;
    tt::tt_metal::MemoryConfig output_mem_config;
    std::optional<tt::tt_metal::DataType> output_dtype;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;
    bool untilize_out{false};
    std::optional<CoreCoord> user_core_coord;
    std::optional<ttnn::operations::unary::UnaryWithParam> user_fused_activation;
    bool user_run_batched{false};
    bool transpose_a{false};
    bool transpose_b{false};
    std::optional<tt::tt_metal::Tile> output_tile;
    std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
};

struct tensor_args_t {
    std::vector<Tensor> input_tensors;                                // a,b, weights
    std::vector<std::optional<const Tensor>> optional_input_tensors;  // bias
    std::vector<std::optional<Tensor>> optional_output_tensors;       // output
};

using spec_return_value_t = std::vector<ttnn::TensorSpec>;
using tensor_return_value_t = std::vector<Tensor>;

}  // namespace ttnn::operations::matmul
