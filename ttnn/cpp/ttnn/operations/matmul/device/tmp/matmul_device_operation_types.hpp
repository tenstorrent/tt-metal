// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/tmp/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

// TODO [migration]: Remove `using` aliases and the `tmp` namespace after migrating all dependent ops from the old
// infra. Once complete, the old infra code can be deleted. #33531
using namespace ttnn::operations::matmul::config;
namespace ttnn::operations::matmul::tmp {

enum class Matmul1DType { MCAST_IN0, GATHER_IN0, MCAST_IN1 };
}

namespace ttnn::operations::matmul {

// from struct Matrix which is used but run
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
    Tensor input_tensor_a;
    Tensor input_tensor_b;
    std::optional<Tensor> bias;
    std::optional<Tensor> output_tensor;
};

using spec_return_value_t = ttnn::TensorSpec;
using tensor_return_value_t = Tensor;

}  // namespace ttnn::operations::matmul
