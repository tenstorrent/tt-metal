// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct SparseMatmulParams {
    std::optional<uint32_t> nnz;
    bool is_input_a_sparse;
    bool is_input_b_sparse;
    std::optional<const operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    std::optional<DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    std::optional<const CoreCoord> user_core_coord = std::nullopt;
    std::optional<const tt::tt_metal::Tile> output_tile;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id;
};

struct SparseMatmulInputs {
    std::vector<Tensor> input_tensors;
    std::vector<std::optional<const Tensor>> optional_input_tensors;
    std::vector<std::optional<Tensor>> optional_output_tensors;
};

}  // namespace ttnn::prim
