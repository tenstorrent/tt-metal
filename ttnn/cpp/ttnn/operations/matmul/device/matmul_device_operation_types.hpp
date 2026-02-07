// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/matmul/device/config/matmul_program_config_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "tt-metalium/global_circular_buffer.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operation.hpp"  // for DEFAULT_OUTPUT_MEMORY_CONFIG
#include <tuple>

namespace ttnn::prim {

struct MatmulParams {
    std::optional<operations::matmul::MatmulProgramConfig> program_config = std::nullopt;
    std::optional<bool> bcast_batch = std::nullopt;
    tt::tt_metal::MemoryConfig output_mem_config = tt::tt_metal::operation::DEFAULT_OUTPUT_MEMORY_CONFIG;
    std::optional<tt::tt_metal::DataType> output_dtype = std::nullopt;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config = std::nullopt;
    bool untilize_out = false;
    std::optional<CoreCoord> user_core_coord = std::nullopt;
    std::optional<ttnn::operations::unary::UnaryWithParam> user_fused_activation = std::nullopt;
    bool user_run_batched = false;
    bool transpose_a = false;
    bool transpose_b = false;
    std::optional<tt::tt_metal::Tile> output_tile = std::nullopt;
    std::optional<tt::tt_metal::experimental::GlobalCircularBuffer> global_cb = std::nullopt;
    std::optional<tt::tt_metal::SubDeviceId> sub_device_id = std::nullopt;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "program_config",
        "bcast_batch",
        "output_mem_config",
        "output_dtype",
        "compute_kernel_config",
        "untilize_out",
        "user_core_coord",
        "user_fused_activation",
        "user_run_batched",
        "transpose_a",
        "transpose_b",
        "output_tile",
        "global_cb",
        "sub_device_id");
    auto attribute_values() const {
        return std::forward_as_tuple(
            program_config,
            bcast_batch,
            output_mem_config,
            output_dtype,
            compute_kernel_config,
            untilize_out,
            user_core_coord,
            user_fused_activation,
            user_run_batched,
            transpose_a,
            transpose_b,
            output_tile,
            global_cb,
            sub_device_id);
    }
};

struct MatmulInputs {
    std::vector<Tensor> input_tensors;                                // a,b, weights
    std::vector<std::optional<const Tensor>> optional_input_tensors;  // bias
    std::vector<std::optional<Tensor>> optional_output_tensors;       // output

    static constexpr auto attribute_names =
        std::forward_as_tuple("input_tensors", "optional_input_tensors", "optional_output_tensors");
    auto attribute_values() const {
        return std::forward_as_tuple(input_tensors, optional_input_tensors, optional_output_tensors);
    }
};

}  // namespace ttnn::prim
