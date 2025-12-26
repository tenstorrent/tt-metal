// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_backward_operation_types.hpp"

#include <tt-metalium/kernel_types.hpp>
#include <ttnn/device_operation.hpp>

#include <cstddef>

namespace ttnn::operations::normalization::softmax_backward {

struct SoftmaxBackwardProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id;
        tt::tt_metal::KernelHandle unary_writer_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        std::size_t num_cores;
        std::size_t num_cores_y;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

private:
    // Helper functions for small and large kernel implementations
    static cached_program_t create_small_kernel_program(
        tt::tt_metal::Program& program,
        distributed::MeshDevice* device,
        const ttnn::Tensor& softmax_output,
        const ttnn::Tensor& upstream_grad,
        ttnn::Tensor& tensor_return_value,
        uint32_t num_rows,
        uint32_t width_tiles,
        uint32_t mask_w,
        tt::DataFormat input_data_format,
        tt::DataFormat output_data_format,
        tt::DataFormat intermed_data_format,
        uint32_t input_tile_size,
        uint32_t output_tile_size,
        uint32_t intermed_tile_size);

    static cached_program_t create_large_kernel_program(
        tt::tt_metal::Program& program,
        distributed::MeshDevice* device,
        const ttnn::Tensor& softmax_output,
        const ttnn::Tensor& upstream_grad,
        ttnn::Tensor& tensor_return_value,
        uint32_t num_rows,
        uint32_t width_tiles,
        uint32_t mask_w,
        tt::DataFormat input_data_format,
        tt::DataFormat output_data_format,
        tt::DataFormat intermed_data_format,
        uint32_t input_tile_size,
        uint32_t output_tile_size,
        uint32_t intermed_tile_size);
};

}  // namespace ttnn::operations::normalization::softmax_backward
