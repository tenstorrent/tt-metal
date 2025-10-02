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
};

}  // namespace ttnn::operations::normalization::softmax_backward
