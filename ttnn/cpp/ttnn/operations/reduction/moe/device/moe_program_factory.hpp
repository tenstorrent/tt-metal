// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/moe/device/moe_device_operation_types.hpp"

namespace ttnn::operations::reduction::moe::program {

struct MoeSharedVariables {
    tt::tt_metal::KernelHandle unary_reader_kernel_id{};
    tt::tt_metal::KernelHandle unary_writer_kernel_id{};
};

struct MoeProgramFactory {
    using shared_variables_t = MoeSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::reduction::moe::program
