// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "prod_all_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct ProdAllProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle unary_reader_kernel_id{};
        tt::tt_metal::KernelHandle unary_writer_kernel_id{};
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ProdAllParams& operation_attributes, const ProdAllInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ProdAllParams& operation_attributes,
        const ProdAllInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
