// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/reduction/generic/device/reduce_op_device_operation_types.hpp"

namespace ttnn::operations::reduction::generic::program {

struct ReduceMultiCoreHProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::CBHandle cb_src1{};
        tt::tt_metal::CBHandle cb_output{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const GenericParams& operation_attributes, const GenericInputs& tensor_args, Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const GenericParams& operation_attributes,
        const GenericInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::operations::reduction::generic::program
