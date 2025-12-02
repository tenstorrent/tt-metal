// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "reduce_op_device_operation_types.hpp"

namespace ttnn::operations::reduction::generic::program {

struct ReduceSingleCoreHwProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        tt::tt_metal::CoreCoord selected_core_coord;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        ReduceTensorReturnValue& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        ReduceTensorReturnValue& tensor_return_value);
};

struct ReduceMultiCoreHProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::CBHandle cb_src1;
        tt::tt_metal::CBHandle cb_output;
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        ReduceTensorReturnValue& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        ReduceTensorReturnValue& tensor_return_value);
};

struct ReduceMultiCoreWProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        ReduceTensorReturnValue& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        ReduceTensorReturnValue& tensor_return_value);
};

}  // namespace ttnn::operations::reduction::generic::program
