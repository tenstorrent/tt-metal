// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/repeat/device/repeat_operation_types.hpp"

namespace ttnn::operations::data_movement::repeat {

tt::tt_metal::operation::ProgramWithCallbacks rm_repeat_program_factory(
    const Tensor& input, uint32_t num_repeats, const Tensor& output, bool is_last_dim);

struct RepeatSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id;
    tt::tt_metal::CoreRange total_cores;
};

struct RepeatProgramFactorySecondDim {
    using shared_variables_t = RepeatSharedVariables;
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

struct RepeatProgramFactoryLastDim {
    using shared_variables_t = RepeatSharedVariables;
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

}  // namespace ttnn::operations::data_movement::repeat
