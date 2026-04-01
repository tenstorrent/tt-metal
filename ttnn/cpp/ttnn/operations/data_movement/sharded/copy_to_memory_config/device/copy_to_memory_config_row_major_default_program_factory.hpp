// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device_operation.hpp"
#include "copy_to_memory_config_device_operation_types.hpp"

namespace ttnn::prim {

struct CopyToMemoryConfigRowMajorDefaultProgramFactory {
    using operation_attributes_t = CopyToMemoryConfigOperationAttributes;
    using tensor_args_t = CopyToMemoryConfigTensorArgs;
    using spec_return_value_t = CopyToMemoryConfigSpecReturnValue;
    using tensor_return_value_t = CopyToMemoryConfigTensorReturnValue;

    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id{};
        tt::tt_metal::KernelHandle writer_kernel_id{};
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& output_tensor);
};

}  // namespace ttnn::prim
