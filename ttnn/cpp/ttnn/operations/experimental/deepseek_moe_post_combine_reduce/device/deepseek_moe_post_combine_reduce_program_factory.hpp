// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "deepseek_moe_post_combine_reduce_device_operation.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

using DeepseekMoEPostCombineReduceParams = DeepseekMoEPostCombineReduceDeviceOperationImpl::operation_attributes_t;
using DeepseekMoEPostCombineReduceInputs = DeepseekMoEPostCombineReduceDeviceOperationImpl::tensor_args_t;

struct DeepseekMoEPostCombineReduceProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle compute_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        tt::tt_metal::CBHandle output_cb_handle;
        std::vector<tt::tt_metal::CoreCoord> cores;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const DeepseekMoEPostCombineReduceParams& operation_attributes,
        const DeepseekMoEPostCombineReduceInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const DeepseekMoEPostCombineReduceParams& operation_attributes,
        const DeepseekMoEPostCombineReduceInputs& tensor_args,
        ttnn::Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim