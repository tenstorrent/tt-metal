// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "create_qkv_heads_from_separate_tensors_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::operations::experimental::create_qkv_heads_from_separate_tensors {

struct CreateQKVHeadsSeparateTensorsProgramFactory {
    struct shared_variables_t {
        uint32_t cb_in0_id;
        uint32_t cb_in1_id;
        uint32_t cb_out0_id;
        uint32_t cb_out1_id;
        uint32_t cb_out2_id;
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

}  // namespace ttnn::operations::experimental::create_qkv_heads_from_separate_tensors
