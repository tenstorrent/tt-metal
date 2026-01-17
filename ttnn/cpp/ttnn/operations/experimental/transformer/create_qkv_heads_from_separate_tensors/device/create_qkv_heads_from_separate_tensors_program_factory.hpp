// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "create_qkv_heads_from_separate_tensors_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

struct CreateQKVHeadsSeparateTensorsProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle cb_in0_id = 0;
        tt::tt_metal::CBHandle cb_in1_id = 0;
        tt::tt_metal::CBHandle cb_out0_id = 0;
        tt::tt_metal::CBHandle cb_out1_id = 0;
        tt::tt_metal::CBHandle cb_out2_id = 0;
    };

    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const CreateQKVHeadsFromSeparateTensorsParams& operation_attributes,
        const CreateQKVHeadsFromSeparateTensorsInputs& tensor_args,
        CreateQKVHeadsFromSeparateTensorsResult& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const CreateQKVHeadsFromSeparateTensorsParams& operation_attributes,
        const CreateQKVHeadsFromSeparateTensorsInputs& tensor_args,
        CreateQKVHeadsFromSeparateTensorsResult& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
