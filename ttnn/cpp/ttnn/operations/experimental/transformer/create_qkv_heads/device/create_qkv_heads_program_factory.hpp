// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "create_qkv_heads_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct CreateQKVHeadsSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle compute_kernel_id = 0;
    tt::tt_metal::CBHandle cb_in0_id = 0;
    tt::tt_metal::CBHandle cb_out0_id = 0;
    tt::tt_metal::CBHandle cb_out1_id = 0;
    tt::tt_metal::CBHandle cb_out2_id = 0;
    CoreRangeSet all_cores;
    bool has_compute_kernel = false;
};

struct CreateQKVHeadsProgramFactory {
    using shared_variables_t = CreateQKVHeadsSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const CreateQKVHeadsParams& operation_attributes,
        const CreateQKVHeadsInputs& tensor_args,
        CreateQKVHeadsResult& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const CreateQKVHeadsParams& operation_attributes,
        const CreateQKVHeadsInputs& tensor_args,
        CreateQKVHeadsResult& output);
};

}  // namespace ttnn::experimental::prim
