// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "fused_single_user_program_factory.hpp"
#include "fused_single_user_types.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::hyperconnection {

struct FusedSingleUserDotSharedVariables {
    FusedSingleUserSharedVariables hc;
    tt::tt_metal::KernelHandle dot_reader_kernel_id = 0;
    std::vector<CoreCoord> dot_cores;
    // Each dot core's K-slice: its byte offset into the collapse core's streams shard.
    std::vector<uint32_t> a_offsets;
    // fn_weight in L1: cb_in1 aliases its shard, else the reader copies its slab from DRAM.
    bool b_in_l1 = false;
    tt::tt_metal::CBHandle in1_cb = 0;
};

// The single-user program plus the fn dot that produces fused_w (see ttnn::prim::fused_hyperconnection_single_user).
struct FusedSingleUserDotProgramFactory {
    using shared_variables_t = FusedSingleUserDotSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const FusedSingleUserParams& operation_attributes,
        const FusedSingleUserInputs& tensor_args,
        FusedSingleUserTensorReturn& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const FusedSingleUserParams& operation_attributes,
        const FusedSingleUserInputs& tensor_args,
        FusedSingleUserTensorReturn& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::hyperconnection
