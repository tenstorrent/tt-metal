// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "sharded_rmsnorm_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm {

struct ShardedRmsNormSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id = 0;
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    tt::tt_metal::CBHandle cb_in = 0;
    tt::tt_metal::CBHandle cb_x = 0;
    tt::tt_metal::CBHandle cb_out = 0;
    tt::tt_metal::CBHandle cb_tile_out = 0;
    bool input_is_rm = false;
    bool output_is_sharded = false;
    bool output_is_rm = false;
    std::vector<CoreCoord> cores;
};

struct ShardedRmsNormProgramFactory {
    using shared_variables_t = ShardedRmsNormSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const ShardedRmsNormParams& operation_attributes,
        const ShardedRmsNormInputs& tensor_args,
        ShardedRmsNormTensorReturn& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const ShardedRmsNormParams& operation_attributes,
        const ShardedRmsNormInputs& tensor_args,
        ShardedRmsNormTensorReturn& tensor_return_value);
};

}  // namespace ttnn::operations::experimental::deepseek::sharded_rmsnorm
