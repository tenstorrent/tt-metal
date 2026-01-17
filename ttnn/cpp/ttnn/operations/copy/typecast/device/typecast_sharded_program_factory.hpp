// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "typecast_device_op_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct TypecastShardedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle cb_src0;
        tt::tt_metal::CBHandle out_cb;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const TypecastParams& operation_attributes,
        const TypecastInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
