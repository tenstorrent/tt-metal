// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unary_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct UnaryShardedProgramFactory {
    struct shared_variables_t {
        tt::tt_metal::CBHandle cb_src0;
        tt::tt_metal::CBHandle out_cb;
    };
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(const UnaryParams& args, const UnaryInputs& tensor_args, Tensor& output);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const UnaryParams& operation_attributes,
        const UnaryInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
