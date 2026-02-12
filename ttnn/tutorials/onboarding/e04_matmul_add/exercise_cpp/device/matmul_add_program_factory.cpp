// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Exercise: Implement the ProgramFactory for matmul + add

#include "matmul_add_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::onboarding::exercise {

MatmulAddOperation::ProgramFactory::cached_program_t MatmulAddOperation::ProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    // TODO: Implement ProgramFactory::create
    TT_THROW("Not implemented");
}

void MatmulAddOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t&, const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&) {
    // TODO: Implement override_runtime_arguments
}

}  // namespace ttnn::operations::onboarding::exercise
