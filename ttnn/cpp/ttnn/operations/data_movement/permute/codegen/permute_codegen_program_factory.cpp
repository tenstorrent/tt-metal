// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/permute/codegen/permute_codegen_program_factory.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor PermuteCodegenProgramFactory::create_descriptor(
    const PermuteCodegenParams& /*operation_attributes*/,
    const PermuteCodegenInputs& /*tensor_args*/,
    Tensor& /*tensor_return_value*/) {
    TT_THROW("PermuteCodegenProgramFactory is not implemented");
}

}  // namespace ttnn::prim
