// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/concat/codegen/concat_codegen_program_factory.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor ConcatCodegenProgramFactory::create_descriptor(
    const ConcatCodegenParams& /*operation_attributes*/,
    const ConcatCodegenInputs& /*tensor_args*/,
    Tensor& /*tensor_return_value*/) {
    TT_THROW("ConcatCodegenProgramFactory::create_descriptor is not yet implemented");
}

}  // namespace ttnn::prim
