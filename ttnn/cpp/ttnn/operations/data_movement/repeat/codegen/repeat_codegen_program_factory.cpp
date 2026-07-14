// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.hpp"

#include <tt_stl/assert.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor RepeatCodegenProgramFactory::create_descriptor(
    const RepeatCodegenParams& /*operation_attributes*/,
    const RepeatCodegenInputs& /*tensor_args*/,
    Tensor& /*tensor_return_value*/) {
    TT_THROW("RepeatCodegenProgramFactory::create_descriptor is not yet implemented");
}

}  // namespace ttnn::prim
