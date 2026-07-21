// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pad_codegen_program_factory.hpp"

#include "tt_stl/assert.hpp"

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor PadCodegenProgramFactory::create_descriptor(
    const PadCodegenParams& /*operation_attributes*/,
    const PadCodegenInputs& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    TT_THROW("PadCodegenProgramFactory::create_descriptor: not yet implemented");
}

}  // namespace ttnn::prim
