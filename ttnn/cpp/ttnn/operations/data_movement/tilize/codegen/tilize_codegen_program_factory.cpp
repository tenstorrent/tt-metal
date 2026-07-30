// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_program_factory.hpp"

#include <tt_stl/assert.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

ProgramDescriptor TilizeCodegenProgramFactory::create_descriptor(
    const TilizeCodegenParams& /*operation_attributes*/,
    const TilizeCodegenInputs& /*tensor_args*/,
    Tensor& /*tensor_return_value*/) {
    // Placeholder for phase 4a: builder translation into descriptor CBs/kernels/runtime args.
    TT_THROW("TilizeCodegenProgramFactory::create_descriptor is not yet implemented");
}

}  // namespace ttnn::prim
