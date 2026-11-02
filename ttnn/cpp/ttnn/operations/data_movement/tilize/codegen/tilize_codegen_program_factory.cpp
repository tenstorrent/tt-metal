// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_codegen_program_factory.hpp"

#include <tt_stl/assert.hpp>

#include "tilize_codegen_device_operation.hpp"

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor TilizeCodegenProgramFactory::create_descriptor(
    const TilizeCodegenOperationAttributes&, const TilizeCodegenTensorArgs&, const Tensor&) {
    TT_THROW("TilizeCodegenProgramFactory is not implemented yet");
}

}  // namespace ttnn::prim
