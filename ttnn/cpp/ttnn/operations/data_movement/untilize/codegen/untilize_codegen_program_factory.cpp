// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_codegen_program_factory.hpp"

#include "untilize_codegen_device_operation.hpp"

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor UntilizeCodegenProgramFactory::create_descriptor(
    const UntilizeCodegenOperationAttributes& /*operation_attributes*/,
    const UntilizeCodegenTensorArgs& /*tensor_args*/,
    const Tensor& /*tensor_return_value*/) {
    return {};
}

}  // namespace ttnn::prim
