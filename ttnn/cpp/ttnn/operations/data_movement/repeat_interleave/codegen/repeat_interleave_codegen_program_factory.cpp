// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat_interleave/codegen/repeat_interleave_codegen_program_factory.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor RepeatInterleaveCodegenProgramFactory::create_descriptor(
    const RepeatInterleaveCodegenParams& /*operation_attributes*/,
    const RepeatInterleaveCodegenInputs& /*tensor_args*/,
    Tensor& /*tensor_return_value*/) {
    TT_FATAL(false, "repeat_interleave codegen program factory is not implemented yet");
    return ProgramDescriptor{};
}

}  // namespace ttnn::prim
