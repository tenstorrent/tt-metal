// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "pad_codegen_device_operation_types.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct PadCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PadCodegenParams& operation_attributes, const PadCodegenInputs& tensor_args, Tensor& output_tensor);
};

}  // namespace ttnn::prim
