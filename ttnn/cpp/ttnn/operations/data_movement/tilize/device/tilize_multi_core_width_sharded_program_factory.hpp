// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

struct TilizeMultiCoreWidthShardedProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
