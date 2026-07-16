// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "tilize_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// Retile factory: accepts an already-tiled input whose tile shape differs from the
// tile shape requested on the op, and re-lays it out into the requested tile shape.
struct TilizeMultiCoreRetileProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value);
};
}  // namespace ttnn::prim
