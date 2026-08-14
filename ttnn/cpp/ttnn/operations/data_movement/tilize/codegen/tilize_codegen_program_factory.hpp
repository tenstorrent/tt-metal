// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"

namespace ttnn::prim {

struct TilizeCodegenOperationAttributes;
struct TilizeCodegenTensorArgs;

struct TilizeCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const TilizeCodegenOperationAttributes& operation_attributes,
        const TilizeCodegenTensorArgs& tensor_args,
        const Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
