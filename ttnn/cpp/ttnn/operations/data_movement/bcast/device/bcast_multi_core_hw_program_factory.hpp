// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "bcast_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

struct BcastMultiCoreHWProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const BcastParams& operation_attributes, const BcastInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
