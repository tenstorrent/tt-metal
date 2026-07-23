// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/device_operation.hpp"
#include "reshape_device_operation_types.hpp"

namespace ttnn::prim {

struct ReshapeRMProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ttnn::prim::ReshapeOnDeviceParams& operation_attributes,
        const ttnn::prim::ReshapeOnDeviceInputs& tensor_args,
        ttnn::Tensor& output_tensor);
};

}  // namespace ttnn::prim
