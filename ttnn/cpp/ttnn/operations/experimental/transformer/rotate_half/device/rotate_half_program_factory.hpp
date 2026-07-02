// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "rotate_half_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct RotateHalfProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RotateHalfParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
