// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "plusone_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct PlusOneProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PlusoneParams& operation_attributes, const Tensor& input, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
