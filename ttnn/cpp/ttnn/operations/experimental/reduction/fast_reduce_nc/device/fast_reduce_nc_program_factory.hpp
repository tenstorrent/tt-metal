// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "fast_reduce_nc_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct FastReduceNCProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FastReduceNCParams& operation_attributes,
        const FastReduceNCInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
