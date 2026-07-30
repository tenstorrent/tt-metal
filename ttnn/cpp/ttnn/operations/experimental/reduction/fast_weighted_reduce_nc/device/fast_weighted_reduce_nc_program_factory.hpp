// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "fast_weighted_reduce_nc_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct FastWeightedReduceNCProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const FastWeightedReduceNCParams& operation_attributes,
        const FastWeightedReduceNCInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
