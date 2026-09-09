// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "attn_res_weighted_reduce_nc_device_operation_types.hpp"

namespace ttnn::experimental::prim {

struct AttnResWeightedReduceNCProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AttnResWeightedReduceNCParams& operation_attributes,
        const AttnResWeightedReduceNCInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
