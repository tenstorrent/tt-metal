// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "dit_layernorm_post_all_gather_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

struct PostAllGatherWelfordProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const DitLayernormPostAllGatherParams& operation_attributes,
        const DitLayernormPostAllGatherInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::experimental::prim
