// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "all_gather_for_matmul_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::prim {

struct AllGatherForMatmulProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const AllGatherForMatmulParams& operation_attributes,
        const AllGatherForMatmulInputs& tensor_args,
        Tensor& output);
};

}  // namespace ttnn::prim
