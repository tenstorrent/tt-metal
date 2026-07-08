// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "repeat_and_interleave_eltwise_mul_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct RepeatAndInterleaveEltwiseMulProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const RepeatMulParams& operation_attributes, const RepeatMulInputs& tensor_args, Tensor& tensor_return_value);
};

}  // namespace ttnn::experimental::prim
