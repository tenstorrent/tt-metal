// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_fused_swiglu_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu {

tt::tt_metal::ProgramDescriptor create_moe_fused_swiglu_program_descriptor(
    const OperationArguments& operation_arguments,
    const TensorArguments& tensor_arguments,
    Tensor& tensor_return_value);

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_fused_swiglu
