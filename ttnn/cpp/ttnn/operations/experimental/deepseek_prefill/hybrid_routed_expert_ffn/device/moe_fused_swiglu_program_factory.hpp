// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "moe_fused_swiglu_types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::fused {

// Appends this implementation's circular buffers, semaphores and kernels to a descriptor the
// caller owns, so the merged op can carry both implementations in ONE program. Everything it adds
// is additive -- it reads no existing entry -- which is what lets the two halves be emitted in
// either order.
void append_to_descriptor(
    tt::tt_metal::ProgramDescriptor& descriptor,
    const OperationArguments& operation_arguments,
    const TensorArguments& tensor_arguments,
    Tensor& tensor_return_value);

// The standalone form, for a program that carries this implementation alone.
tt::tt_metal::ProgramDescriptor create_moe_fused_swiglu_program_descriptor(
    const OperationArguments& operation_arguments,
    const TensorArguments& tensor_arguments,
    Tensor& tensor_return_value);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::fused
