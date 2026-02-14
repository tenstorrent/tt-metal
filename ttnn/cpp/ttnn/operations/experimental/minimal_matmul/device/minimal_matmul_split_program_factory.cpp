// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_split_program_factory.hpp"
#include "minimal_matmul_program_factory.hpp"

namespace ttnn::experimental::prim {

MinimalMatmulSplitProgramFactory::cached_program_t MinimalMatmulSplitProgramFactory::create(
    const MinimalMatmulSplitParams& operation_attributes,
    const MinimalMatmulSplitInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    std::optional<ttnn::experimental::ccl::MinimalMatmulFusedOpSignaler> empty_fused_op_signaler;

    // Use the shared implementation from minimal_matmul_program_factory
    auto shared_vars = minimal_matmul_factory_helper_common(
        program,
        tensor_args.input_tensor,
        tensor_args.weight_tensor,
        tensor_args.bias_tensor,
        operation_attributes.fused_activation,
        operation_attributes.config,
        tensor_return_value,
        operation_attributes.compute_kernel_config,
        empty_fused_op_signaler,
        static_cast<uint32_t>(operation_attributes.chunks));

    return {std::move(program), std::move(shared_vars)};
}

void MinimalMatmulSplitProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const MinimalMatmulSplitParams& /*operation_attributes*/,
    const MinimalMatmulSplitInputs& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto in0_addr = tensor_args.input_tensor.mesh_buffer()->address();
    auto in1_addr = tensor_args.weight_tensor.mesh_buffer()->address();
    auto in2_addr = tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().mesh_buffer()->address() : 0;
    uint32_t in3_addr = 0;  // split variant doesn't use optional_input_tensor

    std::vector<uint32_t> output_addrs;
    output_addrs.reserve(tensor_return_value.size());
    for (const auto& output_tensor : tensor_return_value) {
        output_addrs.push_back(output_tensor.mesh_buffer()->address());
    }

    override_runtime_arguments_common(cached_program, in0_addr, in1_addr, in2_addr, in3_addr, output_addrs);
}

}  // namespace ttnn::experimental::prim
