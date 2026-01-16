// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "minimal_matmul_split_program_factory.hpp"
#include "minimal_matmul_program_factory.hpp"

namespace ttnn::operations::experimental::minimal_matmul::program {

MinimalMatmulSplitProgramFactory::cached_program_t MinimalMatmulSplitProgramFactory::create(
    const split_operation_attributes_t& operation_attributes,
    const split_tensor_args_t& tensor_args,
    split_tensor_return_value_t& tensor_return_value) {
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
    const split_operation_attributes_t& /*operation_attributes*/,
    const split_tensor_args_t& tensor_args,
    split_tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& override_variables = cached_program.shared_variables;

    auto in0_addr = tensor_args.input_tensor.buffer()->address();
    auto in1_addr = tensor_args.weight_tensor.buffer()->address();
    auto in2_addr = tensor_args.bias_tensor.has_value() ? tensor_args.bias_tensor.value().buffer()->address() : 0;

    auto& in0_sender_runtime_args = GetRuntimeArgs(program, override_variables.in0_sender_kernels_id);
    auto& in0_receiver_runtime_args = GetRuntimeArgs(program, override_variables.in0_receiver_kernels_id);
    auto& in1_sender_runtime_args = GetRuntimeArgs(program, override_variables.in1_sender_kernels_id);
    auto& in1_receiver_runtime_args = GetRuntimeArgs(program, override_variables.in1_receiver_kernels_id);

    // RT args layout for in0: [in0_addr, in2_addr, in3_addr, is_sink, noc_coords(4), tile_ranges(4), defer_k,
    // out_addrs(N)...] RT args layout for in1: [in1_addr, in2_addr, is_sink, noc_coords(4), tile_ranges(4), defer_k,
    // out_addrs(N)...]
    constexpr uint32_t in0_in2_addr_idx = 1;
    constexpr uint32_t in0_out_addr_start_idx = 13;  // After defer_write_k_block
    constexpr uint32_t in1_in2_addr_idx = 1;
    constexpr uint32_t in1_out_addr_start_idx = 12;  // After defer_write_k_block

    for (uint32_t i = 0; i < override_variables.num_cores; ++i) {
        CoreCoord core = override_variables.cores.at(i);
        uint32_t in0_idx = override_variables.transpose_core_grid ? core.x : core.y;
        uint32_t in1_idx = override_variables.transpose_core_grid ? core.y : core.x;

        if (in1_idx == 0) {
            auto& in0_sender_args = in0_sender_runtime_args[core.x][core.y];
            in0_sender_args[0] = in0_addr;
            in0_sender_args[in0_in2_addr_idx] = in2_addr;
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in0_sender_args[in0_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        } else {
            auto& in0_receiver_args = in0_receiver_runtime_args[core.x][core.y];
            in0_receiver_args[in0_in2_addr_idx] = in2_addr;
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in0_receiver_args[in0_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        }

        if (in0_idx == 0) {
            auto& in1_sender_args = in1_sender_runtime_args[core.x][core.y];
            in1_sender_args[0] = in1_addr;
            in1_sender_args[in1_in2_addr_idx] = in2_addr;
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in1_sender_args[in1_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        } else {
            auto& in1_receiver_args = in1_receiver_runtime_args[core.x][core.y];
            in1_receiver_args[in1_in2_addr_idx] = in2_addr;
            // Update N output addresses at the end
            for (size_t out_idx = 0; out_idx < tensor_return_value.size(); ++out_idx) {
                in1_receiver_args[in1_out_addr_start_idx + out_idx] = tensor_return_value[out_idx].buffer()->address();
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::minimal_matmul::program
