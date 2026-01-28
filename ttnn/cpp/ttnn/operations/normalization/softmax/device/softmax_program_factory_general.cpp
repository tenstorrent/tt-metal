// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_program_factory_general.hpp"

#include <tt-metalium/host_api.hpp>

namespace ttnn::prim {
// General-purpose softmax with arbitrary dimension support
void SoftmaxProgramFactoryGeneral::override_runtime_arguments(
    cached_program_t& cached_program,
    const SoftmaxParams& /*attributes*/,
    const SoftmaxInputs& tensor_args,
    Tensor& output_tensor) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;
    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = tensor_args.input_tensor.buffer()->address();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output_tensor.buffer()->address();
        }
    }
}

}  // namespace ttnn::prim
