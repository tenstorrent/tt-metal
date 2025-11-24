// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_gate_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::reduction {

GroupedGateDeviceOperation::ProgramFactory::cached_program_t GroupedGateDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& scores = tensor_args.scores;
    const auto& bias = tensor_args.bias;
    auto& output_weights = std::get<0>(tensor_return_value);
    auto& output_indices = std::get<1>(tensor_return_value);

    IDevice* device = scores.device();
    auto program = CreateProgram();

    CoreRangeSet all_cores({
        CoreRange(CoreCoord(0, 0), CoreCoord(0, 0))  // Dummy core range
    });

    // Compute kernel
    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/grouped_gate/device/kernels/compute/grouped_gate.cpp",
        all_cores,
        ComputeConfig{.math_approx_mode = true, .compile_args = {}});

    // Reader kernel
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/grouped_gate/device/kernels/dataflow/reader_grouped_gate.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default});

    // Writer kernel
    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/grouped_gate/device/kernels/dataflow/writer_grouped_gate.cpp",
        all_cores,
        DataMovementConfig{.processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default});

    return {std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id}};
}

void GroupedGateDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Placeholder for runtime argument override logic
}

}  // namespace ttnn::operations::reduction
