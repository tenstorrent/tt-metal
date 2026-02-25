// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "sign_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::onboarding {

SignBuggyOperation::ProgramFactory::cached_program_t SignBuggyOperation::ProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input = tensor_args.input;

    auto shape = input.logical_shape();
    uint32_t num_tiles = shape[-2] / TILE_HEIGHT * shape[-1] / TILE_WIDTH;

    Program program{};
    CoreCoord core{0, 0};

    tt::DataFormat cb_format = datatype_to_dataformat_converter(input.dtype());
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;

    // Circular buffers
    auto cb_in = CBIndex::c_0;    // Reader -> Compute
    auto cb_out = CBIndex::c_16;  // Compute -> Writer
    create_cb(cb_in, program, core, tile_size, 2, cb_format);
    create_cb(cb_out, program, core, tile_size, 2, cb_format);

    // Reader kernel
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct_args);
    reader_ct_args.push_back(num_tiles);

    auto reader_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e05_debugging/exercise_cpp/device/kernels/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    // Compute kernel
    auto compute_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e05_debugging/exercise_cpp/device/kernels/compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {num_tiles}});

    // Writer kernel
    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);
    writer_ct_args.push_back(num_tiles);

    auto writer_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e05_debugging/exercise_cpp/device/kernels/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    // Runtime args: only buffer addresses
    SetRuntimeArgs(program, reader_id, core, {input.buffer()->address()});
    SetRuntimeArgs(program, writer_id, core, {output.buffer()->address()});

    return {std::move(program), {reader_id, writer_id, compute_id}};
}

void SignBuggyOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    CoreCoord core{0, 0};

    auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_id, core);
    reader_args[0] = tensor_args.input.buffer()->address();

    auto& writer_args = GetRuntimeArgs(program, cached_program.shared_variables.writer_id, core);
    writer_args[0] = output.buffer()->address();
}

}  // namespace ttnn::operations::onboarding
