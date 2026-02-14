// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_add_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;

namespace ttnn::operations::onboarding {

MatmulAddOperation::ProgramFactory::cached_program_t MatmulAddOperation::ProgramFactory::create(
    const operation_attributes_t&, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& a = tensor_args.a;
    const auto& b = tensor_args.b;
    const auto& c = tensor_args.c;

    auto a_shape = a.logical_shape();
    auto b_shape = b.logical_shape();

    uint32_t M = a_shape[-2];
    uint32_t K = a_shape[-1];
    uint32_t N = b_shape[-1];
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    Program program{};
    CoreCoord core{0, 0};

    tt::DataFormat cb_format = datatype_to_dataformat_converter(a.dtype());
    // Tile size: element_size * TILE_HEIGHT * TILE_WIDTH
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2048 bytes for bfloat16

    // Circular buffers
    auto cb_a = CBIndex::c_0;
    auto cb_b = CBIndex::c_1;
    auto cb_c = CBIndex::c_2;
    auto cb_out = CBIndex::c_16;

    create_cb(cb_a, program, core, tile_size, 2, cb_format);
    create_cb(cb_b, program, core, tile_size, 2, cb_format);
    create_cb(cb_c, program, core, tile_size, 2, cb_format);
    create_cb(cb_out, program, core, tile_size, 2, cb_format);

    // Custom kernels
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*a.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*b.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*c.buffer()).append_to(reader_ct_args);

    auto reader_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e04_matmul_add/solution_cpp/device/kernels/reader.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);

    auto writer_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e04_matmul_add/solution_cpp/device/kernels/writer.cpp",
        core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    auto compute_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e04_matmul_add/solution_cpp/device/kernels/compute.cpp",
        core,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {Mt, Kt, Nt}});

    SetRuntimeArgs(
        program, reader_id, core, {a.buffer()->address(), b.buffer()->address(), c.buffer()->address(), Mt, Kt, Nt});
    SetRuntimeArgs(program, writer_id, core, {output.buffer()->address(), Mt, Nt});

    return {std::move(program), {reader_id, writer_id, compute_id}};
}

void MatmulAddOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    CoreCoord core{0, 0};

    auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_id, core);
    reader_args[0] = tensor_args.a.buffer()->address();
    reader_args[1] = tensor_args.b.buffer()->address();
    reader_args[2] = tensor_args.c.buffer()->address();

    auto& writer_args = GetRuntimeArgs(program, cached_program.shared_variables.writer_id, core);
    writer_args[0] = output.buffer()->address();
}

}  // namespace ttnn::operations::onboarding
