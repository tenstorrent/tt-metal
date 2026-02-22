// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_add_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <fmt/ostream.h>
#include <cstdint>
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

    fmt::print("Creating program for MatmulAdd with A shape {}, B shape {}\n", a_shape, b_shape);

    uint32_t M = a_shape[-2];
    uint32_t K = a_shape[-1];
    uint32_t N = b_shape[-1];
    uint32_t Mt = M / TILE_HEIGHT;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t Nt = N / TILE_WIDTH;

    tt::tt_metal::IDevice* device = a.device();  // ← Get device from tensor

    auto core_grid = device->compute_with_storage_grid_size();
    auto num_cores_y = core_grid.y;
    auto num_output_tiles_total = (M * N) / TILE_HW;

    auto [num_cores, all_cores, core_group_1, core_group_2, work_per_core1, work_per_core2] =
        split_work_to_cores(core_grid, num_output_tiles_total);

    Program program{};
    // CoreCoord core{0, 0};

    tt::DataFormat cb_format = datatype_to_dataformat_converter(a.dtype());
    // Tile size: element_size * TILE_HEIGHT * TILE_WIDTH
    constexpr uint32_t tile_size = sizeof(bfloat16) * TILE_HEIGHT * TILE_WIDTH;  // 2048 bytes for bfloat16

    // Circular buffers
    auto cb_a = CBIndex::c_0;
    auto cb_b = CBIndex::c_1;
    auto cb_c = CBIndex::c_2;
    auto cb_out = CBIndex::c_16;

    create_cb(cb_a, program, all_cores, tile_size, 2, cb_format);
    create_cb(cb_b, program, all_cores, tile_size, 2, cb_format);
    create_cb(cb_c, program, all_cores, tile_size, 2, cb_format);
    create_cb(cb_out, program, all_cores, tile_size, 2, cb_format);

    // Custom kernels - compile args include TensorAccessorArgs + dimensions
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*a.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*b.buffer()).append_to(reader_ct_args);
    TensorAccessorArgs(*c.buffer()).append_to(reader_ct_args);
    reader_ct_args.push_back(Mt);
    reader_ct_args.push_back(Kt);
    reader_ct_args.push_back(Nt);

    auto reader_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e04_matmul_add/solution_cpp/device/kernels/reader.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_1, .noc = NOC::RISCV_1_default, .compile_args = reader_ct_args});

    std::vector<uint32_t> writer_ct_args;
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct_args);
    writer_ct_args.push_back(Mt);
    writer_ct_args.push_back(Nt);

    auto writer_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e04_matmul_add/solution_cpp/device/kernels/writer.cpp",
        all_cores,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .compile_args = writer_ct_args});

    auto compute_id = CreateKernel(
        program,
        "ttnn/tutorials/onboarding/e04_matmul_add/solution_cpp/device/kernels/compute.cpp",
        all_cores,
        ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .compile_args = {Mt, Kt, Nt}});

    // Runtime args: only buffer addresses (dimensions are compile args)
    uint32_t work_offset = 0;
    auto work_groups = {std::make_pair(core_group_1, work_per_core1), std::make_pair(core_group_2, work_per_core2)};

    // Iterate through each work group and assign work to cores
    for (const auto& [ranges, work_per_core] : work_groups) {
        for (const auto& range : ranges.ranges()) {
            for (const auto& core : range) {
                // Set arguments for the reader kernel (data input)
                tt_metal::SetRuntimeArgs(
                    program,
                    reader_id,
                    core,
                    {a.buffer()->address(),  // Address of matrix A in DRAM
                     b.buffer()->address(),  // Address of matrix B in DRAM
                     c.buffer()->address(),  // Address of matrix C in DRAM
                     work_offset,            // Starting offset for this core's work
                     work_per_core});        // Amount of work for this core

                // Set arguments for the writer kernel (data output)
                tt_metal::SetRuntimeArgs(
                    program, writer_id, core, {output.buffer()->address(), work_per_core, work_offset});
                // Set arguments for the compute kernel
                tt_metal::SetRuntimeArgs(
                    program,
                    compute_id,
                    core,
                    {
                        work_per_core  // Amount of work for this core
                    });
                work_offset += work_per_core;  // Update offset for next core
            }
        }
    }

    return {std::move(program), {reader_id, writer_id, compute_id, num_cores, num_cores_y}};
}

void MatmulAddOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    auto& program = cached_program.program;
    // CoreCoord core{0, 0};

    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_id, core);
            reader_args[0] = tensor_args.a.buffer()->address();
            reader_args[1] = tensor_args.b.buffer()->address();
            reader_args[2] = tensor_args.c.buffer()->address();
        }

        {
            auto& writer_args = GetRuntimeArgs(program, cached_program.shared_variables.writer_id, core);
            writer_args[0] = output.buffer()->address();
        }
    }

    // auto& reader_args = GetRuntimeArgs(program, cached_program.shared_variables.reader_id, all_cores);
    // reader_args[0] = tensor_args.a.buffer()->address();
    // reader_args[1] = tensor_args.b.buffer()->address();
    // reader_args[2] = tensor_args.c.buffer()->address();

    // auto& writer_args = GetRuntimeArgs(program, cached_program.shared_variables.writer_id, all_cores);
    // writer_args[0] = output.buffer()->address();
}

}  // namespace ttnn::operations::onboarding
