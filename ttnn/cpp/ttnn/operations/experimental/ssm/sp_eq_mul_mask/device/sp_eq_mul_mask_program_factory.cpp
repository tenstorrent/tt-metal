// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "sp_eq_mul_mask_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "sp_eq_mul_mask_device_operation_types.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

SpEqMulMaskProgramFactory::cached_program_t SpEqMulMaskProgramFactory::create(
    const SpEqMulMaskParams& /*operation_attributes*/, const SpEqMulMaskInputs& tensor_args, Tensor& output) {
    Program program = CreateProgram();

    auto* input_a_buffer = tensor_args.a.buffer();
    auto* input_b_buffer = tensor_args.b.buffer();
    Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer must be allocated on device");

    const auto& shape = tensor_args.a.padded_shape();
    const uint32_t num_tiles = shape.volume() / (TILE_HEIGHT * TILE_WIDTH);

    const bool row_major = false;
    auto compute_with_storage_grid = tensor_args.a.device()->compute_with_storage_grid_size();
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_g1, num_tiles_per_core_g2] =
            split_work_to_cores(compute_with_storage_grid, num_tiles, row_major);

    const tt::DataFormat data_format = datatype_to_dataformat_converter(tensor_args.a.dtype());
    const uint32_t tile_size = tt::tile_size(data_format);

    constexpr uint32_t cb_size = 2;  // double-buffer

    const auto mk_cb = [&program, &cores = all_cores, tile_size, data_format](uint32_t index) {
        const CircularBufferConfig cfg =
            CircularBufferConfig(cb_size * tile_size, {{index, data_format}}).set_page_size(index, tile_size);
        return CreateCircularBuffer(program, cores, cfg);
    };

    const uint32_t cb_a = tt::CBIndex::c_0;
    const uint32_t cb_b = tt::CBIndex::c_1;
    const uint32_t cb_out = tt::CBIndex::c_2;
    mk_cb(cb_a);
    mk_cb(cb_b);
    mk_cb(cb_out);

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(input_a_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(input_b_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args;
    TensorAccessorArgs(out_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/sp_eq_mul_mask/device/kernels/sp_eq_mul_mask_reader.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/sp_eq_mul_mask/device/kernels/sp_eq_mul_mask_writer.cpp",
        all_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/ssm/sp_eq_mul_mask/device/kernels/sp_eq_mul_mask_compute.cpp",
        all_cores,
        ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = false,
            .math_approx_mode = false,
            .compile_args = {}});

    uint32_t g1_numcores = core_group_1.num_cores();
    uint32_t g2_numcores = core_group_2.num_cores();
    std::vector<CoreCoord> cores =
        grid_to_cores(num_cores, compute_with_storage_grid.x, compute_with_storage_grid.y, row_major);

    uint32_t tiles_written = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t tiles_this_core =
            (i < g1_numcores) ? num_tiles_per_core_g1 : num_tiles_per_core_g2;

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            cores[i],
            {input_a_buffer->address(),
             input_b_buffer->address(),
             tiles_this_core,
             tiles_written});

        SetRuntimeArgs(
            program,
            writer_kernel_id,
            cores[i],
            {out_buffer->address(), tiles_this_core, tiles_written});

        SetRuntimeArgs(program, compute_kernel_id, cores[i], {tiles_this_core});

        tiles_written += tiles_this_core;
    }

    SpEqMulMaskSharedVariables shared{
        reader_kernel_id,
        writer_kernel_id,
        compute_kernel_id,
        num_cores,
        all_cores,
        cores,
        g1_numcores,
        g2_numcores,
        num_tiles_per_core_g1,
        num_tiles_per_core_g2,
    };
    return cached_program_t{std::move(program), std::move(shared)};
}

void SpEqMulMaskProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const SpEqMulMaskParams& /*operation_attributes*/,
    const SpEqMulMaskInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;

    Buffer* a_buf = tensor_args.a.buffer();
    Buffer* b_buf = tensor_args.b.buffer();
    Buffer* out_buf = tensor_return_value.buffer();

    uint32_t tiles_written = 0;
    for (uint32_t i = 0; i < shared.num_cores; ++i) {
        uint32_t tiles_this_core =
            (i < shared.g1_numcores) ? shared.num_tiles_per_core_g1 : shared.num_tiles_per_core_g2;

        SetRuntimeArgs(
            program,
            shared.reader_kernel_id,
            shared.cores[i],
            {a_buf->address(), b_buf->address(), tiles_this_core, tiles_written});
        SetRuntimeArgs(
            program,
            shared.writer_kernel_id,
            shared.cores[i],
            {out_buf->address(), tiles_this_core, tiles_written});
        SetRuntimeArgs(program, shared.compute_kernel_id, shared.cores[i], {tiles_this_core});

        tiles_written += tiles_this_core;
    }
}

}  // namespace ttnn::experimental::prim
