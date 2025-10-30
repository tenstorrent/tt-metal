// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ax_plus_b_device_operation.hpp"
#include "tt_stl/assert.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::ax_plus_b {
AX_plus_B_DeviceOperation::SingleCore::cached_program_t AX_plus_B_DeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_args.tensor_x.dtype());

    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const uint32_t num_tiles = tensor_args.tensor_x.physical_volume() / tt::constants::TILE_HW;

    CoreCoord compute_with_storage_grid_size = {1, 1};
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    auto create_input_circular_buffer = [&](uint32_t cb_index) {
        const uint32_t num_input_tiles = 2;
        tt::tt_metal::CircularBufferConfig cb_config =
            tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{cb_index, cb_data_format}})
                .set_page_size(cb_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_config);
    };

    // Create circular buffers for 'a', 'x' and 'b' inputs and 'y' output
    create_input_circular_buffer(tt::CBIndex::c_0);   // cb_a
    create_input_circular_buffer(tt::CBIndex::c_1);   // cb_x
    create_input_circular_buffer(tt::CBIndex::c_2);   // cb_b
    create_input_circular_buffer(tt::CBIndex::c_16);  // cb_b

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*tensor_args.tensor_a.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*tensor_args.tensor_x.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*tensor_args.tensor_b.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ax_plus_b/device/kernels/reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ax_plus_b/device/kernels/compute.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{.math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false});

    std::vector<uint32_t> writer_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*tensor_return_value.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/ax_plus_b/device/kernels/writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {tensor_args.tensor_a.buffer()->address(),
             tensor_args.tensor_x.buffer()->address(),
             tensor_args.tensor_b.buffer()->address(),
             num_tiles_per_core,
             num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles_per_core, num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {tensor_return_value.buffer()->address(), num_tiles_per_core, num_tiles_written});

        num_tiles_written += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .num_cores = num_cores,
         .num_cores_y = num_cores_y}};
}

void AX_plus_B_DeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& num_cores = cached_program.shared_variables.num_cores;
    auto& num_cores_y = cached_program.shared_variables.num_cores_y;

    // todo: assert num_cores is 1 and num_cores_y is 1, as this is a single-core version.
    TT_ASSERT(num_cores == 1 && num_cores_y == 1);

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = tensor_args.tensor_a.buffer()->address();
            runtime_args[1] = tensor_args.tensor_x.buffer()->address();
            runtime_args[2] = tensor_args.tensor_b.buffer()->address();
        }

        {
            auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = tensor_return_value.buffer()->address();
        }
    }
}

}  // namespace ttnn::operations::ax_plus_b
