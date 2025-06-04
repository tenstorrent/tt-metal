// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "example_multiple_return_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::examples {
ExampleMultipleReturnDeviceOperation::SingleCore::cached_program_t
ExampleMultipleReturnDeviceOperation::SingleCore::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;

    auto output_tensor1 = tensor_return_value.at(0);
    auto output_tensor2 = tensor_return_value.at(1);

    auto src_buffer = input_tensor.buffer();

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(cb_data_format);

    auto output_dtype =
        output_tensor1.has_value() ? output_tensor1.value().get_dtype() : output_tensor2.value().get_dtype();
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output_dtype);
    uint32_t single_tile_size_output = tt::tt_metal::detail::TileSize(cb_data_format_output);

    uint32_t num_tiles = input_tensor.volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input_tensor.device();

    CoreCoord compute_with_storage_grid_size = {1, 1};
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    auto cb_src0 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    auto cb_output1 = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};

    bool dst_is_dram1 = output_tensor1.has_value()
                            ? (output_tensor1.value().buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0)
                            : false;
    bool dst_is_dram2 = output_tensor2.has_value()
                            ? (output_tensor2.value().buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0)
                            : false;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index, (std::uint32_t)dst_is_dram1, (std::uint32_t)dst_is_dram2};

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/examples/example_multiple_return/device/kernels/writer_multiple.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1                            // per_core_block_size
    };

    bool math_approx_mode = false;
    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1                            // per_core_block_size
        };

        auto eltwise_unary_kernel_group_2_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/eltwise_sfpu.cpp",
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2});
    }

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
            program, unary_reader_kernel_id, core, {src_buffer->address(), num_tiles_per_core, num_tiles_written});

        auto dst_buffer1_address = output_tensor1.has_value() ? output_tensor1.value().buffer()->address() : 0;
        auto dst_buffer2_address = output_tensor2.has_value() ? output_tensor2.value().buffer()->address() : 0;
        tt::tt_metal::SetRuntimeArgs(
            program,
            unary_writer_kernel_id,
            core,
            {dst_buffer1_address, dst_buffer2_address, num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.unary_reader_kernel_id = unary_reader_kernel_id, .unary_writer_kernel_id = unary_writer_kernel_id}};
}

void ExampleMultipleReturnDeviceOperation::SingleCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input_tensor = tensor_args.input_tensor;
    auto output_tensor1 = tensor_return_value.at(0);
    auto output_tensor2 = tensor_return_value.at(1);

    auto src_buffer = input_tensor.buffer();
    auto dst_buffer1 = output_tensor1.has_value() ? output_tensor1.value().buffer() : 0;
    auto dst_buffer2 = output_tensor2.has_value() ? output_tensor2.value().buffer() : 0;

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_reader_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = src_buffer->address();
    }

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, CoreCoord{0, 0});
        if (output_tensor1.has_value()) {
            runtime_args[0] = dst_buffer1->address();
        }
        if (output_tensor2.has_value()) {
            runtime_args[1] = dst_buffer2->address();
        }
    }
}

}  // namespace ttnn::operations::examples
