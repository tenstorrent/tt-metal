// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "gelu_backward_program_factory.hpp"
#include "gelu_backward_device_operation_types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::gelu_backward::program {

using namespace tt::constants;

GeluBackwardProgramFactory::cached_program_t GeluBackwardProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;              // src0
    const auto& grad_output = tensor_args.grad_output;  // src1

    tt::tt_metal::Program program{};

    tt::DataFormat src0_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat src1_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(grad_output.dtype());
    uint32_t src1_single_tile_size = tt::tile_size(src1_cb_data_format);
    tt::DataFormat dst_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    // NOTE: There is an assumption that number of tiles in grad_output is the same as in input
    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t num_input_tiles = 2;
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * src0_single_tile_size, {{src0_cb_index, src0_cb_data_format}})
            .set_page_size(src0_cb_index, src0_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * src1_single_tile_size, {{src1_cb_index, src1_cb_data_format}})
            .set_page_size(src1_cb_index, src1_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    uint32_t num_output_tiles = 2;
    uint32_t output_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * dst_single_tile_size, {{output_cb_index, dst_cb_data_format}})
            .set_page_size(output_cb_index, dst_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto* src0_buffer = grad_output.buffer();
    auto* src1_buffer = input.buffer();

    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {0};
    tt::tt_metal::TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*src1_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle binary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    bool fp32_dest_acc_en = (dst_cb_data_format == tt::DataFormat::Float32) ||
                            (dst_cb_data_format == tt::DataFormat::Int32) ||
                            (dst_cb_data_format == tt::DataFormat::UInt32);

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    unpack_to_dest_mode[src1_cb_index] = UnpackToDestMode::UnpackToDestFp32;

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        args.approximate == "tanh" ? "ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/"
                                     "kernels/compute/eltwise_bw_gelu_approx_tanh.cpp"
                                   : "ttnn/cpp/ttnn/operations/experimental/unary_backward/gelu_backward/device/"
                                     "kernels/compute/eltwise_bw_gelu_approx_none.cpp",
        all_cores,
        tt::tt_metal::ComputeConfig{.fp32_dest_acc_en = fp32_dest_acc_en, .unpack_to_dest_mode = unpack_to_dest_mode});

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
            binary_reader_kernel_id,
            core,
            {src0_buffer->address(), src1_buffer->address(), num_tiles_per_core, num_tiles_written, 0, 0, num_cores_y});

        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {num_tiles_per_core, 1});

        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    return cached_program_t{
        std::move(program),
        {binary_reader_kernel_id, compute_kernel_id, unary_writer_kernel_id, num_cores, num_cores_y}};
}

void GeluBackwardProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& output) {
    using namespace tt::tt_metal;

    auto& shared_vars = cached_program.shared_variables;
    auto& gelu_bw_reader_kernel_id = shared_vars.gelu_bw_reader_kernel_id;
    auto& gelu_bw_compute_kernel_id = shared_vars.gelu_bw_compute_kernel_id;
    auto& gelu_bw_writer_kernel_id = shared_vars.gelu_bw_writer_kernel_id;
    auto& program = cached_program.program;

    uint32_t num_cores = shared_vars.num_cores;
    uint32_t num_cores_y = shared_vars.num_cores_y;

    const auto& input = tensor_args.input;
    const auto& grad_output = tensor_args.grad_output;
    auto* src0_buffer = grad_output.buffer();
    auto* src1_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    // Only update buffer addresses
    auto& reader_runtime_args = GetRuntimeArgs(program, gelu_bw_reader_kernel_id);
    auto& compute_runtime_args = GetRuntimeArgs(program, gelu_bw_compute_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, gelu_bw_writer_kernel_id);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [_, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

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

        // Update reader args
        reader_runtime_args[core.x][core.y][0] = src0_buffer->address();
        reader_runtime_args[core.x][core.y][1] = src1_buffer->address();
        reader_runtime_args[core.x][core.y][2] = num_tiles_per_core;
        reader_runtime_args[core.x][core.y][3] = num_tiles_written;

        // Update compute args
        compute_runtime_args[core.x][core.y][0] = num_tiles_per_core;

        // Update writer args
        writer_runtime_args[core.x][core.y][0] = dst_buffer->address();
        writer_runtime_args[core.x][core.y][1] = num_tiles_per_core;
        writer_runtime_args[core.x][core.y][2] = num_tiles_written;

        num_tiles_written += num_tiles_per_core;
    }
}

}  // namespace ttnn::operations::experimental::gelu_backward::program
