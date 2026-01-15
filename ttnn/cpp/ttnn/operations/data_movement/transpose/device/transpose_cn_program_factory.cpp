// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_cn_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::transpose::program {

TransposeCNProgramFactory::cached_program_t TransposeCNProgramFactory::create(
    const transpose::TransposeParams& /*operation_attributes*/,
    const transpose::TransposeInputs& tensor_args,
    transpose::tensor_return_value_t& tensor_return_value) {
    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = tensor_return_value;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_cn needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_cn needs to be allocated in a buffer on device!");

    Program program = Program();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();
    IDevice* device = input_tensor.device();

    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;
    CoreRange total_cores({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = 2;
    CircularBufferConfig cb_src0_config =
        CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    CreateCircularBuffer(program, all_cores, cb_src0_config);

    std::vector<uint32_t> reader_compile_time_args = {static_cast<uint32_t>(src0_cb_index)};
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(src0_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelHandle reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_cn_interleaved_start_id.cpp",
        total_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    KernelHandle writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        total_cores,
        WriterDataMovementConfig(writer_compile_time_args));

    // Set runtime arguments for each core
    auto input_shape = input_tensor.padded_shape();
    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_tiles;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores_total; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        }

        uint32_t hw = num_tiles_read % HtWt;
        uint32_t curr_c = num_tiles_read / HtWt;
        uint32_t n = curr_c % N;
        uint32_t start_tile = num_tiles_read + (curr_c * batch_step) - (curr_c / N * channel_step);

        SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {src0_buffer->address(), N, C, HtWt, batch_step, channel_step, num_tiles_per_core, start_tile, hw, n});

        SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_read});

        num_tiles_read += num_tiles_per_core;
    }

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .core_group_1 = core_group_1,
         .core_group_2 = core_group_2,
         .num_cores_total = num_cores_total,
         .num_cores_y = num_cores_y,
         .num_tiles_per_core_group_1 = num_tiles_per_core_group_1,
         .num_tiles_per_core_group_2 = num_tiles_per_core_group_2}};
}

void TransposeCNProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const transpose::TransposeParams& /*operation_attributes*/,
    const transpose::TransposeInputs& tensor_args,
    transpose::tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    const auto& input_tensor = tensor_args.input;
    auto& output_tensor = tensor_return_value;

    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output_tensor.buffer();
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2], C = input_shape[1], N = input_shape[0];
    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;
    uint32_t num_tensor_tiles = N * C * H * W / TILE_HW;
    uint32_t HtWt = Ht * Wt;
    uint32_t CHtWt = C * HtWt;
    uint32_t NCHtWt = num_tensor_tiles;
    uint32_t batch_step = CHtWt - HtWt;
    uint32_t channel_step = NCHtWt - HtWt;

    auto& cached_reader_args = GetRuntimeArgs(program, shared_variables.reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, shared_variables.writer_kernel_id);

    for (uint32_t i = 0, num_tiles_read = 0; i < shared_variables.num_cores_total; i++) {
        CoreCoord core = {i / shared_variables.num_cores_y, i % shared_variables.num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (shared_variables.core_group_1.contains(core)) {
            num_tiles_per_core = shared_variables.num_tiles_per_core_group_1;
        } else if (shared_variables.core_group_2.contains(core)) {
            num_tiles_per_core = shared_variables.num_tiles_per_core_group_2;
        }

        uint32_t hw = num_tiles_read % HtWt;
        uint32_t curr_c = num_tiles_read / HtWt;
        uint32_t n = curr_c % N;
        uint32_t start_tile = num_tiles_read + (curr_c * batch_step) - (curr_c / N * channel_step);

        auto& reader_args = cached_reader_args.at(core.x).at(core.y);
        auto& writer_args = cached_writer_args.at(core.x).at(core.y);

        reader_args[0] = input_buffer->address();
        reader_args[1] = N;
        reader_args[2] = C;
        reader_args[3] = HtWt;
        reader_args[4] = batch_step;
        reader_args[5] = channel_step;
        reader_args[6] = num_tiles_per_core;
        reader_args[7] = start_tile;
        reader_args[8] = hw;
        reader_args[9] = n;

        writer_args[0] = output_buffer->address();
        writer_args[1] = num_tiles_per_core;
        writer_args[2] = num_tiles_read;

        num_tiles_read += num_tiles_per_core;
    }
}

}  // namespace ttnn::operations::data_movement::transpose::program
