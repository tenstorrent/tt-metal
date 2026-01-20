// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_program_factory.hpp"
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::tt_metal;

namespace ttnn::prim {

IndexedFillProgramFactory::cached_program_t IndexedFillProgramFactory::create(
    const IndexedFillParams& /*operation_attributes*/, const IndexedFillInputs& tensor_args, Tensor& output) {
    const auto& batch_ids = tensor_args.batch_id;
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;

    tt::tt_metal::Program program{};
    IDevice* device = input_a.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto set_of_core_ranges =
        tt::tt_metal::num_cores_to_corerangeset(num_cores_x * num_cores_y, compute_with_storage_grid_size);
    CoreRangeSet all_cores(set_of_core_ranges);

    uint32_t B = input_a.padded_shape()[0];
    uint32_t b = input_b.padded_shape()[0];

    TT_ASSERT(batch_ids.padded_shape()[-1] == b);

    // parallelize across batch
    uint32_t cb_index = 0;
    uint32_t batch_cb_index = 1;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_a.dtype());

    uint32_t page_size = input_a.padded_shape()[-1] * input_a.element_size();
    uint32_t rounded_page_size = round_up_to_mul32(page_size);
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(2 * rounded_page_size, {{cb_index, cb_data_format}})
            .set_page_size(cb_index, rounded_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t batch_page_size = round_up_to_mul32(b * sizeof(uint32_t));
    tt::tt_metal::CircularBufferConfig batch_cb_config =
        tt::tt_metal::CircularBufferConfig(2 * batch_page_size, {{batch_cb_index, cb_data_format}})
            .set_page_size(batch_cb_index, batch_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, batch_cb_config);

    // Create Kernels
    // reader
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)cb_index, (std::uint32_t)batch_cb_index, page_size};
    TensorAccessorArgs(*input_a.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*input_b.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(*batch_ids.buffer()).append_to(reader_compile_time_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/indexed_fill/device/kernels/dataflow/indexed_fill_reader.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)cb_index, (std::uint32_t)page_size};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    auto cores = grid_to_cores(num_cores_x * num_cores_y, num_cores_x, num_cores_y, false);

    uint32_t batch_size_in_sticks = input_a.padded_shape()[1] * input_a.padded_shape()[2];

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        uint32_t local_b = (i < B) ? b : 0;
        uint32_t local_batch_size_in_sticks = (i < B) ? batch_size_in_sticks : 0;

        const std::array reader_runtime_args = {
            batch_ids.buffer()->address(),
            local_b,
            input_a.buffer()->address(),
            input_b.buffer()->address(),
            local_batch_size_in_sticks,
            i};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        const std::array writer_runtime_args = {
            output.buffer()->address(), page_size, local_batch_size_in_sticks, i * local_batch_size_in_sticks};
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
    }

    return cached_program_t{
        std::move(program),
        IndexedFillSharedVariables{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .cores = std::move(cores),
            .page_size = page_size,
        }};
}

void IndexedFillProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const IndexedFillParams& /*operation_attributes*/,
    const IndexedFillInputs& tensor_args,
    Tensor& output) {
    const auto& batch_ids = tensor_args.batch_id;
    const auto& input_a = tensor_args.input_tensor_a;
    const auto& input_b = tensor_args.input_tensor_b;

    auto& program = cached_program.program;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& page_size = cached_program.shared_variables.page_size;

    uint32_t B = input_a.padded_shape()[0];
    uint32_t b = input_b.padded_shape()[0];
    uint32_t batch_size_in_sticks = input_a.padded_shape()[1] * input_a.padded_shape()[2];

    uint32_t core_id = 0;
    for (const auto& core : cores) {
        uint32_t local_b = (core_id < B) ? b : 0;
        uint32_t local_batch_size_in_sticks = (core_id < B) ? batch_size_in_sticks : 0;
        const std::array reader_runtime_args = {
            batch_ids.buffer()->address(),
            local_b,
            input_a.buffer()->address(),
            input_b.buffer()->address(),
            local_batch_size_in_sticks,
            core_id};
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);

        const std::array writer_runtime_args = {
            output.buffer()->address(), page_size, local_batch_size_in_sticks, core_id * local_batch_size_in_sticks};

        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        core_id++;
    }
}

}  // namespace ttnn::prim
