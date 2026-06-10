// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "topk_xl_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::topk_xl::program {

namespace {

uint32_t flattened_rows_excluding_last_dim(const ttnn::Shape& shape) {
    uint32_t rows = 1;
    for (uint32_t i = 0; i + 1 < shape.rank(); ++i) {
        rows *= shape[i];
    }
    return rows;
}

}  // namespace

TopkXLProgramFactory::cached_program_t TopkXLProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto program = tt::tt_metal::CreateProgram();

    const auto& input = tensor_args.input_tensor;
    auto& values = std::get<0>(tensor_return_value);
    auto& indices = std::get<1>(tensor_return_value);

    const auto shape = input.logical_shape();
    const uint32_t k = operation_attributes.k;
    const uint32_t n = shape[shape.rank() - 1];
    const uint32_t num_rows = flattened_rows_excluding_last_dim(shape);
    const uint32_t num_chunks = n / k;
    const uint32_t tiles_per_sequence = (k + tt::constants::TILE_HW - 1) / tt::constants::TILE_HW;

    const auto core = CoreCoord{0, 0};
    const auto core_range = tt::tt_metal::CoreRangeSet(std::vector{tt::tt_metal::CoreRange(core, core)});

    constexpr uint32_t cb_in = tt::CBIndex::c_0;
    constexpr uint32_t cb_values = tt::CBIndex::c_1;
    constexpr uint32_t cb_indices = tt::CBIndex::c_2;

    const uint32_t input_chunk_bytes = k * input.element_size();
    const uint32_t input_row_bytes = n * input.element_size();
    const uint32_t input_tile_bytes = tt::constants::TILE_HW * input.element_size();
    constexpr uint32_t row_slice_elements = tt::constants::FACE_WIDTH;
    const uint32_t output_slices_per_row = k / row_slice_elements;
    const uint32_t values_slice_bytes = row_slice_elements * values.element_size();
    const uint32_t indices_slice_bytes = row_slice_elements * indices.element_size();
    const uint32_t values_row_bytes = k * values.element_size();
    const uint32_t indices_row_bytes = k * indices.element_size();
    const uint32_t values_tile_bytes = tt::constants::TILE_HW * values.element_size();
    const uint32_t indices_tile_bytes = tt::constants::TILE_HW * indices.element_size();

    const uint32_t cb_depth = 2;
    const auto input_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * input_tile_bytes, {{cb_in, tt::DataFormat::Float16_b}})
            .set_page_size(cb_in, input_tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range, input_cb_config);

    const auto values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * values_tile_bytes, {{cb_values, tt::DataFormat::Float16_b}})
            .set_page_size(cb_values, values_tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range, values_cb_config);

    const auto indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_depth * tiles_per_sequence * indices_tile_bytes, {{cb_indices, tt::DataFormat::Float32}})
            .set_page_size(cb_indices, indices_tile_bytes);
    tt::tt_metal::CreateCircularBuffer(program, core_range, indices_cb_config);

    std::vector<uint32_t> reader_compile_args = {
        cb_in, num_rows, num_chunks, input_chunk_bytes, input_row_bytes, input_tile_bytes, tiles_per_sequence};
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(reader_compile_args);

    auto reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_xl/device/kernels/reader.cpp",
        core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
            .noc = tt::tt_metal::NOC::NOC_0,
            .compile_args = reader_compile_args});

    std::vector<uint32_t> compute_compile_args = {cb_in, cb_values, cb_indices, num_rows, num_chunks, k};
    auto compute_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_xl/device/kernels/compute.cpp",
        core_range,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi2,
            .fp32_dest_acc_en = true,
            .dst_full_sync_en = true,
            .bfp8_pack_precise = false,
            .math_approx_mode = false,
            .compile_args = compute_compile_args});

    std::vector<uint32_t> writer_compile_args = {
        cb_values,
        cb_indices,
        num_rows,
        output_slices_per_row,
        values_slice_bytes,
        indices_slice_bytes,
        values_row_bytes,
        indices_row_bytes,
        tiles_per_sequence};
    tt::tt_metal::TensorAccessorArgs(*values.buffer()).append_to(writer_compile_args);

    auto writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/topk_xl/device/kernels/writer.cpp",
        core_range,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0,
            .noc = tt::tt_metal::NOC::NOC_1,
            .compile_args = writer_compile_args});

    tt::tt_metal::SetRuntimeArgs(program, reader_kernel, core, {input.buffer()->address()});
    tt::tt_metal::SetRuntimeArgs(program, compute_kernel, core, {});
    tt::tt_metal::SetRuntimeArgs(
        program, writer_kernel, core, {values.buffer()->address(), indices.buffer()->address()});

    return cached_program_t{
        std::move(program),
        TopkXLSharedVariables{
            .reader_kernel_id = reader_kernel,
            .compute_kernel_id = compute_kernel,
            .writer_kernel_id = writer_kernel,
            .core = core}};
}

void TopkXLProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t&,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared = cached_program.shared_variables;
    auto& reader_args = tt::tt_metal::GetRuntimeArgs(program, shared.reader_kernel_id, shared.core);
    reader_args[0] = tensor_args.input_tensor.buffer()->address();

    auto& writer_args = tt::tt_metal::GetRuntimeArgs(program, shared.writer_kernel_id, shared.core);
    writer_args[0] = std::get<0>(tensor_return_value).buffer()->address();
    writer_args[1] = std::get<1>(tensor_return_value).buffer()->address();
}

}  // namespace ttnn::operations::experimental::topk_xl::program
