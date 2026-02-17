// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_rm_chunked_program_factory.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/span.hpp>

namespace ttnn::prim {

using namespace tt::constants;

namespace {
struct ChunkSizeConfig {
    uint32_t input_full_chunk_size_bytes;
    uint32_t output_full_chunk_size_bytes;
    uint32_t input_partial_chunk_size_bytes;
    uint32_t output_partial_chunk_size_bytes;
    uint32_t full_chunks_per_row;
    uint32_t partial_chunks_per_row;
};

ChunkSizeConfig calculate_chunk_config(
    uint32_t row_width_elements, uint32_t input_element_size, uint32_t output_element_size) {
    constexpr uint32_t max_elements_per_chunk = 1024;
    const uint32_t elements_per_full_chunk = std::min(max_elements_per_chunk, row_width_elements);

    // Calculate chunk sizes in bytes (logical size, not including padding)
    const uint32_t input_full_chunk_size_bytes = elements_per_full_chunk * input_element_size;
    const uint32_t output_full_chunk_size_bytes = elements_per_full_chunk * output_element_size;

    // Calculate how many chunks per row
    const uint32_t full_chunks_per_row = row_width_elements / elements_per_full_chunk;
    const uint32_t remainder = row_width_elements % elements_per_full_chunk;
    const uint32_t partial_chunks_per_row = (remainder > 0) ? 1 : 0;
    const uint32_t input_partial_chunk_size_bytes = remainder * input_element_size;
    const uint32_t output_partial_chunk_size_bytes = remainder * output_element_size;

    return ChunkSizeConfig{
        .input_full_chunk_size_bytes = input_full_chunk_size_bytes,
        .output_full_chunk_size_bytes = output_full_chunk_size_bytes,
        .input_partial_chunk_size_bytes = input_partial_chunk_size_bytes,
        .output_partial_chunk_size_bytes = output_partial_chunk_size_bytes,
        .full_chunks_per_row = full_chunks_per_row,
        .partial_chunks_per_row = partial_chunks_per_row,
    };
}

}  // anonymous namespace

TypecastRowMajorChunkedProgramFactory::cached_program_t TypecastRowMajorChunkedProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const DataType& input_dtype = args.input_dtype;
    const DataType& output_dtype = args.output_dtype;

    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "This factory is only for ROW_MAJOR layout");

    tt::tt_metal::Program program{};

    const tt::DataFormat cb_data_format_input = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_element_size = tt::datum_size(cb_data_format_input);
    const tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_element_size = tt::datum_size(cb_data_format_output);

    const auto* device = input.device();

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Get row information
    const auto& padded_shape = input.padded_shape();
    const uint32_t row_width_elements = padded_shape[padded_shape.rank() - 1];
    const uint32_t num_rows = src_buffer->num_pages();

    // Calculate chunk configuration
    const ChunkSizeConfig chunk_config =
        calculate_chunk_config(row_width_elements, input_element_size, output_element_size);

    const uint32_t input_full_chunk_size_bytes = chunk_config.input_full_chunk_size_bytes;
    const uint32_t output_full_chunk_size_bytes = chunk_config.output_full_chunk_size_bytes;
    const uint32_t input_partial_chunk_size_bytes = chunk_config.input_partial_chunk_size_bytes;
    const uint32_t output_partial_chunk_size_bytes = chunk_config.output_partial_chunk_size_bytes;
    const uint32_t full_chunks_per_row = chunk_config.full_chunks_per_row;
    const uint32_t partial_chunks_per_row = chunk_config.partial_chunks_per_row;

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Split work by rows (each core handles complete rows with both full and partial chunks)
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, true);

    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t output_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t num_input_pages = 2;   // Always use double buffering
    constexpr uint32_t num_output_pages = 2;  // Always use double buffering

    tt::tt_metal::CircularBufferConfig cb_input_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_pages * input_full_chunk_size_bytes, {{input_cb_index, cb_data_format_input}})
            .set_page_size(input_cb_index, input_full_chunk_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_input_config);

    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_pages * output_full_chunk_size_bytes, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, output_full_chunk_size_bytes);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    // Create compile-time args for unified kernels (handle both full and partial chunks)
    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,                  // cb_id_in
        input_full_chunk_size_bytes,     // full_chunk_size_bytes
        full_chunks_per_row,             // full_chunks_per_row
        input_partial_chunk_size_bytes,  // partial_chunk_size_bytes
        partial_chunks_per_row,          // partial_chunks_per_row (0 or 1)
        src_buffer->page_size()          // row_page_size_bytes
    };
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,                  // cb_id_out
        output_full_chunk_size_bytes,     // full_chunk_size_bytes
        full_chunks_per_row,              // full_chunks_per_row
        output_partial_chunk_size_bytes,  // partial_chunk_size_bytes
        partial_chunks_per_row,           // partial_chunks_per_row (0 or 1)
        dst_buffer->page_size()           // row_page_size_bytes
    };
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle typecast_reader_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle typecast_writer_kernel = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Create compute kernels - compute per_core_block_cnt as total chunks (full + partial) per core
    const uint32_t chunks_per_row_total = full_chunks_per_row + partial_chunks_per_row;

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_rows_per_core_group_1 * chunks_per_row_total,  // per_core_block_cnt (rows * total_chunks_per_row)
        1,
        input_cb_index,
        output_cb_index};

    std::vector<uint32_t> compute_kernel_args_group_2 = {
        num_rows_per_core_group_2 * chunks_per_row_total,  // per_core_block_cnt (rows * total_chunks_per_row)
        1,
        input_cb_index,
        output_cb_index};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    constexpr bool math_approx_mode = false;

    std::map<std::string, std::string> unary_defines;
    unary_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    unary_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const char* const path = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

    if (!core_group_1.ranges().empty()) {
        tt::tt_metal::CreateKernel(
            program,
            path,
            core_group_1,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_1,
                .defines = unary_defines});
    }

    if (!core_group_2.ranges().empty()) {
        tt::tt_metal::CreateKernel(
            program,
            path,
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = unary_defines});
    }

    // Assign runtime args to cores (distributing rows)
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, true);
    uint32_t row_idx = 0;

    for (const auto& core : cores_vec) {
        bool is_group_1 = core_group_1.contains(core);
        uint32_t num_rows_for_core = is_group_1 ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        uint32_t start_row_id = row_idx;

        tt::tt_metal::SetRuntimeArgs(
            program, typecast_reader_kernel, core, {src_buffer->address(), num_rows_for_core, start_row_id});

        tt::tt_metal::SetRuntimeArgs(
            program, typecast_writer_kernel, core, {dst_buffer->address(), num_rows_for_core, start_row_id});

        row_idx += num_rows_for_core;
    }

    return cached_program_t{
        std::move(program),
        {typecast_reader_kernel,
         typecast_writer_kernel,
         num_cores,
         full_chunks_per_row,
         input_full_chunk_size_bytes,
         output_full_chunk_size_bytes}};
}

void TypecastRowMajorChunkedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams& /*operation_attributes*/,
    const TypecastInputs& tensor_args,
    Tensor& output) {
    const tt::tt_metal::KernelHandle typecast_reader_kernel_id =
        cached_program.shared_variables.typecast_reader_kernel_id;
    const tt::tt_metal::KernelHandle typecast_writer_kernel_id =
        cached_program.shared_variables.typecast_writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;

    auto& program = cached_program.program;

    const Tensor& input = tensor_args.input;
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Recalculate row distribution (in case tensor changed)
    const uint32_t num_rows = src_buffer->num_pages();
    const uint32_t num_rows_per_core = (num_rows + num_cores - 1) / num_cores;

    // Reconstruct all_cores CoreRangeSet to use corerange_to_cores for consistency
    const CoreCoord compute_with_storage_grid_size = input.device()->compute_with_storage_grid_size();
    const CoreRangeSet all_cores =
        tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, true);

    uint32_t num_rows_written = 0;
    for (const CoreCoord& core : cores_vec) {
        uint32_t rows_for_this_core = std::min(num_rows_per_core, num_rows - num_rows_written);

        {
            auto& runtime_args = GetRuntimeArgs(program, typecast_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
            runtime_args[1] = rows_for_this_core;
            runtime_args[2] = num_rows_written;
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, typecast_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
            runtime_args[1] = rows_for_this_core;
            runtime_args[2] = num_rows_written;
        }

        num_rows_written += rows_for_this_core;
    }
}

}  // namespace ttnn::prim
