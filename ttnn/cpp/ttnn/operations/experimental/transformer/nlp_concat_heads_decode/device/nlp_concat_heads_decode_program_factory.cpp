// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "nlp_concat_heads_decode_program_factory.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::nlp_concat_heads_decode::program {

using namespace tt;
using namespace tt::constants;

NLPConcatHeadsDecodeProgramFactory::cached_program_t NLPConcatHeadsDecodeProgramFactory::create(
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    tt_metal::Program program = tt_metal::CreateProgram();

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t head_dim = input_shape[-1];
    const uint32_t batch = input_shape[1];

    tt_metal::IDevice* device = input_tensor.device();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output.shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;
    auto in_shard_spec = input_tensor.shard_spec().value();
    auto in_cores = in_shard_spec.grid;

    uint32_t q_output_cb_index = CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_q_output_config =
        tt_metal::CircularBufferConfig(q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_q_output = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    uint32_t q_base_addr = input_tensor.buffer()->address();

    // cores to read and write to output
    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    // cores for input
    auto in_core_grid = in_cores.bounding_box();
    uint32_t in_num_cores_x = in_core_grid.end_coord.x + 1, in_num_cores_y = in_core_grid.end_coord.y + 1;

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores_x);
    for (uint32_t x = 0; x < in_num_cores_x; ++x) {
        noc_x_coords.push_back(device->worker_core_from_logical_core({x, 0}).x);
    }
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores_y);
    for (uint32_t y = 0; y < in_num_cores_y; ++y) {
        noc_y_coords.push_back(device->worker_core_from_logical_core({0, y}).y);
    }

    // We parallize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2 of a
    // tile respectively)
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        q_output_cb_index,
        head_size,
        batch,
        head_tiles,
        1,  // read the first phase
        in_num_cores_x,
        in_num_cores_y};
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode.cpp",
        q_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    reader_compile_time_args[6] = 2;  // read the second phase
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode.cpp",
        q_cores,
        tt_metal::WriterDataMovementConfig(reader_compile_time_args));

    uint32_t q_start_addr = q_base_addr;

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t in_tile_offset_by_batch =
            i < 16 ? i * sub_tile_line_bytes : ((i - 16) * sub_tile_line_bytes) + (512 * element_size);

        const auto& core = cores[i];
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(2 + in_num_cores_x + in_num_cores_y);
        reader_runtime_args = {
            in_tile_offset_by_batch,
            q_start_addr,
        };
        reader_runtime_args.insert(reader_runtime_args.end(), noc_x_coords.begin(), noc_x_coords.end());
        reader_runtime_args.insert(reader_runtime_args.end(), noc_y_coords.begin(), noc_y_coords.end());

        tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, reader_runtime_args);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .cores = cores,
            .element_size = element_size,
            .sub_tile_line_bytes = sub_tile_line_bytes,
            .num_cores = num_cores,
            .cb_q_output = cb_q_output}};
}

void NLPConcatHeadsDecodeProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto *dst_buffer_query = output.buffer();
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_q_output, *dst_buffer_query);

    uint32_t q_base_addr = input_tensor.buffer()->address();
    uint32_t q_start_addr = q_base_addr;

    for (uint32_t i = 0; i < shared_variables.num_cores; ++i) {
        uint32_t in_tile_offset_by_batch =
            i < 16 ? i * shared_variables.sub_tile_line_bytes
                   : ((i - 16) * shared_variables.sub_tile_line_bytes) + (512 * shared_variables.element_size);
        const auto& core = shared_variables.cores[i];
        auto& runtime_args = GetRuntimeArgs(program, shared_variables.reader_kernel_id, core);
        runtime_args[0] = in_tile_offset_by_batch;
        runtime_args[1] = q_start_addr;

        auto& runtime_args_writer = GetRuntimeArgs(program, shared_variables.writer_kernel_id, core);
        runtime_args_writer[0] = in_tile_offset_by_batch;
        runtime_args_writer[1] = q_start_addr;
    }
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads_decode::program
