// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include "nlp_concat_heads_decode_subcoregrids_program_factory.hpp"
#include "nlp_concat_heads_decode_device_operation.hpp"
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::nlp_concat_heads_decode::program {

using namespace tt;
using namespace tt::constants;

NLPConcatHeadsDecodeSubcoregridsProgramFactory::cached_program_t NLPConcatHeadsDecodeSubcoregridsProgramFactory::create(
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

    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    auto tile_h = tile_shape[0];
    auto tile_w = tile_shape[1];
    auto tile_hw = tile_h * tile_w;

    auto face_shape = input_tensor.tensor_spec().tile().get_face_shape();
    auto face_h = face_shape[0];
    auto face_w = face_shape[1];
    auto face_hw = face_h * face_w;

    const uint32_t head_tiles = head_dim / tile_w;
    const uint32_t head_size = head_tiles * single_tile_size;

    const uint32_t element_size = input_tensor.element_size();
    const uint32_t sub_tile_line_bytes = face_w * element_size;
    const auto q_shard_spec = output.shard_spec().value();
    const auto q_cores = q_shard_spec.grid;
    const auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / tile_hw;
    const auto in_shard_spec = input_tensor.shard_spec().value();
    const auto in_cores = in_shard_spec.grid;

    uint32_t q_output_cb_index = CBIndex::c_16;
    tt_metal::CircularBufferConfig cb_q_output_config =
        tt_metal::CircularBufferConfig(q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output.buffer());
    auto cb_q_output = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);

    uint32_t q_start_addr = input_tensor.buffer()->address();

    // cores to read and write to output
    const uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    const auto& cores = corerange_to_cores(q_cores, num_cores, true);

    // cores for input
    const uint32_t in_num_cores = in_cores.num_cores();  // number of cores of the input
    const auto& in_cores_vec = corerange_to_cores(in_cores, in_num_cores, true);

    std::vector<uint32_t> noc_x_coords;
    noc_x_coords.reserve(in_num_cores);
    std::vector<uint32_t> noc_y_coords;
    noc_y_coords.reserve(in_num_cores);
    for (uint32_t i = 0; i < in_num_cores; ++i) {
        noc_x_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).x);
        noc_y_coords.push_back(device->worker_core_from_logical_core(in_cores_vec[i]).y);
    }

    // We parallize the reader on risc0 and risc1 as two phases, where each risc reads half-tile of the input (Phase 1
    // reads left half-tile and Phase 2 reads right half-tile respectively)
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)element_size,
        (std::uint32_t)sub_tile_line_bytes,
        q_output_cb_index,
        head_size,
        batch,
        head_tiles,
        1,  // read the first phase
        in_num_cores,
        face_h,
        face_hw};
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp",
        q_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args));
    reader_compile_time_args[6] = 2;  // read the second phase
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_concat_heads_decode/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_concat_heads_decode_subcoregrid.cpp",
        q_cores,
        tt_metal::WriterDataMovementConfig(reader_compile_time_args));

    for (uint32_t i = 0; i < num_cores; ++i) {
        // in_tile_offset_by_batch is the start address of each batch in the input tile. The first face_h batches are in
        // the upper half of the tile and rest are in the lower half of tile.
        uint32_t in_tile_offset_by_batch = i < face_h ? i * sub_tile_line_bytes : (i + face_h) * sub_tile_line_bytes;

        const auto& core = cores[i];
        std::vector<uint32_t> reader_runtime_args;
        reader_runtime_args.reserve(2 + (2 * in_num_cores));
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
        {/* reader_kernel_id     = */ reader_kernel_id,
         /* writer_kernel_id     = */ writer_kernel_id,
         /* cores                = */ cores,
         /* element_size         = */ element_size,
         /* sub_tile_line_bytes  = */ sub_tile_line_bytes,
         /* num_cores            = */ num_cores,
         /* cb_q_output          = */ cb_q_output,
         /* face_h               = */ face_h,
         /* tile_w               = */ tile_w}};
}

void NLPConcatHeadsDecodeSubcoregridsProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const NlpConcatHeadsDecodeParams& /*operation_attributes*/,
    const NlpConcatHeadsDecodeInputs& tensor_args,
    tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.input;
    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto *dst_buffer_query = output.buffer();
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_q_output, *dst_buffer_query);

    uint32_t q_start_addr = input_tensor.buffer()->address();

    auto& reader_args_by_core = GetRuntimeArgs(program, shared_variables.reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, shared_variables.writer_kernel_id);

    for (uint32_t i = 0; i < shared_variables.num_cores; ++i) {
        uint32_t in_tile_offset_by_batch = i < shared_variables.face_h
                                               ? i * shared_variables.sub_tile_line_bytes
                                               : (i + shared_variables.face_h) * shared_variables.sub_tile_line_bytes;
        const auto& core = shared_variables.cores[i];
        auto& runtime_args_reader = reader_args_by_core[core.x][core.y];
        runtime_args_reader[0] = in_tile_offset_by_batch;
        runtime_args_reader[1] = q_start_addr;

        auto& runtime_args_writer = writer_args_by_core[core.x][core.y];
        runtime_args_writer[0] = in_tile_offset_by_batch;
        runtime_args_writer[1] = q_start_addr;
    }
}

}  // namespace ttnn::operations::experimental::nlp_concat_heads_decode::program
