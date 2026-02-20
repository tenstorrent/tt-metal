// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_decode_interleaved_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt;

namespace ttnn::experimental::prim {

NLPCreateQKVHeadsDecodeInterleavedProgramFactory::cached_program_t
NLPCreateQKVHeadsDecodeInterleavedProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt::tt_metal;

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& num_q_heads = operation_attributes.num_q_heads;
    const auto& num_kv_heads = operation_attributes.num_kv_heads;
    const auto& head_dim = operation_attributes.head_dim;

    Program program = CreateProgram();

    tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    uint32_t head_tiles = head_dim / TILE_WIDTH;
    uint32_t head_size = head_tiles * single_tile_size;

    uint32_t element_size = input_tensor.element_size();
    uint32_t sub_tile_line_bytes = 16 * element_size;
    auto q_shard_spec = output[0].shard_spec().value();
    auto q_cores = q_shard_spec.grid;
    auto q_num_tiles = q_shard_spec.shape[0] * q_shard_spec.shape[1] / TILE_HW;

    uint32_t q_output_cb_index = CBIndex::c_16;
    CircularBufferConfig cb_q_output_config =
        CircularBufferConfig(q_num_tiles * single_tile_size, {{q_output_cb_index, cb_data_format}})
            .set_page_size(q_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[0].buffer());
    auto cb_q_output = CreateCircularBuffer(program, q_cores, cb_q_output_config);

    auto k_shard_spec = output[1].shard_spec().value();
    auto k_cores = k_shard_spec.grid;
    auto k_num_tiles = k_shard_spec.shape[0] * k_shard_spec.shape[1] / TILE_HW;

    uint32_t k_output_cb_index = CBIndex::c_17;
    CircularBufferConfig cb_k_output_config =
        CircularBufferConfig(k_num_tiles * single_tile_size, {{k_output_cb_index, cb_data_format}})
            .set_page_size(k_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[1].buffer());
    auto cb_k_output = CreateCircularBuffer(program, k_cores, cb_k_output_config);

    auto v_shard_spec = output[2].shard_spec().value();
    auto v_cores = q_shard_spec.grid;
    auto v_num_tiles = v_shard_spec.shape[0] * v_shard_spec.shape[1] / TILE_HW;

    uint32_t v_output_cb_index = CBIndex::c_18;
    CircularBufferConfig cb_v_output_config =
        CircularBufferConfig(v_num_tiles * single_tile_size, {{v_output_cb_index, cb_data_format}})
            .set_page_size(v_output_cb_index, single_tile_size)
            .set_globally_allocated_address(*output[2].buffer());
    auto cb_v_output = CreateCircularBuffer(program, v_cores, cb_v_output_config);

    uint32_t q_base_addr = input_tensor.buffer()->address();

    // We parallelize the reader on risc0 and risc1, where each risc reads a sub-tile of the input (phase1 and phase2
    // of a tile respectively)
    std::vector<uint32_t> reader_compile_time_args = {
        element_size,
        sub_tile_line_bytes,
        q_output_cb_index,
        k_output_cb_index,
        v_output_cb_index,
        head_size,
        num_q_heads,
        num_kv_heads,
        head_tiles,
        1,  // read the first phase
    };
    tt::tt_metal::TensorAccessorArgs(input_tensor.buffer()).append_to(reader_compile_time_args);
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
        q_cores,
        ReaderDataMovementConfig(reader_compile_time_args));
    reader_compile_time_args[9] = 2;  // read the second phase
    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_decode/device/kernels/"
        "reader_interleaved_tm_tile_layout_nlp_create_qkv_heads_decode.cpp",
        q_cores,
        WriterDataMovementConfig(reader_compile_time_args));

    uint32_t num_cores = q_cores.num_cores();  // number of cores of the output
    auto core_grid = q_cores.bounding_box();
    uint32_t num_cores_x = core_grid.end_coord.x + 1, num_cores_y = core_grid.end_coord.y + 1;
    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, true);

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t in_tile_offset_by_batch =
            i < 16 ? i * sub_tile_line_bytes : ((i - 16) * sub_tile_line_bytes) + (512 * element_size);

        const auto& core = cores[i];
        std::vector<uint32_t> reader_runtime_args = {in_tile_offset_by_batch, q_base_addr};

        SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        SetRuntimeArgs(program, writer_kernel_id, core, reader_runtime_args);
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = reader_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .num_cores = num_cores,
            .cb_q_output = cb_q_output,
            .cb_k_output = cb_k_output,
            .cb_v_output = cb_v_output,
            .cores = cores,
            .element_size = element_size,
            .sub_tile_line_bytes = sub_tile_line_bytes}};
}

void NLPCreateQKVHeadsDecodeInterleavedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensors) {
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& num_cores = cached_program.shared_variables.num_cores;
    const auto& cb_q_output = cached_program.shared_variables.cb_q_output;
    const auto& cb_k_output = cached_program.shared_variables.cb_k_output;
    const auto& cb_v_output = cached_program.shared_variables.cb_v_output;
    const auto& cores = cached_program.shared_variables.cores;
    const auto& element_size = cached_program.shared_variables.element_size;
    const auto& sub_tile_line_bytes = cached_program.shared_variables.sub_tile_line_bytes;

    auto *dst_buffer_query = output_tensors.at(0).buffer();
    auto *dst_buffer_key = output_tensors.at(1).buffer();
    auto *dst_buffer_value = output_tensors.at(2).buffer();

    UpdateDynamicCircularBufferAddress(program, cb_q_output, *dst_buffer_query);
    UpdateDynamicCircularBufferAddress(program, cb_k_output, *dst_buffer_key);
    UpdateDynamicCircularBufferAddress(program, cb_v_output, *dst_buffer_value);

    uint32_t q_base_addr = tensor_args.input_tensor.buffer()->address();
    uint32_t q_start_addr = q_base_addr;

    for (uint32_t i = 0; i < num_cores; ++i) {
        uint32_t in_tile_offset_by_batch =
            i < 16 ? i * sub_tile_line_bytes : ((i - 16) * sub_tile_line_bytes) + (512 * element_size);
        const auto& core = cores[i];
        auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        runtime_args[0] = in_tile_offset_by_batch;
        runtime_args[1] = q_start_addr;

        auto& runtime_args_writer = GetRuntimeArgs(program, writer_kernel_id, core);
        runtime_args_writer[0] = in_tile_offset_by_batch;
        runtime_args_writer[1] = q_start_addr;
    }
}

}  // namespace ttnn::experimental::prim
