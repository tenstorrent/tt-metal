// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_nd_sharded_program_factory.hpp"

#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_reader_kernel_args.hpp"
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

namespace ttnn::prim {

EmbeddingsNDShardedProgramFactory::cached_program_t EmbeddingsNDShardedProgramFactory::create(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;
    auto& output = tensor_return_value;
    const auto& embeddings_type = operation_attributes.embeddings_type;
    const auto& pad_token = operation_attributes.pad_token;

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    tt::tt_metal::Buffer* out_buffer = output.buffer();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    // IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    bool output_sharded = is_sharded(output.buffer()->buffer_layout());

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]

    // no padding for now
    const auto& a_nd_shard_spec = a.nd_shard_spec().value();
    CoreRangeSet all_cores = a_nd_shard_spec.grid;
    const auto distribution_spec = tt::tt_metal::BufferDistributionSpec::from_shard_spec(
        a.padded_shape(),
        a_nd_shard_spec.shard_shape,
        {a_nd_shard_spec.shard_shape[-2], a_nd_shard_spec.shard_shape[-1]},  //? why is this here?
        a_nd_shard_spec.grid,
        a_nd_shard_spec.orientation,
        a_nd_shard_spec.shard_distribution_strategy);
    // uint32_t num_shards_per_core = distribution_spec.num_shards_per_core(0);
    const auto page_mapping = distribution_spec.compute_page_mapping();
    // const auto& groups = distribution_spec.core_groups();
    // uint32_t num_compute_cores = all_cores.num_cores();
    uint32_t index_elems_per_row = a.nd_shard_spec().value().shard_shape[-1];  //(alignment / input_element_size_bytes);
    uint32_t input_page_size = index_elems_per_row * input_element_size_bytes;
    std::cout << "a.nd_shard_spec().value().shard_shape: " << a.nd_shard_spec().value().shard_shape << std::endl;

    //********** Create Buffers **********
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    constexpr uint32_t out_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(weight_page_size, {{out_cb_index, weights_cb_data_format}})
            .set_page_size(out_cb_index, weight_page_size);
    cb_out_config.set_globally_allocated_address(*out_buffer);
    auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(input_page_size, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, input_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    // ********** Create Kernels **********
    // reader
    std::vector<uint32_t> embedding_compile_time_args =
        ttnn::kernel_utils::to_vector(ttnn::kernel::CompileTimeEmbeddingsReaderKernelArgs{
            .cb_id_output = out_cb_index,
            .cb_id_index = src1_cb_index,
            .input_page_size = input_page_size,
            .weight_stick_size = weight_page_size,
            .elems_per_page = index_elems_per_row,
            .input_block_size_bytes = index_elems_per_row * input_element_size_bytes,
            .input_buf_alignment = a.buffer()->alignment()});
    tt::tt_metal::TensorAccessorArgs(*a.buffer()).append_to(embedding_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weights.buffer()).append_to(embedding_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(embedding_compile_time_args);

    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    std::map<std::string, std::string> embedding_defines = {
        {enchantum::to_string(embeddings_type).data(), "1"}, {enchantum::to_string(embeddings_index_type).data(), "1"}};

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_nd_sharded.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(embedding_compile_time_args, embedding_defines));

    auto reader_runtime_args = ttnn::kernel::EmbeddingsReaderKernelArgs{
        .input_buffer_src_addr = a.buffer()->address(),
        .weight_buffer_src_addr = weights.buffer()->address(),
        .output_buffer_src_addr = out_buffer->address(),
        .start_shard_id = 0,
        .next_shard_offset = 0,
        .num_shards = 0,
        .index_idx = 0,
    };

    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_args.index_idx = pad_token.value();
    }
    // ********** Set Runtime Arguments per core **********
    bool row_major = false;
    // TODO: What do we do here?
    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    // we load shards in round robin order,so core_id is also the start_shard_id
    for (uint32_t start_shard_id = 0, core_id = 0; core_id < cores.size(); ++core_id, ++start_shard_id) {
        const CoreCoord& core = cores[core_id];

        // Reader run-time args
        reader_runtime_args.start_shard_id = start_shard_id;

        // offset to get next shard, since we distribute shards round robin
        // 0 shard goes to core 0, 1 shard goes to core 1, etc.
        // to get next shard for core 0, we need to add amount of cores to start_shard_id
        reader_runtime_args.next_shard_offset = cores.size();
        reader_runtime_args.num_shards = distribution_spec.num_shards_per_core(core_id);
        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, ttnn::kernel_utils::to_vector(reader_runtime_args));
    }

    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = cores,
         .cb_out = cb_out,
         .output_sharded = output_sharded}};
}

void EmbeddingsNDShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const EmbeddingParams& /*operation_attributes*/,
    const EmbeddingInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_variables = cached_program.shared_variables;
    const auto& reader_kernel_id = shared_variables.reader_kernel_id;
    const auto& writer_kernel_id = shared_variables.writer_kernel_id;
    const auto& cores = shared_variables.cores;
    const auto& cb_out = shared_variables.cb_out;
    const auto& output_sharded = shared_variables.output_sharded;

    auto* output_buffer = tensor_return_value.buffer();
    auto output_buffer_address = output_buffer->address();
    auto input_buffer_address = tensor_args.input_tensor_arg.buffer()->address();
    auto weights_buffer_address = tensor_args.weight_arg.buffer()->address();

    if (output_sharded) {
        UpdateDynamicCircularBufferAddress(program, cb_out, *output_buffer);
    }

    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

    for (const auto& core : cores) {
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[0] = input_buffer_address;
            runtime_args[1] = weights_buffer_address;
        }

        if (!output_sharded) {
            auto& runtime_args = writer_runtime_args[core.x][core.y];
            runtime_args[0] = output_buffer_address;
        }
    }
}
}  // namespace ttnn::prim
