// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_nd_sharded_program_factory.hpp"

#include "ttnn/operations/embedding/device/kernels/dataflow/embeddings_reader_kernel_args.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/buffer_distribution_spec.hpp>

namespace ttnn::prim {

EmbeddingsNDShardedProgramFactory::cached_program_t EmbeddingsNDShardedProgramFactory::create(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& input = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;
    auto& output = tensor_return_value;
    const auto& embeddings_type = operation_attributes.embeddings_type;
    const auto& pad_token = operation_attributes.pad_token;

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    uint32_t input_element_size_bytes = input.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;

    // no padding for now
    const auto& input_nd_shard_spec = input.nd_shard_spec().value();
    CoreRangeSet all_cores = input_nd_shard_spec.grid;

    const uint32_t tile_height = input.tensor_spec().tile().get_height();
    const uint32_t tile_width = input.tensor_spec().tile().get_width();
    const bool input_is_tile_layout = (input.layout() == tt::tt_metal::Layout::TILE);

    auto shard_args = input.tensor_spec().compute_buffer_sharding_args();

    TT_FATAL(shard_args.buffer_distribution_spec().has_value(), "Buffer distribution spec is not set");
    const auto distribution_spec = shard_args.buffer_distribution_spec().value();
    const auto& cores = distribution_spec.cores_with_data();  // TODO: distribution_spec.cores(); Unutilise cores?

    // Align input page size to buffer alignment (same as embeddings RM) so that small shard shapes
    // work: TensorAccessor and NoC reads use the same stride as the buffer's page_address().
    // For TILE layout, each page is one tile (TILE_HEIGHT x TILE_WIDTH elements).
    auto alignment = input.buffer()->alignment();
    uint32_t index_elems_per_page{};
    uint32_t input_page_size{};
    if (input_is_tile_layout) {
        index_elems_per_page = tile_height * tile_width;
        input_page_size = input.buffer()->aligned_page_size();
    } else {
        index_elems_per_page = input_nd_shard_spec.shard_shape[-1];
        input_page_size = tt::align(static_cast<uint32_t>(index_elems_per_page * input_element_size_bytes), alignment);
    }

    //********** Create Buffers **********
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    tt::tt_metal::Buffer* out_buffer = output.buffer();
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
    std::vector<uint32_t> embedding_compile_time_args = ttnn::kernel_utils::to_vector(
        ttnn::kernel::CompileTimeEmbeddingsReaderKernelArgs{
            .cb_id_index = src1_cb_index,
            .input_page_size = input_page_size,
            .weight_stick_size = weight_page_size,
            .elems_per_page = index_elems_per_page,
            .input_block_size_bytes = index_elems_per_page * input_element_size_bytes,
            .input_buf_alignment = input.buffer()->alignment()});
    tt::tt_metal::TensorAccessorArgs(*input.buffer()).append_to(embedding_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weights.buffer()).append_to(embedding_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(out_buffer).append_to(embedding_compile_time_args);

    EmbeddingsIndexType embeddings_index_type;
    if (input.dtype() == DataType::BFLOAT16) {
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
        .input_buffer_src_addr = input.buffer()->address(),
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
    for (uint32_t start_shard_id = 0, core_id = 0; core_id < cores.size(); ++core_id, ++start_shard_id) {
        const CoreCoord& core = cores[core_id];

        // Reader run-time args
        reader_runtime_args.start_shard_id = start_shard_id;

        // offset to get next shard, for round robin shard distribution
        // to get next shard for same core, we need to add amount of cores to start_shard_id
        reader_runtime_args.next_shard_offset = cores.size();
        reader_runtime_args.num_shards = distribution_spec.num_shards_per_core(core_id);
        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, ttnn::kernel_utils::to_vector(reader_runtime_args));
    }

    return cached_program_t{
        std::move(program), {.reader_kernel_id = reader_kernel_id, .cores = cores, .cb_out = cb_out}};
}

void EmbeddingsNDShardedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const EmbeddingParams& /*operation_attributes*/,
    const EmbeddingInputs& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_variables = cached_program.shared_variables;
    const auto& reader_kernel_id = shared_variables.reader_kernel_id;
    const auto& cores = shared_variables.cores;
    const auto& cb_out = shared_variables.cb_out;

    auto* output_buffer = tensor_return_value.buffer();
    auto input_buffer_address = tensor_args.input_tensor_arg.buffer()->address();
    auto weights_buffer_address = tensor_args.weight_arg.buffer()->address();

    UpdateDynamicCircularBufferAddress(program, cb_out, *output_buffer);

    auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id);

    for (const auto& core : cores) {
        {
            auto& runtime_args = reader_runtime_args[core.x][core.y];
            runtime_args[0] = input_buffer_address;
            runtime_args[1] = weights_buffer_address;
        }
    }
}
}  // namespace ttnn::prim
