// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_rm_program_factory.hpp"
#include "embedding_program_factory_common.hpp"
#include <tt-metalium/tt_align.hpp>

namespace ttnn::prim {

EmbeddingsRMProgramFactory::cached_program_t EmbeddingsRMProgramFactory::create(
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
    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////
    Program program{};

    bool output_sharded = is_sharded(output.buffer()->buffer_layout());

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t output_element_size_bytes = output.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.padded_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;
    uint32_t output_page_size = output.padded_shape()[-1] * output_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]

    uint32_t batch_size = a.padded_shape()[0];
    uint32_t num_output_rows_per_batch = a.padded_shape()[-1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    auto alignment = a.buffer()->alignment();
    uint32_t block_height = (alignment / input_element_size_bytes);
    uint32_t num_blocks = num_output_rows;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch;

    // setup problem and grid size

    uint32_t problem_size = num_blocks;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    uint32_t num_cores, num_blocks_per_core_group_1, num_blocks_per_core_group_2;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    bool row_major = false;
    if (output_sharded) {
        const auto& shard_spec = output.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_cores = all_cores.num_cores();
        num_blocks_per_core_group_1 = shard_spec.shape[0];
        num_blocks_per_core_group_2 = 0;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    } else {
        std::tie(
            num_cores,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);
    }
    uint32_t g1_numcores = core_group_1.num_cores();

    // Create Buffers
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    constexpr uint32_t out_cb_index = tt::CBIndex::c_0;
    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);
    uint32_t out_cb_size;
    if (output_sharded) {
        out_cb_size = output.buffer()->aligned_size_per_bank();
    } else {
        uint32_t buffering_size = (num_blocks_per_core_group_1 > 1 || num_blocks_per_core_group_2 > 1) ? 2 : 1;
        out_cb_size = buffering_size * rounded_weight_page_size;
    }
    tt::tt_metal::CircularBufferConfig cb_out_config =
        tt::tt_metal::CircularBufferConfig(out_cb_size, {{out_cb_index, weights_cb_data_format}})
            .set_page_size(out_cb_index, rounded_weight_page_size);
    if (output_sharded) {
        cb_out_config.set_globally_allocated_address(*out_buffer);
    }
    auto cb_out = tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_out_config);

    constexpr uint32_t src1_cb_index = tt::CBIndex::c_1;
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    tt::tt_metal::CircularBufferConfig cb_src1_config =
        tt::tt_metal::CircularBufferConfig(block_height * index_page_size, {{src1_cb_index, input_cb_data_format}})
            .set_page_size(src1_cb_index, block_height * index_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src1_config);

    constexpr uint32_t src2_cb_index = tt::CBIndex::c_2;
    if (embeddings_type == EmbeddingsType::PADDED) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt::tt_metal::CircularBufferConfig cb_src2_config =
            tt::tt_metal::CircularBufferConfig(cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    } else if (embeddings_type == EmbeddingsType::BINARY) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        tt::tt_metal::CircularBufferConfig cb_src2_config =
            tt::tt_metal::CircularBufferConfig(2 * cache_page_size, {{src2_cb_index, weights_cb_data_format}})
                .set_page_size(src2_cb_index, cache_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src2_config);
    }

    // Create Kernels
    // reader
    std::vector<uint32_t> embedding_compile_time_args = {
        (std::uint32_t)out_cb_index,
        (std::uint32_t)src1_cb_index,
        (std::uint32_t)src2_cb_index,
        (std::uint32_t)input_page_size,
        (std::uint32_t)weight_page_size,
        (std::uint32_t)block_height,
        (std::uint32_t)block_height * input_element_size_bytes};
    tt::tt_metal::TensorAccessorArgs(*a.buffer()).append_to(embedding_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*weights.buffer()).append_to(embedding_compile_time_args);

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
        "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(embedding_compile_time_args, embedding_defines));

    // Tilized writer
    tt::tt_metal::KernelHandle writer_kernel_id = 0;
    if (!output_sharded) {
        std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)out_cb_index, (std::uint32_t)output_page_size};
        tt::tt_metal::TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

        writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/deprecated/tt_dnn/kernels/dataflow/writer_unary_stick_layout_interleaved_start_id.cpp",
            all_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));
    }

    uint32_t input_offset = 0;

    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);
    std::vector<uint32_t> reader_runtime_args = {
        (std::uint32_t)a.buffer()->address(),
        (std::uint32_t)weights.buffer()->address(),
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
        (std::uint32_t)0,
    };
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_args.push_back(pad_token.value());
    }
    std::vector<uint32_t> writer_runtime_args = {
        (std::uint32_t)output.buffer()->address(), (std::uint32_t)output_page_size, (std::uint32_t)0, (std::uint32_t)0};

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            reader_runtime_args[2] = input_offset / num_blocks_per_batch;
            reader_runtime_args[3] =
                tt::round_down(input_offset % num_blocks_per_batch, block_height) * input_element_size_bytes;
            reader_runtime_args[4] = local_num_blocks;
            reader_runtime_args[5] = input_offset % num_blocks_per_batch % block_height;
            tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        }

        // Writer
        if (!output_sharded) {
            writer_runtime_args[2] = local_num_blocks;
            writer_runtime_args[3] = input_offset;
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }

        input_offset += local_num_blocks;
    }

    return cached_program_t{
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .writer_kernel_id = writer_kernel_id,
         .cores = cores,
         .cb_out = cb_out,
         .output_sharded = output_sharded}};
}

void EmbeddingsRMProgramFactory::override_runtime_arguments(
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
