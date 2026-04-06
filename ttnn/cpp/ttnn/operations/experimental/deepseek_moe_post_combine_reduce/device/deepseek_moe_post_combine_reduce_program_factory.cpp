// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_moe_post_combine_reduce_program_factory.hpp"
#include "deepseek_moe_post_combine_reduce_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

DeepseekMoEPostCombineReduceProgramFactory::cached_program_t DeepseekMoEPostCombineReduceProgramFactory::create(
    const DeepseekMoEPostCombineReduceParams& operation_attributes,
    const DeepseekMoEPostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& combine_output = tensor_args.combine_output;
    const auto& weights = tensor_args.weights;
    auto* device = combine_output.device();

    // Get tensor shapes
    const auto& combine_shape = combine_output.padded_shape();

    const uint32_t expert_dim = operation_attributes.expert_dim;

    // Calculate dimensions
    // combine_output: [..., seq_len, num_experts, emb_dim] in ROW_MAJOR
    // weights: [..., seq_len, num_experts]
    // For simplicity, assume expert_dim is the second-to-last dimension
    const uint32_t emb_dim = combine_shape[-1];
    const uint32_t num_experts = combine_shape[expert_dim];

    // Calculate total tokens (product of all dims before expert_dim)
    uint32_t num_tokens = 1;
    for (uint32_t i = 0; i < expert_dim; ++i) {
        num_tokens *= combine_shape[i];
    }

    // Calculate tiles: each tile = 32×32 = 1024 elements
    constexpr uint32_t TILE_SIZE = 1024;  // 32 × 32
    const uint32_t emb_dim_tiles = emb_dim / TILE_SIZE;

    TT_FATAL(
        emb_dim % TILE_SIZE == 0,
        "Embedding dimension {} must be divisible by tile size (1024), got {} tiles",
        emb_dim,
        emb_dim_tiles);
    TT_FATAL(
        emb_dim_tiles <= 8, "Embedding dimension tiles {} must fit in 8 DST registers for batching", emb_dim_tiles);

    // Hardware tilize requires exactly 32 tokens per core
    constexpr uint32_t REQUIRED_TOKENS_PER_CORE = 32;
    TT_FATAL(
        num_tokens % REQUIRED_TOKENS_PER_CORE == 0,
        "Number of tokens {} must be divisible by {} for hardware tilization",
        num_tokens,
        REQUIRED_TOKENS_PER_CORE);

    // Use row-major ordering: cores go (0,0), (1,0), (2,0), ... then (0,1), (1,1), ...
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    // Check if we have enough cores for the required tokens per core
    uint32_t num_cores_needed = num_tokens / REQUIRED_TOKENS_PER_CORE;
    TT_FATAL(
        num_cores_needed <= num_cores_total,
        "Need {} cores ({} tokens / {} tokens per core) but only {} cores available",
        num_cores_needed,
        num_tokens,
        REQUIRED_TOKENS_PER_CORE,
        num_cores_total);

    // Use exactly num_tokens / 32 cores (validated above)
    uint32_t num_cores = num_cores_needed;

    constexpr bool row_major = true;

    // Create CoreRangeSet covering only the cores we actually use
    auto core_range_set = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, row_major);

    // Get cores in row-major order
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(combine_output.dtype());
    tt::DataFormat weight_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());

    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    // Circular Buffer Setup (Optimized - No intermediate tilize buffers!)
    // ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    uint32_t tile_size = tt::tile_size(input_cb_data_format);

    // std::cout << "combine_cb tiles: " << num_experts * emb_dim_tiles << std::endl;
    // std::cout << "weight_cb tiles: " << num_experts << std::endl;
    // std::cout << "output_cb tiles: " << 2*emb_dim_tiles << std::endl;
    // std::cout << "accumulator_cb tiles: " << 2*emb_dim_tiles << std::endl;

    // CB0: combine_input - ROW_MAJOR expert output treated as "fake tiles"
    // BULK LOADING: Hold ALL experts for one token (num_experts × emb_dim_tiles tiles)
    // For 8 experts × 7 tiles = 56 tiles = 114,688 bytes
    uint32_t combine_cb_size = num_experts * emb_dim_tiles * tile_size;
    tt::tt_metal::CircularBufferConfig cb_combine_config =
        tt::tt_metal::CircularBufferConfig(combine_cb_size, {{tt::CBIndex::c_0, input_cb_data_format}})
            .set_page_size(tt::CBIndex::c_0, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_combine_config);

    // CB1: weights - BULK LOADING: Hold ALL weights for one token
    // For 8 experts = 8 weight tiles (only first element of each tile used for SCALAR broadcast)
    uint32_t weight_cb_size = num_experts * tile_size;
    tt::tt_metal::CircularBufferConfig cb_weight_config =
        tt::tt_metal::CircularBufferConfig(weight_cb_size, {{tt::CBIndex::c_1, weight_cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_weight_config);

    // CB16: output - TILE_LAYOUT final result
    // Size: 224 tiles (32 tokens × 7 tile-rows per token)
    // Single-buffered (not double-buffered) to save L1 space - we produce/consume all at once
    uint32_t output_cb_size = 224 * tile_size;  // 32 tokens worth of tiles
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(output_cb_size, {{tt::CBIndex::c_16, output_cb_data_format}})
            .set_page_size(tt::CBIndex::c_16, tile_size);
    auto cb_output_handle = tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_output_config);

    // CB17: intermediate buffer for batching 32 tokens before tilize
    // Also used for L1 accumulation directly (no separate accumulator CB needed)
    // Size: 32 tokens × 7 tiles × 2048 bytes = 458,752 bytes
    // Page size = tile_size so pack_tile and TileSlice work naturally
    uint32_t rowmajor_cb_size = 32 * emb_dim_tiles * tile_size;
    tt::tt_metal::CircularBufferConfig cb_rowmajor_config =
        tt::tt_metal::CircularBufferConfig(rowmajor_cb_size, {{tt::CBIndex::c_17, output_cb_data_format}})
            .set_page_size(tt::CBIndex::c_17, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_rowmajor_config);

    // Buffer info
    auto* combine_buffer = combine_output.buffer();
    auto* weight_buffer = weights.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    bool combine_is_dram = combine_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool weight_is_dram = weight_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // Reader kernel compile-time args
    std::vector<uint32_t> reader_compile_time_args = {
        static_cast<uint32_t>(combine_is_dram),
        static_cast<uint32_t>(weight_is_dram),
        num_experts,
        emb_dim,
        emb_dim_tiles,
    };

    // Compute kernel compile-time args
    std::vector<uint32_t> compute_compile_time_args = {
        num_experts,
        emb_dim_tiles,
    };

    // Writer kernel compile-time args
    std::vector<uint32_t> writer_compile_time_args = {
        num_tokens,
        emb_dim_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_reader.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_compute.cpp",
        core_range_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi2,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = true,
            .compile_args = compute_compile_time_args,
        });

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_moe_post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_writer.cpp",
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Each core processes exactly 32 tokens (validated above)
    uint32_t token_start = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores[i];

        std::vector<uint32_t> reader_runtime_args = {
            combine_buffer->address(),
            weight_buffer->address(),
            token_start,
        };

        std::vector<uint32_t> compute_runtime_args = {
            token_start,
        };

        std::vector<uint32_t> writer_runtime_args = {
            output_buffer->address(),
            token_start,
        };

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        token_start += REQUIRED_TOKENS_PER_CORE;
    }

    return cached_program_t{
        std::move(program),
        {
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .output_cb_handle = cb_output_handle,
            .cores = cores,
        }};
}

void DeepseekMoEPostCombineReduceProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    [[maybe_unused]] const DeepseekMoEPostCombineReduceParams& operation_attributes,
    const DeepseekMoEPostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;

    auto* combine_buffer = tensor_args.combine_output.buffer();
    auto* weight_buffer = tensor_args.weights.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    for (const auto& core : cores) {
        // Update reader buffer addresses
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        reader_runtime_args[0] = combine_buffer->address();
        reader_runtime_args[1] = weight_buffer->address();

        // Update writer buffer address
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        writer_runtime_args[0] = output_buffer->address();
    }
}

}  // namespace ttnn::experimental::prim
