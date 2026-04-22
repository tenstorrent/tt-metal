// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "post_combine_reduce_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce {

namespace {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->num_pages(); }
uint32_t get_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->page_size(); }
uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) { return (uint32_t)tensor.buffer()->aligned_page_size(); }

struct CreatedProgram {
    tt::tt_metal::Program program;
    PostCombineReduceProgramFactory::shared_variables_t shared_variables;
};

CreatedProgram create_at(
    const PostCombineReduceParams& operation_attributes,
    [[maybe_unused]] const ttnn::MeshCoordinate& mesh_coordinate,
    const PostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const auto& combine_output = tensor_args.combine_output;
    const auto& weights = tensor_args.weights;
    const auto& indices = tensor_args.indices;
    const auto& expert_dispatch_table = tensor_args.expert_dispatch_table;
    auto* device = combine_output.device();

    const auto& combine_shape = combine_output.padded_shape();

    const uint32_t expert_dim = operation_attributes.expert_dim;

    const uint32_t emb_dim = combine_shape[-1];
    const uint32_t num_experts = combine_shape[expert_dim];

    uint32_t num_tokens = 1;
    for (uint32_t i = 0; i < expert_dim; ++i) {
        num_tokens *= combine_shape[i];
    }

    constexpr uint32_t TILE_SIZE = 1024;  // 32 x 32
    const uint32_t emb_dim_tiles = emb_dim / TILE_SIZE;

    TT_FATAL(
        emb_dim % TILE_SIZE == 0,
        "Embedding dimension {} must be divisible by tile size (1024), got {} tiles",
        emb_dim,
        emb_dim_tiles);
    TT_FATAL(
        emb_dim_tiles <= 8, "Embedding dimension tiles {} must fit in 8 DST registers for batching", emb_dim_tiles);

    constexpr uint32_t TOKENS_PER_CHUNK = 32;
    TT_FATAL(
        num_tokens % TOKENS_PER_CHUNK == 0,
        "Number of tokens {} must be divisible by {} for hardware tilization",
        num_tokens,
        TOKENS_PER_CHUNK);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    uint32_t num_cores_total = num_cores_x * num_cores_y;

    const uint32_t total_chunks = num_tokens / TOKENS_PER_CHUNK;
    TT_FATAL(total_chunks > 0, "post_combine_reduce: num_tokens must be > 0, got {}", num_tokens);
    const uint32_t num_cores = std::min(total_chunks, num_cores_total);
    const uint32_t base_chunks_per_core = total_chunks / num_cores;
    const uint32_t extra_chunks = total_chunks % num_cores;

    constexpr bool row_major = true;

    auto core_range_set = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, row_major);

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(combine_output.dtype());
    tt::DataFormat weight_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());
    tt::DataFormat indices_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices.dtype());
    tt::DataFormat dispatch_table_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(expert_dispatch_table.dtype());

    uint32_t tile_size = tt::tile_size(input_cb_data_format);

    // c_0: Stream one expert at a time through c_0 to minimize L1 footprint.
    uint32_t combine_cb_size = emb_dim_tiles * tile_size;
    tt::tt_metal::CircularBufferConfig cb_combine_config =
        tt::tt_metal::CircularBufferConfig(combine_cb_size, {{tt::CBIndex::c_0, input_cb_data_format}})
            .set_page_size(tt::CBIndex::c_0, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_combine_config);

    // c_1: Stream one weight at a time (matching expert-by-expert input streaming).
    uint32_t weight_cb_size = tile_size;
    tt::tt_metal::CircularBufferConfig cb_weight_config =
        tt::tt_metal::CircularBufferConfig(weight_cb_size, {{tt::CBIndex::c_1, weight_cb_data_format}})
            .set_page_size(tt::CBIndex::c_1, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_weight_config);

    // c_2: Dispatch table scratch — loaded once by writer, read by compute.
    uint32_t dispatch_table_num_pages = get_num_pages(expert_dispatch_table);
    uint32_t dispatch_table_aligned_page_size = get_aligned_page_size(expert_dispatch_table);
    uint32_t dispatch_table_cb_size = dispatch_table_num_pages * dispatch_table_aligned_page_size;
    tt::tt_metal::CircularBufferConfig cb_dispatch_table_config =
        tt::tt_metal::CircularBufferConfig(dispatch_table_cb_size, {{tt::CBIndex::c_2, dispatch_table_cb_data_format}})
            .set_page_size(tt::CBIndex::c_2, dispatch_table_aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_dispatch_table_config);

    // c_3: Indices scratch — sized for a single chunk. Writer reloads it at the
    // start of each chunk; compute pops it at the end, so the same CB slot is
    // reused across all chunks regardless of how many chunks a core processes.
    uint32_t indices_page_size = get_page_size(indices);
    uint32_t indices_aligned_page_size = get_aligned_page_size(indices);
    uint32_t indices_pages_per_core = TOKENS_PER_CHUNK;
    uint32_t indices_cb_size = indices_pages_per_core * indices_aligned_page_size;
    tt::tt_metal::CircularBufferConfig cb_indices_config =
        tt::tt_metal::CircularBufferConfig(indices_cb_size, {{tt::CBIndex::c_3, indices_cb_data_format}})
            .set_page_size(tt::CBIndex::c_3, indices_aligned_page_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_indices_config);

    // c_16: Output
    uint32_t output_cb_size = TOKENS_PER_CHUNK * emb_dim_tiles * tile_size;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(output_cb_size, {{tt::CBIndex::c_16, output_cb_data_format}})
            .set_page_size(tt::CBIndex::c_16, tile_size);
    auto cb_output_handle = tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_output_config);

    uint32_t rowmajor_cb_size = TOKENS_PER_CHUNK * emb_dim_tiles * tile_size;
    tt::tt_metal::CircularBufferConfig cb_rowmajor_config =
        tt::tt_metal::CircularBufferConfig(rowmajor_cb_size, {{tt::CBIndex::c_17, output_cb_data_format}})
            .set_page_size(tt::CBIndex::c_17, tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_range_set, cb_rowmajor_config);

    auto* combine_buffer = combine_output.buffer();
    auto* weight_buffer = weights.buffer();
    auto* indices_buffer = indices.buffer();
    auto* dispatch_table_buffer = expert_dispatch_table.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    // Reader compile-time args: num_experts, emb_dim_tiles, combine accessor
    std::vector<uint32_t> reader_compile_time_args = {
        num_experts,
        emb_dim_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(combine_buffer).append_to(reader_compile_time_args);

    // Compute compile-time args: num_experts, emb_dim_tiles,
    //   dispatch_table_num_pages, indices_pages_per_core
    std::vector<uint32_t> compute_compile_time_args = {
        num_experts,
        emb_dim_tiles,
        dispatch_table_num_pages,
        indices_pages_per_core,
    };

    // Writer compile-time args: num_experts, emb_dim_tiles,
    //   dispatch_table pages/sizes, indices pages/sizes,
    //   weight accessor, output accessor, dispatch_table accessor, indices accessor
    std::vector<uint32_t> writer_compile_time_args = {
        num_experts,
        emb_dim_tiles,
        dispatch_table_num_pages,
        get_page_size(expert_dispatch_table),
        dispatch_table_aligned_page_size,
        indices_pages_per_core,
        indices_page_size,
        indices_aligned_page_size,
    };
    tt::tt_metal::TensorAccessorArgs(weight_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dispatch_table_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_buffer).append_to(writer_compile_time_args);

    auto reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_reader.cpp",
        core_range_set,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_compute.cpp",
        core_range_set,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = false,
            .dst_full_sync_en = false,
            .compile_args = compute_compile_time_args,
        });

    auto writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/post_combine_reduce/device/kernels/"
        "deepseek_moe_post_combine_reduce_writer.cpp",
        core_range_set,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    // Distribute chunks of 32 tokens across cores. The first `extra_chunks` cores
    // get (base_chunks_per_core + 1) chunks; the remaining cores get base_chunks_per_core.
    uint32_t token_start = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t chunks_this_core = base_chunks_per_core + (i < extra_chunks ? 1 : 0);

        std::vector<uint32_t> reader_runtime_args = {
            combine_buffer->address(),
            token_start,
            chunks_this_core,
        };

        std::vector<uint32_t> compute_runtime_args = {
            token_start,
            chunks_this_core,
        };

        std::vector<uint32_t> writer_runtime_args = {
            weight_buffer->address(),
            output_buffer->address(),
            dispatch_table_buffer->address(),
            indices_buffer->address(),
            token_start,
            chunks_this_core,
        };

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);

        token_start += chunks_this_core * TOKENS_PER_CHUNK;
    }

    return CreatedProgram{
        std::move(program),
        {
            .reader_kernel_id = reader_kernel_id,
            .compute_kernel_id = compute_kernel_id,
            .writer_kernel_id = writer_kernel_id,
            .output_cb_handle = cb_output_handle,
            .cores = cores,
        }};
}

}  // namespace

PostCombineReduceProgramFactory::cached_mesh_workload_t PostCombineReduceProgramFactory::create_mesh_workload(
    const PostCombineReduceParams& operation_attributes,
    const ttnn::MeshCoordinateRangeSet& tensor_coords,
    const PostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    tt::tt_metal::distributed::MeshWorkload mesh_workload;
    std::unordered_map<ttnn::MeshCoordinateRange, shared_variables_t> shared_variables;

    for (const auto& coord : tensor_coords.coords()) {
        auto result = create_at(operation_attributes, coord, tensor_args, tensor_return_value);
        auto coord_range = ttnn::MeshCoordinateRange(coord);
        mesh_workload.add_program(coord_range, std::move(result.program));
        shared_variables.emplace(coord_range, std::move(result.shared_variables));
    }

    return cached_mesh_workload_t{std::move(mesh_workload), std::move(shared_variables)};
}

void PostCombineReduceProgramFactory::override_runtime_arguments(
    cached_mesh_workload_t& cached_workload,
    [[maybe_unused]] const PostCombineReduceParams& operation_attributes,
    const PostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    auto* combine_buffer = tensor_args.combine_output.buffer();
    auto* weight_buffer = tensor_args.weights.buffer();
    auto* indices_buffer = tensor_args.indices.buffer();
    auto* dispatch_table_buffer = tensor_args.expert_dispatch_table.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    for (auto& [range, program] : cached_workload.workload.get_programs()) {
        const auto& svars = cached_workload.shared_variables.at(range);

        for (const auto& core : svars.cores) {
            auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, svars.reader_kernel_id, core);
            reader_runtime_args[0] = combine_buffer->address();

            auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, svars.writer_kernel_id, core);
            writer_runtime_args[0] = weight_buffer->address();
            writer_runtime_args[1] = output_buffer->address();
            writer_runtime_args[2] = dispatch_table_buffer->address();
            writer_runtime_args[3] = indices_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::post_combine_reduce
