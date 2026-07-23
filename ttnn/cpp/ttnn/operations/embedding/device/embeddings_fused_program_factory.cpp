// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_fused_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts EmbeddingsFusedProgramFactory::create_program_artifacts(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    // Metal 2.0 named resource handles (function-local so the three factory TUs never collide under
    // unity build).
    const DFBSpecName SRC0_DFB{"src0"};            // legacy c_0: weights-in
    const DFBSpecName IDX_DFB{"index_scratch"};    // legacy c_1
    const DFBSpecName OUT_DFB{"out"};              // legacy c_2
    const DFBSpecName WCACHE_DFB{"weight_cache"};  // legacy c_3 (PADDED / BINARY only)
    const TensorParamName INPUT{"input"};
    const TensorParamName WEIGHTS{"weights"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    const auto& a = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;
    auto& output = tensor_return_value;
    const auto& embeddings_type = operation_attributes.embeddings_type;
    const auto& pad_token = operation_attributes.pad_token;

    const auto& input_mt = a.mesh_tensor();
    const auto& weights_mt = weights.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();

    ////////////////////////////////////////////////////////////////////////////
    //                 Buffer Setup
    ////////////////////////////////////////////////////////////////////////////

    IDevice* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    bool output_sharded = is_sharded(output.buffer()->buffer_layout());

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();

    // row major, page size is last dim
    uint32_t input_page_size = a.padded_shape()[-1] * input_element_size_bytes;
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]

    uint32_t batch_size = a.padded_shape()[0];
    uint32_t num_output_rows_per_batch = a.padded_shape()[-1];
    uint32_t num_output_rows = num_output_rows_per_batch * batch_size;
    // Note: num_blocks is just blocks along height
    uint32_t num_blocks = num_output_rows / TILE_HEIGHT;
    uint32_t num_blocks_per_batch = num_output_rows_per_batch / TILE_HEIGHT;
    uint32_t num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0, num_tiles_per_block = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    bool row_major = false;
    if (output_sharded) {
        const auto& shard_spec = output.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = shard_spec.shape[0] / TILE_HEIGHT;
        num_blocks_per_core_group_2 = 0;
        num_tiles_per_block = shard_spec.shape[1] / TILE_WIDTH;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    } else {
        auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
        std::tie(
            std::ignore,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);
        num_tiles_per_block = weights.padded_shape()[-1] / TILE_WIDTH;
    }
    uint32_t g1_numcores = core_group_1.num_cores();

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());
    uint32_t weights_single_tile_size = tt::tile_size(weights_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    // Hardcoded limit to reduce SRAM usage. Should be updated to be tuned based on overall SRAM usage
    constexpr uint32_t max_double_buffer_tiles = 64;

    constexpr uint32_t max_l1_budget_bytes = 1024 * 1024;  // 1MB budget for embedding CB
    uint32_t max_tiles_per_chunk = std::min(max_l1_budget_bytes / weights_single_tile_size, num_tiles_per_block);
    max_tiles_per_chunk = std::max(max_tiles_per_chunk, 1U);

    uint32_t required_memory_bytes = 2 * num_tiles_per_block * weights_single_tile_size;
    bool use_chunked_processing = required_memory_bytes > max_l1_budget_bytes;

    // For very large embeddings, use chunked processing
    uint32_t tiles_per_chunk;
    uint32_t num_chunks;
    uint32_t last_chunk_tiles;
    uint32_t buffering;

    if (use_chunked_processing) {
        // Keep tiles_per_chunk near the cap and let the last chunk be partial.
        // Reader/compute kernels handle the partial trailing chunk explicitly
        // via last_chunk_tiles.
        tiles_per_chunk = std::min(max_tiles_per_chunk, max_double_buffer_tiles);
        num_chunks = (num_tiles_per_block + tiles_per_chunk - 1) / tiles_per_chunk;
        last_chunk_tiles = num_tiles_per_block - (num_chunks - 1) * tiles_per_chunk;
        buffering = tiles_per_chunk > max_double_buffer_tiles ? 1 : 2;
    } else {
        // Use original non-chunked approach for smaller embeddings
        tiles_per_chunk = num_tiles_per_block;
        num_chunks = 1;
        last_chunk_tiles = num_tiles_per_block;
        buffering = num_tiles_per_block > max_double_buffer_tiles ? 1 : 2;
    }

    uint32_t weight_block_size;
    if (output_sharded) {
        weight_block_size = output.shard_spec().value().shape[1] * weights_element_size_bytes;
    } else {
        weight_block_size = weight_page_size;
    }

    // TODO: Can increase size for larger reads
    uint32_t input_block_size_bytes = TILE_HEIGHT * input_element_size_bytes;

    ////////////////////////////////////////////////////////////////////////////
    //                 DataflowBuffers
    ////////////////////////////////////////////////////////////////////////////
    const bool has_weight_cache =
        embeddings_type == EmbeddingsType::PADDED || embeddings_type == EmbeddingsType::BINARY;

    std::vector<DataflowBufferSpec> dfbs;

    // c_0 weights-in (reader produces, compute consumes)
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = SRC0_DFB,
        .entry_size = weights_single_tile_size,
        .num_entries = buffering * tiles_per_chunk,
        .data_format_metadata = weights_cb_data_format,
    });

    // c_1 index scratch (single toucher, reader-only)
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IDX_DFB,
        .entry_size = TILE_HEIGHT * input_element_size_bytes,
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    });

    // c_2 output. On sharded output the buffer is the resident output shard (borrowed memory) and no
    // writer runs; on interleaved output the compute produces into it and the writer drains it.
    uint32_t output_cb_num_entries;
    if (output_sharded) {
        output_cb_num_entries = output.buffer()->aligned_size_per_bank() / output_single_tile_size;
    } else {
        output_cb_num_entries = buffering * tiles_per_chunk;
    }
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = output_cb_num_entries,
        .data_format_metadata = output_cb_data_format,
    };
    if (output_sharded) {
        out_dfb.borrowed_from = OUTPUT;
    }
    dfbs.push_back(out_dfb);

    // c_3 weight cache (single toucher, reader-only), PADDED / BINARY only
    if (has_weight_cache) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = WCACHE_DFB,
            .entry_size = cache_page_size,
            .num_entries = embeddings_type == EmbeddingsType::BINARY ? 2u : 1u,
            .data_format_metadata = weights_cb_data_format,
        });
    }

    ////////////////////////////////////////////////////////////////////////////
    //                 Kernels
    ////////////////////////////////////////////////////////////////////////////
    KernelSpec::CompilerOptions::Defines embedding_defines = {
        {enchantum::to_string(embeddings_type).data(), "1"}, {enchantum::to_string(embeddings_index_type).data(), "1"}};

    // reader
    Group<DFBBinding> reader_dfb_bindings;
    reader_dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = SRC0_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER});
    reader_dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = IDX_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::PRODUCER});
    reader_dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = IDX_DFB, .accessor_name = "in1", .endpoint_type = DFBEndpointType::CONSUMER});
    if (has_weight_cache) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WCACHE_DFB, .accessor_name = "weight_cache", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WCACHE_DFB, .accessor_name = "weight_cache", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    Group<std::string> reader_rta_names = {"input_start_id", "input_start_offset", "weight_offset", "num_blocks"};
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_rta_names.push_back("pad_token");
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            std::filesystem::path{"ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize.cpp"},
        .compiler_options = {.defines = embedding_defines},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
             TensorBinding{.tensor_parameter_name = WEIGHTS, .accessor_name = "weights"}},
        .compile_time_args =
            {{"input_page_size", input_page_size},
             {"weight_block_size", weight_block_size},
             {"tiles_per_chunk", tiles_per_chunk},
             {"input_block_size_bytes", input_block_size_bytes},
             {"num_chunks", num_chunks},
             {"last_chunk_tiles", last_chunk_tiles}},
        .runtime_arg_schema = {.runtime_arg_names = std::move(reader_rta_names)},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // compute: one KernelSpec per non-empty core group, same source (chunked own kernel or the Metal 2.0
    // tilize fork), differing only in the per_core_block_cnt CTA.
    const std::filesystem::path compute_source =
        use_chunked_processing
            ? std::filesystem::path{"ttnn/cpp/ttnn/operations/embedding/device/kernels/compute/tilize_chunked.cpp"}
            : std::filesystem::path{
                  "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_metal2.cpp"};

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t per_core_block_cnt) {
        Group<DFBBinding> compute_dfb_bindings = {
            DFBBinding{.dfb_spec_name = SRC0_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}};
        if (output_sharded) {
            // Single-toucher borrowed output: self-loop the producing compute kernel.
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER});
        }
        KernelSpec::CompileTimeArgs compute_cta;
        if (use_chunked_processing) {
            compute_cta = {
                {"per_core_block_cnt", per_core_block_cnt},
                {"tiles_per_chunk", tiles_per_chunk},
                {"num_chunks", num_chunks},
                {"last_chunk_tiles", last_chunk_tiles}};
        } else {
            compute_cta = {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_tile_cnt", tiles_per_chunk}};
        }
        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_source,
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args = std::move(compute_cta),
            .hw_config = ComputeHardwareConfig{ComputeGen1Config{}},
        };
    };

    // writer (interleaved output only): Metal 2.0 fork of the shared interleaved writer.
    std::optional<KernelSpec> writer;
    if (!output_sharded) {
        writer = KernelSpec{
            .unique_id = WRITER,
            .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                            "writer_unary_interleaved_start_id_metal2.cpp"},
            .dfb_bindings = {DFBBinding{
                .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        };
    }

    ////////////////////////////////////////////////////////////////////////////
    //                 Runtime args (per node)
    ////////////////////////////////////////////////////////////////////////////
    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    KernelRunArgs reader_kra{.kernel = READER};
    std::optional<KernelRunArgs> writer_kra;
    if (!output_sharded) {
        writer_kra = KernelRunArgs{.kernel = WRITER};
    }

    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        AddRuntimeArgsForNode(
            reader_kra.runtime_arg_values,
            core,
            {{"input_start_id", input_offset / num_blocks_per_batch},
             {"input_start_offset", input_offset % num_blocks_per_batch * input_block_size_bytes},
             {"weight_offset", weight_offset},
             {"num_blocks", local_num_blocks}});
        if (embeddings_type == EmbeddingsType::PADDED) {
            AddRuntimeArgsForNode(reader_kra.runtime_arg_values, core, {{"pad_token", pad_token.value()}});
        }

        // Writer
        if (!output_sharded) {
            AddRuntimeArgsForNode(
                writer_kra->runtime_arg_values,
                core,
                {{"num_pages", num_tiles_per_block * local_num_blocks}, {"start_id", tile_offset}});
            tile_offset += local_num_blocks * num_tiles_per_block;
            input_offset += local_num_blocks;
        } else {
            weight_offset += weight_block_size;
            if (weight_offset == weight_page_size) {
                weight_offset = 0;
                input_offset += local_num_blocks;
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////
    //                 Assemble
    ////////////////////////////////////////////////////////////////////////////
    Group<KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    if (writer.has_value()) {
        kernels.push_back(std::move(*writer));
    }

    // WorkUnits: reader (and writer) run on all_cores (they appear in both groups' work units); each
    // compute KernelSpec covers one disjoint core group. Per node this yields exactly one reader, one
    // compute and (interleaved) one writer instance.
    std::vector<WorkUnitSpec> work_units;
    if (num_blocks_per_core_group_1 > 0) {
        kernels.push_back(make_compute(COMPUTE_G1, num_blocks_per_core_group_1));
        Group<KernelSpecName> wu_kernels = {READER, COMPUTE_G1};
        if (writer_kra.has_value()) {
            wu_kernels.push_back(WRITER);
        }
        work_units.push_back(
            WorkUnitSpec{.name = "wu_g1", .kernels = std::move(wu_kernels), .target_nodes = core_group_1});
    }
    if (num_blocks_per_core_group_2 > 0) {
        kernels.push_back(make_compute(COMPUTE_G2, num_blocks_per_core_group_2));
        Group<KernelSpecName> wu_kernels = {READER, COMPUTE_G2};
        if (writer_kra.has_value()) {
            wu_kernels.push_back(WRITER);
        }
        work_units.push_back(
            WorkUnitSpec{.name = "wu_g2", .kernels = std::move(wu_kernels), .target_nodes = core_group_2});
    }

    ProgramSpec spec{
        .name = "embedding_fused",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input_mt.tensor_spec()},
             TensorParameter{.unique_id = WEIGHTS, .spec = weights_mt.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output_mt.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_kra));
    if (writer_kra.has_value()) {
        run_args.kernel_run_args.push_back(std::move(*writer_kra));
    }
    run_args.tensor_args = {{INPUT, input_mt}, {WEIGHTS, weights_mt}, {OUTPUT, output_mt}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
