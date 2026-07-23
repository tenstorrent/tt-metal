// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_rm_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts EmbeddingsRMProgramFactory::create_program_artifacts(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    // Metal 2.0 named resource handles (function-local so the three factory TUs never collide under
    // unity build).
    const DFBSpecName OUT_DFB{"out"};              // legacy c_0
    const DFBSpecName IDX_DFB{"index_scratch"};    // legacy c_1
    const DFBSpecName WCACHE_DFB{"weight_cache"};  // legacy c_2 (PADDED / BINARY only)
    const TensorParamName INPUT{"input"};
    const TensorParamName WEIGHTS{"weights"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

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

    uint32_t num_blocks_per_core_group_1 = 0, num_blocks_per_core_group_2 = 0;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    bool row_major = false;
    if (output_sharded) {
        const auto& shard_spec = output.shard_spec().value();
        all_cores = shard_spec.grid;
        core_group_1 = all_cores;
        num_blocks_per_core_group_1 = shard_spec.shape[0];
        num_blocks_per_core_group_2 = 0;
        row_major = shard_spec.orientation == ShardOrientation::ROW_MAJOR;
    } else {
        std::tie(
            std::ignore,
            all_cores,
            core_group_1,
            core_group_2,
            num_blocks_per_core_group_1,
            num_blocks_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);
    }
    uint32_t g1_numcores = core_group_1.num_cores();

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);

    constexpr uint32_t max_l1_budget_bytes = 1024 * 1024;  // 1MB budget for embedding CB
    uint32_t chunk_size;
    uint32_t num_chunks;
    uint32_t last_chunk_size;
    bool use_chunked = !output_sharded && rounded_weight_page_size > max_l1_budget_bytes;
    if (use_chunked) {
        chunk_size = (max_l1_budget_bytes / alignment) * alignment;
        chunk_size = std::max(chunk_size, alignment);
        num_chunks = (rounded_weight_page_size + chunk_size - 1) / chunk_size;
        last_chunk_size = rounded_weight_page_size - (num_chunks - 1) * chunk_size;
    } else {
        chunk_size = rounded_weight_page_size;
        num_chunks = 1;
        last_chunk_size = rounded_weight_page_size;
    }

    uint32_t out_cb_size;
    if (output_sharded) {
        out_cb_size = output.buffer()->aligned_size_per_bank();
    } else {
        uint32_t buffering_size = (num_blocks_per_core_group_1 > 1 || num_blocks_per_core_group_2 > 1) ? 2 : 1;
        out_cb_size = buffering_size * chunk_size;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                 DataflowBuffers
    ////////////////////////////////////////////////////////////////////////////
    const bool has_weight_cache =
        embeddings_type == EmbeddingsType::PADDED || embeddings_type == EmbeddingsType::BINARY;

    std::vector<DataflowBufferSpec> dfbs;

    // c_0 output. On sharded output the buffer is the resident output shard (borrowed memory) and no
    // writer runs; on interleaved output the reader produces into it and the writer drains it.
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = chunk_size,
        .num_entries = out_cb_size / chunk_size,
        .data_format_metadata = weights_cb_data_format,
    };
    if (output_sharded) {
        out_dfb.borrowed_from = OUTPUT;
    }
    dfbs.push_back(out_dfb);

    // c_1 index scratch (single toucher, reader-only)
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IDX_DFB,
        .entry_size = block_height * index_page_size,
        .num_entries = 1,
        .data_format_metadata = input_cb_data_format,
    });

    // c_2 weight cache (single toucher, reader-only), PADDED / BINARY only
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
    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }
    KernelSpec::CompilerOptions::Defines embedding_defines = {
        {enchantum::to_string(embeddings_type).data(), "1"}, {enchantum::to_string(embeddings_index_type).data(), "1"}};

    // reader
    Group<DFBBinding> reader_dfb_bindings;
    reader_dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER});
    if (output_sharded) {
        // Single-toucher borrowed output: self-loop the producing reader.
        reader_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER});
    }
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

    Group<std::string> reader_rta_names = {"batch_offset", "weights_offset", "num_rows", "index_idx"};
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_rta_names.push_back("pad_token");
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path{"ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp"},
        .compiler_options = {.defines = embedding_defines},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
             TensorBinding{.tensor_parameter_name = WEIGHTS, .accessor_name = "weights"}},
        .compile_time_args =
            {{"input_page_size", input_page_size},
             {"weight_stick_size", weight_page_size},
             {"rows_per_block", block_height},
             {"input_block_size_bytes", block_height * input_element_size_bytes},
             {"chunk_size", chunk_size},
             {"num_chunks", num_chunks},
             {"last_chunk_size", last_chunk_size}},
        .runtime_arg_schema = {.runtime_arg_names = std::move(reader_rta_names)},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // writer (interleaved output only)
    std::optional<KernelSpec> writer;
    if (!output_sharded) {
        Group<std::string> writer_rta_names;
        std::filesystem::path writer_source;
        if (use_chunked) {
            writer_source =
                "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_rm_writer_chunked.cpp";
            writer_rta_names = {"num_sticks", "start_id"};
            writer = KernelSpec{
                .unique_id = WRITER,
                .source = writer_source,
                .dfb_bindings = {DFBBinding{
                    .dfb_spec_name = OUT_DFB, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER}},
                .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
                .compile_time_args =
                    {{"chunk_size", chunk_size}, {"num_chunks", num_chunks}, {"last_chunk_size", last_chunk_size}},
                .runtime_arg_schema = {.runtime_arg_names = std::move(writer_rta_names)},
                .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
            };
        } else {
            // Metal 2.0 fork of the shared stick-layout writer.
            writer_source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp";
            writer_rta_names = {"stick_size", "num_sticks", "start_id"};
            writer = KernelSpec{
                .unique_id = WRITER,
                .source = writer_source,
                .dfb_bindings = {DFBBinding{
                    .dfb_spec_name = OUT_DFB, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER}},
                .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
                .runtime_arg_schema = {.runtime_arg_names = std::move(writer_rta_names)},
                .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
            };
        }
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
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        AddRuntimeArgsForNode(
            reader_kra.runtime_arg_values,
            core,
            {{"batch_offset", input_offset / num_blocks_per_batch},
             {"weights_offset",
              tt::round_down(input_offset % num_blocks_per_batch, block_height) * input_element_size_bytes},
             {"num_rows", local_num_blocks},
             {"index_idx", input_offset % num_blocks_per_batch % block_height}});
        if (embeddings_type == EmbeddingsType::PADDED) {
            AddRuntimeArgsForNode(reader_kra.runtime_arg_values, core, {{"pad_token", pad_token.value()}});
        }

        // Writer
        if (!output_sharded) {
            if (use_chunked) {
                AddRuntimeArgsForNode(
                    writer_kra->runtime_arg_values,
                    core,
                    {{"num_sticks", local_num_blocks}, {"start_id", input_offset}});
            } else {
                AddRuntimeArgsForNode(
                    writer_kra->runtime_arg_values,
                    core,
                    {{"stick_size", static_cast<uint32_t>(output_page_size)},
                     {"num_sticks", local_num_blocks},
                     {"start_id", input_offset}});
            }
        }

        input_offset += local_num_blocks;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                 Assemble
    ////////////////////////////////////////////////////////////////////////////
    Group<KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    Group<KernelSpecName> wu_kernels = {READER};
    if (writer.has_value()) {
        kernels.push_back(std::move(*writer));
        wu_kernels.push_back(WRITER);
    }

    ProgramSpec spec{
        .name = "embedding_rm",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input_mt.tensor_spec()},
             TensorParameter{.unique_id = WEIGHTS, .spec = weights_mt.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output_mt.tensor_spec()}},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = std::move(wu_kernels), .target_nodes = all_cores}},
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
