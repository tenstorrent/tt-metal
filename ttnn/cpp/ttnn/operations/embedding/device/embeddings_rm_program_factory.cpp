// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_rm_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts EmbeddingsRMProgramFactory::create_program_artifacts(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;
    auto& output = tensor_return_value;
    const auto& embeddings_type = operation_attributes.embeddings_type;
    const auto& pad_token = operation_attributes.pad_token;

    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& weights_mesh_tensor = weights.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
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

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    tt::DataFormat weights_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);

    constexpr uint32_t max_l1_budget_bytes = 1024 * 1024;  // 1MB budget for embedding output staging
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

    uint32_t out_dfb_total_size;
    if (output_sharded) {
        out_dfb_total_size = output.buffer()->aligned_size_per_bank();
    } else {
        uint32_t buffering_size = (num_blocks_per_core_group_1 > 1 || num_blocks_per_core_group_2 > 1) ? 2 : 1;
        out_dfb_total_size = buffering_size * chunk_size;
    }

    // PADDED and BINARY serve some weight rows out of a locally cached copy instead of fetching them
    // per token; the other embeddings types have no such rows, so the cache is absent for them.
    const bool use_local_cache = embeddings_type == EmbeddingsType::PADDED || embeddings_type == EmbeddingsType::BINARY;

    // -----------------------------------------------------------------------
    // Resource names
    // -----------------------------------------------------------------------
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    const DFBSpecName OUTPUT{"output"};
    const DFBSpecName INDEX_SCRATCH{"index_scratch"};
    const DFBSpecName WEIGHT_CACHE{"weight_cache"};

    const TensorParamName INPUT_PARAM{"input"};
    const TensorParamName WEIGHTS_PARAM{"weights"};
    const TensorParamName OUTPUT_PARAM{"output"};

    ProgramSpec spec;
    spec.name = "embeddings_rm";

    // -----------------------------------------------------------------------
    // Dataflow buffers
    // -----------------------------------------------------------------------
    // The staging buffer's total size has to divide evenly by its entry size. The two are derived from
    // different alignments, the entry size from the index buffer's and the sharded total from the
    // output shard's own, so they can disagree. Nothing downstream would catch a truncated entry
    // count: when the output is sharded the reader stages one entry per output row and no kernel
    // drains the buffer, so a short buffer stalls the reader partway through its rows.
    TT_FATAL(
        out_dfb_total_size % chunk_size == 0,
        "Embedding output staging buffer size {} B must be divisible by its entry size {} B",
        out_dfb_total_size,
        chunk_size);

    DataflowBufferSpec out_dfb{
        .unique_id = OUTPUT,
        .entry_size = chunk_size,
        .num_entries = out_dfb_total_size / chunk_size,
        .data_format_metadata = weights_data_format,
    };
    if (output_sharded) {
        // The staging buffer *is* the output shard: it is built on the output tensor's own SRAM, so
        // the rows the reader gathers land in place and no writer kernel is created below.
        out_dfb.borrowed_from = OUTPUT_PARAM;
    }
    spec.dataflow_buffers.push_back(std::move(out_dfb));

    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_SCRATCH,
        .entry_size = block_height * index_page_size,
        .num_entries = 1,
        .data_format_metadata = input_data_format,
    });

    if (use_local_cache) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        // PADDED caches the single pad row; BINARY caches rows 0 and 1.
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = WEIGHT_CACHE,
            .entry_size = cache_page_size,
            .num_entries = (embeddings_type == EmbeddingsType::PADDED) ? 1u : 2u,
            .data_format_metadata = weights_data_format,
        });
    }

    // -----------------------------------------------------------------------
    // Tensor parameters
    //
    // The output parameter is declared in every configuration: the writer binds it when the output is
    // interleaved, and the borrowed staging buffer resolves its address from it when the output is
    // sharded.
    // -----------------------------------------------------------------------
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = INPUT_PARAM, .spec = input_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = WEIGHTS_PARAM, .spec = weights_mesh_tensor.tensor_spec()});
    spec.tensor_parameters.push_back(
        TensorParameter{.unique_id = OUTPUT_PARAM, .spec = output_mesh_tensor.tensor_spec()});

    // -----------------------------------------------------------------------
    // Reader
    // -----------------------------------------------------------------------
    Group<DFBBinding> reader_dfb_bindings;
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = OUTPUT,
        .accessor_name = "in0",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    if (output_sharded) {
        // With no writer kernel the reader is the staging buffer's only endpoint, so it holds both
        // roles.
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUTPUT,
            .accessor_name = "in0",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }
    // The index scratch page never leaves the reader: it reserves the page once, decodes indices out
    // of it, and commits it at the end only to leave the buffer balanced. Both roles are the reader's.
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_SCRATCH,
        .accessor_name = "in1",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = INDEX_SCRATCH,
        .accessor_name = "in1",
        .endpoint_type = DFBEndpointType::CONSUMER,
    });
    if (use_local_cache) {
        // Likewise the weight cache: the reader fills it and reads tokens back out of it, with no
        // hand-off to another kernel.
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WEIGHT_CACHE,
            .accessor_name = "local_cache",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WEIGHT_CACHE,
            .accessor_name = "local_cache",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
    }

    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    // These defines and the weight cache's DFB binding share one condition, the embeddings type. That
    // is what lets the reader name the cache handle at all: a dfb:: handle exists only on the builds
    // where the host binds it, so the reader's reference to it is compiled out under the same defines
    // on the builds where it is not.
    KernelSpec::CompilerOptions::Defines embedding_defines{
        {enchantum::to_string(embeddings_type).data(), "1"},
        {enchantum::to_string(embeddings_index_type).data(), "1"},
    };

    Group<std::string> reader_rta_names = {"batch_offset", "weights_offset", "num_rows", "index_idx"};
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_rta_names.push_back("pad_token");
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings.cpp",
        .compiler_options = {.defines = embedding_defines},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = WEIGHTS_PARAM, .accessor_name = "weights"},
            },
        .compile_time_args =
            {
                {"input_page_size", input_page_size},
                {"weight_stick_size", weight_page_size},
                {"rows_per_block", block_height},
                {"input_block_size_bytes", block_height * input_element_size_bytes},
                {"chunk_size", chunk_size},
                {"num_chunks", num_chunks},
                {"last_chunk_size", last_chunk_size},
            },
        .runtime_arg_schema = {.runtime_arg_names = std::move(reader_rta_names)},
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    // -----------------------------------------------------------------------
    // Writer
    //
    // A sharded output needs no writer: the reader has already staged its rows in the output shard.
    // -----------------------------------------------------------------------
    if (!output_sharded) {
        Group<DFBBinding> writer_dfb_bindings = {
            DFBBinding{
                .dfb_spec_name = OUTPUT,
                .accessor_name = "out0",
                .endpoint_type = DFBEndpointType::CONSUMER,
            },
        };
        if (use_chunked) {
            spec.kernels.push_back(KernelSpec{
                .unique_id = WRITER,
                .source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/"
                          "embeddings_rm_writer_chunked.cpp",
                .dfb_bindings = std::move(writer_dfb_bindings),
                .tensor_bindings =
                    {
                        TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "dst"},
                    },
                .compile_time_args =
                    {
                        {"chunk_size", chunk_size},
                        {"num_chunks", num_chunks},
                        {"last_chunk_size", last_chunk_size},
                    },
                .runtime_arg_schema = {.runtime_arg_names = {"num_sticks", "start_id"}},
                .hw_config = ttnn::create_writer_datamovement_config(),
            });
        } else {
            spec.kernels.push_back(KernelSpec{
                .unique_id = WRITER,
                .source = "ttnn/cpp/ttnn/kernel/dataflow/"
                          "writer_unary_stick_layout_interleaved_start_id_metal2.cpp",
                .dfb_bindings = std::move(writer_dfb_bindings),
                .tensor_bindings =
                    {
                        TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "dst"},
                    },
                .runtime_arg_schema = {.runtime_arg_names = {"stick_size", "num_sticks", "start_id"}},
                .hw_config = ttnn::create_writer_datamovement_config(),
            });
        }
    }

    Group<KernelSpecName> work_unit_kernels = {READER};
    if (!output_sharded) {
        work_unit_kernels.push_back(WRITER);
    }
    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = std::move(work_unit_kernels),
        .target_nodes = all_cores,
    });

    // -----------------------------------------------------------------------
    // Run args
    // -----------------------------------------------------------------------
    uint32_t input_offset = 0;

    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"batch_offset", input_offset / num_blocks_per_batch},
                 {"weights_offset",
                  tt::round_down(input_offset % num_blocks_per_batch, block_height) * input_element_size_bytes},
                 {"num_rows", local_num_blocks},
                 {"index_idx", input_offset % num_blocks_per_batch % block_height}});
            if (embeddings_type == EmbeddingsType::PADDED) {
                AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"pad_token", pad_token.value()}});
            }
        }

        // Writer
        if (!output_sharded) {
            if (use_chunked) {
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"num_sticks", local_num_blocks}, {"start_id", input_offset}});
            } else {
                AddRuntimeArgsForNode(
                    writer_run_args.runtime_arg_values,
                    core,
                    {{"stick_size", output_page_size}, {"num_sticks", local_num_blocks}, {"start_id", input_offset}});
            }
        }

        input_offset += local_num_blocks;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    if (!output_sharded) {
        run_args.kernel_run_args.push_back(std::move(writer_run_args));
    }
    run_args.tensor_args.emplace(INPUT_PARAM, input_mesh_tensor);
    run_args.tensor_args.emplace(WEIGHTS_PARAM, weights_mesh_tensor);
    run_args.tensor_args.emplace(OUTPUT_PARAM, output_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
