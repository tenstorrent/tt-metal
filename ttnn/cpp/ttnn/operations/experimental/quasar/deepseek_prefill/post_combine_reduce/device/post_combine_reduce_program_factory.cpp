// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "post_combine_reduce_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::experimental::quasar::deepseek_prefill::post_combine_reduce {

namespace {

uint32_t get_num_pages(const ttnn::Tensor& tensor) { return (uint32_t)tensor.mesh_buffer().num_pages(); }
uint32_t get_aligned_page_size(const ttnn::Tensor& tensor) {
    return (uint32_t)tensor.mesh_buffer().aligned_page_size();
}

}  // namespace

ttnn::device_operation::ProgramArtifacts PostCombineReduceProgramFactory::create_program_artifacts(
    const PostCombineReduceParams& operation_attributes,
    const PostCombineReduceInputs& tensor_args,
    ttnn::Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    const KernelSpecName READER{"reader"};
    const KernelSpecName COMPUTE{"compute"};
    const KernelSpecName WRITER{"writer"};
    const DFBSpecName COMBINE_INPUT{"combine_input"};
    const DFBSpecName WEIGHTS_DFB{"weights"};
    const DFBSpecName DISPATCH_TABLE_DFB{"dispatch_table"};
    const DFBSpecName INDICES_DFB{"indices"};
    const DFBSpecName OUTPUT_DFB{"output"};
    const DFBSpecName ROWMAJOR{"rowmajor"};
    const ScratchpadSpecName WTOK{"wtok"};
    const ScratchpadSpecName ROW_INDEX_SCRATCH{"row_index"};
    const TensorParamName COMBINE{"combine"};
    const TensorParamName WEIGHT{"weight"};
    const TensorParamName OUTPUT{"output"};
    const TensorParamName DISPATCH_TABLE{"dispatch_table"};
    const TensorParamName INDICES{"indices"};
    const TensorParamName ROW_INDEX{"row_index"};

    const auto& combine_output = tensor_args.combine_output;
    const auto& weights = tensor_args.weights;
    const auto& indices_opt = tensor_args.indices;
    const auto& dispatch_table_opt = tensor_args.expert_dispatch_table;
    // both-or-neither enforced in validate(); here we just pick the skip mode
    const bool use_dispatch_table_skip = indices_opt.has_value();
    auto* device = combine_output.device();

    const ttnn::Shape combine_shape = logical_combine_shape(tensor_args);
    const auto& row_index_opt = tensor_args.combine_row_index;
    const bool row_indexed = row_index_opt.has_value();

    const uint32_t expert_dim = operation_attributes.expert_dim;

    const uint32_t emb_dim = combine_shape[-1];
    const uint32_t num_experts = combine_shape[expert_dim];

    uint32_t num_tokens = 1;
    for (uint32_t i = 0; i < expert_dim; ++i) {
        num_tokens *= combine_shape[i];
    }

    constexpr uint32_t TILE_SIZE = 1024;  // 32 x 32 bfloat16 tile (element count)
    constexpr uint32_t TILE_WIDTH = 32;   // allow-redefined-helper: verbatim Quasar copy of the original op
    constexpr uint32_t BF16_BYTES = 2;

    // Number of tile-sized DFB entries needed to hold one emb_dim row.
    // ceil(emb_dim / 1024) supports non-1024-aligned dims (e.g. GPT-OSS 2880).
    const uint32_t emb_dim_dfb_tiles = (emb_dim + TILE_SIZE - 1) / TILE_SIZE;
    // Number of real 32x32 output tiles per 32-token block.
    const uint32_t emb_dim_out_tiles = emb_dim / TILE_WIDTH;
    // Raw byte count for NoC reads in the reader (handles non-aligned emb_dim).
    const uint32_t emb_dim_bytes = emb_dim * BF16_BYTES;

    TT_FATAL(
        emb_dim % TILE_WIDTH == 0,
        "Embedding dimension {} must be divisible by tile width ({}); remainder is {}",
        emb_dim,
        TILE_WIDTH,
        emb_dim % TILE_WIDTH);
    TT_FATAL(
        emb_dim_dfb_tiles <= 8,
        "Embedding dimension tiles {} must fit in 8 DST registers for batching",
        emb_dim_dfb_tiles);

    constexpr uint32_t TOKENS_PER_CHUNK = 32;
    TT_FATAL(num_tokens > 0, "post_combine_reduce: num_tokens must be > 0, got {}", num_tokens);
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
    const uint32_t num_cores = std::min(total_chunks, num_cores_total);
    const uint32_t base_chunks_per_core = total_chunks / num_cores;
    const uint32_t extra_chunks = total_chunks % num_cores;

    constexpr bool row_major = true;

    auto core_range_set = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, row_major);

    auto cores = tt::tt_metal::grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    tt::DataFormat input_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(combine_output.dtype());
    tt::DataFormat weight_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());
    tt::DataFormat output_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor_return_value.dtype());

    uint32_t tile_size = tt::tile_size(input_dfb_data_format);
    // Per-token weights ([..., num_experts]): one page holds a token's num_experts weights; the writer reads a
    // 32-token chunk of pages at once into the wtok scratchpad instead of one DRAM read per (token, expert).
    const bool weights_per_token = weights.padded_shape().rank() + 1 == combine_shape.rank();
    const uint32_t weights_aligned_page_size = get_aligned_page_size(weights);

    // combine_input: the reader fetches all of a token's expert rows under one NoC barrier and double-buffers
    // tokens, so reads overlap compute instead of paying a round trip per row. Only when two tokens'
    // rows fit in C0_BATCH_BUDGET_BYTES; wider rows (DeepSeek/Kimi 7168 x 8 experts) keep streaming one
    // expert at a time to hold the L1 footprint. Row-indexed combine also streams: the reader zeroes
    // its single combine_input entry once so entries it skips stay finite.
    constexpr uint32_t C0_BATCH_BUDGET_BYTES = 64 * 1024;
    const uint32_t token_rows_bytes = num_experts * emb_dim_dfb_tiles * tile_size;
    const bool batch_experts = !row_indexed && 2 * token_rows_bytes <= C0_BATCH_BUDGET_BYTES;
    const uint32_t experts_per_batch = batch_experts ? num_experts : 1;
    uint32_t combine_dfb_size = batch_experts ? 2 * token_rows_bytes : emb_dim_dfb_tiles * tile_size;
    Group<DataflowBufferSpec> dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = COMBINE_INPUT,
            .entry_size = tile_size,
            .num_entries = combine_dfb_size / tile_size,
            .data_format_metadata = input_dfb_data_format,
        },
    };

    // weights: the same rule for the weights, one tile per expert. The writer reads all of a token's
    // weights under one barrier, two tokens deep, only when that fits in C1_BATCH_BUDGET_BYTES (up to
    // 8 bf16 experts per token, every model today); a wider top-k streams one weight at a time in a
    // single tile, as before, so L1 does not grow with the expert count.
    constexpr uint32_t C1_BATCH_BUDGET_BYTES = 32 * 1024;
    const uint32_t token_weights_bytes = num_experts * tile_size;
    const bool batch_weights = 2 * token_weights_bytes <= C1_BATCH_BUDGET_BYTES;
    const uint32_t weights_per_batch = batch_weights ? num_experts : 1;
    uint32_t weight_dfb_size = batch_weights ? 2 * token_weights_bytes : tile_size;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WEIGHTS_DFB,
        .entry_size = tile_size,
        .num_entries = weight_dfb_size / tile_size,
        .data_format_metadata = weight_dfb_data_format,
    });

    // The dispatch table and indices DFBs are only allocated when the
    // DeepSeek skip path is in use; the GPT-OSS path does not touch them.
    uint32_t dispatch_table_num_pages = 0;
    uint32_t dispatch_table_aligned_page_size = 0;
    uint32_t indices_aligned_page_size = 0;
    uint32_t indices_pages_per_core = 0;

    if (use_dispatch_table_skip) {
        const auto& indices = *indices_opt;
        const auto& expert_dispatch_table = *dispatch_table_opt;

        tt::DataFormat indices_dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(indices.dtype());
        tt::DataFormat dispatch_table_dfb_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(expert_dispatch_table.dtype());

        dispatch_table_num_pages = get_num_pages(expert_dispatch_table);
        dispatch_table_aligned_page_size = get_aligned_page_size(expert_dispatch_table);
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = DISPATCH_TABLE_DFB,
            .entry_size = dispatch_table_aligned_page_size,
            .num_entries = dispatch_table_num_pages,
            .data_format_metadata = dispatch_table_dfb_data_format,
        });

        indices_aligned_page_size = get_aligned_page_size(indices);
        indices_pages_per_core = TOKENS_PER_CHUNK;
        dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = INDICES_DFB,
            .entry_size = indices_aligned_page_size,
            .num_entries = indices_pages_per_core,
            .data_format_metadata = indices_dfb_data_format,
        });
    }

    // wtok and row_index are filled and read by one DM kernel each, so they are scratchpads rather than DFBs.
    Group<ScratchpadSpec> scratchpads;
    if (weights_per_token) {
        scratchpads.push_back(ScratchpadSpec{
            .unique_id = WTOK,
            .size_per_node = TOKENS_PER_CHUNK * weights_aligned_page_size,
        });
    }

    uint32_t row_index_aligned_page_size = 0;
    if (row_indexed) {
        row_index_aligned_page_size = get_aligned_page_size(*row_index_opt);
        scratchpads.push_back(ScratchpadSpec{
            .unique_id = ROW_INDEX_SCRATCH,
            .size_per_node = TOKENS_PER_CHUNK * row_index_aligned_page_size,
        });
    }

    // output: one chunk at a time (compute produces TOKENS_PER_CHUNK tiles per iteration)
    uint32_t output_dfb_size = TOKENS_PER_CHUNK * emb_dim_dfb_tiles * tile_size;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT_DFB,
        .entry_size = tile_size,
        .num_entries = output_dfb_size / tile_size,
        .data_format_metadata = output_dfb_data_format,
    });

    // rowmajor: row-major scratch for tilize — one chunk at a time
    uint32_t rowmajor_dfb_size = TOKENS_PER_CHUNK * emb_dim_dfb_tiles * tile_size;
    dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = ROWMAJOR,
        .entry_size = tile_size,
        .num_entries = rowmajor_dfb_size / tile_size,
        .data_format_metadata = output_dfb_data_format,
    });

    // Reader: does not need to know about expert-skip; that logic lives in compute + writer.
    // Row-indexed combine (ROW_INDEXED): skip row, index page size and the table's row count.
    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/post_combine_reduce/device/kernels/"
            "deepseek_moe_post_combine_reduce_reader.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = COMBINE_INPUT,
                    .accessor_name = "combine_input",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = COMBINE, .accessor_name = "combine"},
            },
        .compile_time_args =
            {
                {"num_experts", num_experts},
                {"emb_dim_dfb_tiles", emb_dim_dfb_tiles},
                {"emb_dim_bytes", emb_dim_bytes},
                {"experts_per_batch", experts_per_batch},
                {"skip_row", operation_attributes.skip_row},
                {"row_index_aligned_page_size", row_index_aligned_page_size},
                {"table_rows", row_indexed ? static_cast<uint32_t>(combine_output.padded_shape()[0]) : 0u},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"token_start_idx", "num_chunks"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    if (row_indexed) {
        reader_spec.compiler_options.defines.emplace("ROW_INDEXED", "1");
        reader_spec.scratchpad_bindings.push_back(
            ScratchpadBinding{.scratchpad_spec_name = ROW_INDEX_SCRATCH, .accessor_name = "row_index"});
        reader_spec.tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = ROW_INDEX, .accessor_name = "row_index"});
    }

    // Compute: the DeepSeek-only arg (dispatch_table page count) is zero in the GPT-OSS path.
    KernelSpec compute_spec{
        .unique_id = COMPUTE,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/post_combine_reduce/device/kernels/"
            "deepseek_moe_post_combine_reduce_compute.cpp"),
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = COMBINE_INPUT,
                    .accessor_name = "combine_input",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = WEIGHTS_DFB,
                    .accessor_name = "weights",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_DFB,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // Compute is the sole toucher of the row-major tilize input: a self-loop.
                DFBBinding{
                    .dfb_spec_name = ROWMAJOR,
                    .accessor_name = "rowmajor",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = ROWMAJOR,
                    .accessor_name = "rowmajor",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .compile_time_args =
            {
                {"num_experts", num_experts},
                {"emb_dim_dfb_tiles", emb_dim_dfb_tiles},
                {"dispatch_table_num_pages", dispatch_table_num_pages},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"token_start_idx", "num_chunks"}},
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = MathFidelity::HiFi4,
                .enable_32_bit_dest = false,
                .double_buffer_dest = true,
            },
    };

    // Writer: in the GPT-OSS path the dispatch_table / indices metadata args carry zeros and their tensors and DFBs
    // are not bound; the kernel guards every use of them with USE_DISPATCH_TABLE_SKIP.
    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/deepseek_prefill/post_combine_reduce/device/kernels/"
            "deepseek_moe_post_combine_reduce_writer.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = WEIGHTS_DFB,
                    .accessor_name = "weights",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT_DFB,
                    .accessor_name = "output",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = WEIGHT, .accessor_name = "weight"},
                TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"},
            },
        .compile_time_args =
            {
                {"num_experts", num_experts},
                {"emb_dim_dfb_tiles", emb_dim_dfb_tiles},
                {"emb_dim_out_tiles", emb_dim_out_tiles},
                {"dispatch_table_num_pages", dispatch_table_num_pages},
                {"dispatch_table_aligned_page_size", dispatch_table_aligned_page_size},
                {"indices_aligned_page_size", indices_aligned_page_size},
                {"weights_aligned_page_size", weights_aligned_page_size},
                {"weights_per_batch", weights_per_batch},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"token_start_idx", "num_chunks"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    if (use_dispatch_table_skip) {
        for (auto* spec : {&compute_spec, &writer_spec}) {
            spec->compiler_options.defines.emplace("USE_DISPATCH_TABLE_SKIP", "1");
        }
        compute_spec.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DISPATCH_TABLE_DFB,
            .accessor_name = "dispatch_table",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_spec.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INDICES_DFB,
            .accessor_name = "indices",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        writer_spec.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DISPATCH_TABLE_DFB,
            .accessor_name = "dispatch_table",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer_spec.dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = INDICES_DFB,
            .accessor_name = "indices",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        writer_spec.tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = DISPATCH_TABLE, .accessor_name = "dispatch_table"});
        writer_spec.tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = INDICES, .accessor_name = "indices"});
    }
    if (weights_per_token) {
        writer_spec.compiler_options.defines.emplace("WEIGHTS_PER_TOKEN", "1");
        writer_spec.scratchpad_bindings.push_back(
            ScratchpadBinding{.scratchpad_spec_name = WTOK, .accessor_name = "wtok"});
    }

    // Distribute chunks of 32 tokens across cores. The first `extra_chunks` cores
    // get (base_chunks_per_core + 1) chunks; the remaining get base_chunks_per_core.
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t token_start = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t chunks_this_core = base_chunks_per_core + (i < extra_chunks ? 1 : 0);

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"token_start_idx", token_start}, {"num_chunks", chunks_this_core}});
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {{"token_start_idx", token_start}, {"num_chunks", chunks_this_core}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"token_start_idx", token_start}, {"num_chunks", chunks_this_core}});

        token_start += chunks_this_core * TOKENS_PER_CHUNK;
    }

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = COMBINE, .spec = combine_output.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = WEIGHT, .spec = weights.mesh_tensor().tensor_spec()},
        TensorParameter{.unique_id = OUTPUT, .spec = tensor_return_value.mesh_tensor().tensor_spec()},
    };
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(COMBINE, combine_output.mesh_tensor());
    run_args.tensor_args.emplace(WEIGHT, weights.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, tensor_return_value.mesh_tensor());
    if (use_dispatch_table_skip) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = DISPATCH_TABLE, .spec = dispatch_table_opt->mesh_tensor().tensor_spec()});
        tensor_parameters.push_back(
            TensorParameter{.unique_id = INDICES, .spec = indices_opt->mesh_tensor().tensor_spec()});
        run_args.tensor_args.emplace(DISPATCH_TABLE, dispatch_table_opt->mesh_tensor());
        run_args.tensor_args.emplace(INDICES, indices_opt->mesh_tensor());
    }
    if (row_indexed) {
        tensor_parameters.push_back(
            TensorParameter{.unique_id = ROW_INDEX, .spec = row_index_opt->mesh_tensor().tensor_spec()});
        run_args.tensor_args.emplace(ROW_INDEX, row_index_opt->mesh_tensor());
    }

    ProgramSpec spec{
        .name = "post_combine_reduce",
        .kernels = {std::move(reader_spec), std::move(compute_spec), std::move(writer_spec)},
        .dataflow_buffers = std::move(dataflow_buffers),
        .scratchpads = std::move(scratchpads),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, COMPUTE, WRITER},
                    .target_nodes = core_range_set,
                },
            },
    };

    run_args.kernel_run_args.reserve(3);
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::operations::experimental::quasar::deepseek_prefill::post_combine_reduce
