// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_fused_runtime_rows_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
// The one common runtime word, 32-index blocks per index row (S / 32). Every kernel derives its
// per-core block range from it and a static core index.
constexpr const char* kLengthArg = "blocks_per_batch";

uint32_t fused_blocks_per_batch(const EmbeddingInputs& tensor_args) {
    return tensor_args.input_tensor_arg.padded_shape()[-1] / TILE_HEIGHT;
}
}  // namespace

ttnn::device_operation::ProgramArtifacts EmbeddingsFusedRuntimeRowsProgramFactory::create_program_artifacts(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;
    auto& output = tensor_return_value;
    const auto& embeddings_type = operation_attributes.embeddings_type;
    const auto& pad_token = operation_attributes.pad_token;

    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& weights_mesh_tensor = weights.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    MeshDevice* device = a.device();

    TT_FATAL(
        !is_sharded(output.buffer()->buffer_layout()),
        "The runtime-rows embedding program factory serves interleaved outputs only");

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();

    // row major, page size is last dim
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]
    uint32_t batch_size = a.padded_shape()[0];
    uint32_t num_blocks_per_batch = fused_blocks_per_batch(tensor_args);
    uint32_t num_tiles_per_block = weights.padded_shape()[-1] / TILE_WIDTH;

    // The full grid: every core derives its block range on device from the common row count.
    auto grid = device->compute_with_storage_grid_size();
    CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = all_cores.num_cores();

    EmbeddingsIndexType embeddings_index_type;
    if (a.dtype() == DataType::BFLOAT16) {
        embeddings_index_type = EmbeddingsIndexType::BFP16;
    } else {
        embeddings_index_type = EmbeddingsIndexType::UINT32;
    }

    tt::DataFormat weights_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());
    uint32_t weights_single_tile_size = tt::tile_size(weights_data_format);
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_data_format);

    // Hardcoded limit to reduce L1 usage. Should be updated to be tuned based on overall L1 usage
    constexpr uint32_t max_double_buffer_tiles = 64;

    constexpr uint32_t max_l1_budget_bytes = 1024 * 1024;  // 1MB budget for the embedding weights staging buffer
    uint32_t max_tiles_per_chunk = std::min(max_l1_budget_bytes / weights_single_tile_size, num_tiles_per_block);
    max_tiles_per_chunk = std::max(max_tiles_per_chunk, 1U);

    uint32_t required_memory_bytes = 2 * num_tiles_per_block * weights_single_tile_size;
    bool use_chunked_processing = required_memory_bytes > max_l1_budget_bytes;

    // PADDED and BINARY serve some weight rows out of a locally cached copy instead of fetching them
    // per token; the other embeddings types have no such rows, so the cache is absent for them.
    const bool use_local_cache = embeddings_type == EmbeddingsType::PADDED || embeddings_type == EmbeddingsType::BINARY;
    // PADDED caches the single pad row; BINARY caches rows 0 and 1.
    const uint32_t cache_entries = embeddings_type == EmbeddingsType::PADDED ? 1u : 2u;

    // For very large embeddings, use chunked processing
    uint32_t tiles_per_chunk;
    uint32_t num_chunks;
    uint32_t last_chunk_tiles;
    uint32_t buffering;
    // One pass has to push exactly this many tiles.
    uint32_t dfb_num_entries;

    if (use_chunked_processing) {
        const uint32_t chunk_cap = std::min(max_tiles_per_chunk, max_double_buffer_tiles);
        const uint32_t row_bytes = num_tiles_per_block * weights_single_tile_size;
        const uint32_t output_row_bytes = num_tiles_per_block * output_single_tile_size;
        // A full row costs one staging buffer plus one output row; both have to fit in the L1 left
        // after the buffers that are always present.
        const uint32_t l1_alignment = device->allocator()->get_alignment(BufferType::L1);
        // The index scratch and weight cache are scratchpads stacked after the DFBs, each start aligned
        // to the DRAM alignment (ProgramImpl::allocate_scratchpads).
        const uint32_t sp_alignment = device->allocator()->get_alignment(BufferType::DRAM);
        const uint64_t other_l1_bytes =
            sp_alignment +  // gap from the end of the DFBs to the first scratchpad
            round_up(TILE_HEIGHT * input_element_size_bytes, sp_alignment) +
            (use_local_cache ? round_up(cache_entries * round_up_to_mul32(weight_page_size), sp_alignment) : 0);
        const uint64_t l1_capacity =
            device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
        const uint64_t full_row_dfb_bytes =
            round_up(row_bytes, l1_alignment) + round_up(output_row_bytes, l1_alignment);
        if (full_row_dfb_bytes + other_l1_bytes <= l1_capacity) {
            // One buffer holds the row, so the short last chunk lands on the end.
            tiles_per_chunk = std::min(chunk_cap, num_tiles_per_block);
            num_chunks = (num_tiles_per_block + tiles_per_chunk - 1) / tiles_per_chunk;
            last_chunk_tiles = num_tiles_per_block - (num_chunks - 1) * tiles_per_chunk;
            buffering = 1;
            dfb_num_entries = num_tiles_per_block;
        } else {
            // Chunk size divides the row, so every push matches the buffer.
            tiles_per_chunk = chunk_cap;
            for (uint32_t divisor = chunk_cap; divisor > 0; --divisor) {
                if (num_tiles_per_block % divisor == 0) {
                    tiles_per_chunk = divisor;
                    break;
                }
            }
            num_chunks = num_tiles_per_block / tiles_per_chunk;
            last_chunk_tiles = tiles_per_chunk;
            buffering = 2 * tiles_per_chunk * weights_single_tile_size <= max_l1_budget_bytes ? 2 : 1;
            dfb_num_entries = buffering * tiles_per_chunk;
        }
    } else {
        // Use original non-chunked approach for smaller embeddings
        tiles_per_chunk = num_tiles_per_block;
        num_chunks = 1;
        last_chunk_tiles = num_tiles_per_block;
        buffering = num_tiles_per_block > max_double_buffer_tiles ? 1 : 2;
        dfb_num_entries = buffering * tiles_per_chunk;
    }

    // -----------------------------------------------------------------------
    // Resource names
    // -----------------------------------------------------------------------
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const DFBSpecName WEIGHTS_STAGING{"weights_staging"};
    // The reader's private page for the block of indices it is working through.
    const ScratchpadSpecName INDEX_SCRATCH_SP{"index_scratch_sp"};
    const DFBSpecName OUTPUT{"output"};
    // The reader's private copy of the weight rows it serves locally: it fills it and reads tokens
    // back out of it, both over the NoC.
    const ScratchpadSpecName WEIGHT_CACHE_SP{"weight_cache_sp"};

    const TensorParamName INPUT_PARAM{"input"};
    const TensorParamName WEIGHTS_PARAM{"weights"};
    const TensorParamName OUTPUT_PARAM{"output"};

    ProgramSpec spec;
    spec.name = "embeddings_fused_runtime_rows";

    // -----------------------------------------------------------------------
    // Dataflow buffers
    // -----------------------------------------------------------------------
    // The reader gathers each block's weight rows (one chunk at a time) into this row-major staging
    // buffer; the compute kernel tilizes them out of it into the output buffer the writer drains.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WEIGHTS_STAGING,
        .entry_size = weights_single_tile_size,
        .num_entries = dfb_num_entries,
        .data_format_metadata = weights_data_format,
    });
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT,
        .entry_size = output_single_tile_size,
        .num_entries = dfb_num_entries,
        .data_format_metadata = output_data_format,
    });

    Group<ScratchpadSpec> scratchpads;
    scratchpads.push_back(
        ScratchpadSpec{.unique_id = INDEX_SCRATCH_SP, .size_per_node = TILE_HEIGHT * input_element_size_bytes});
    if (use_local_cache) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        scratchpads.push_back(
            ScratchpadSpec{.unique_id = WEIGHT_CACHE_SP, .size_per_node = cache_entries * cache_page_size});
    }
    // The reader's index scratchpad, plus its weight cache under PADDED / BINARY.
    spec.scratchpads = std::move(scratchpads);

    // TODO: Can increase size for larger reads
    uint32_t input_block_size_bytes = TILE_HEIGHT * input_element_size_bytes;

    // -----------------------------------------------------------------------
    // Tensor parameters
    //
    // The hash drops the index length and the table rows, so the index page size (a row-major width)
    // rides a runtime word, while the table keeps its fixed width as a static page size.
    // -----------------------------------------------------------------------
    const TensorSpecRelaxations index_relaxations{.dynamic_tensor_shape = true};
    const TensorSpecRelaxations table_relaxations{.dynamic_tensor_shape = true, .match_page_size = true};
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = INPUT_PARAM, .spec = input_mesh_tensor.tensor_spec(), .relaxations = index_relaxations});
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = WEIGHTS_PARAM, .spec = weights_mesh_tensor.tensor_spec(), .relaxations = table_relaxations});
    spec.tensor_parameters.push_back(TensorParameter{
        .unique_id = OUTPUT_PARAM, .spec = output_mesh_tensor.tensor_spec(), .relaxations = index_relaxations});

    // Split inputs of every kernel: static per-core index, common block count.
    const KernelSpec::RuntimeArgSchema split_runtime_arg_schema = {
        .runtime_arg_names = {"core_idx"}, .common_runtime_arg_names = {kLengthArg}};

    // -----------------------------------------------------------------------
    // Reader
    // -----------------------------------------------------------------------
    Group<DFBBinding> reader_dfb_bindings;
    reader_dfb_bindings.push_back(DFBBinding{
        .dfb_spec_name = WEIGHTS_STAGING,
        .accessor_name = "in0",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    // The index scratch page never leaves the reader: it decodes indices out of it, so it is the
    // reader's private scratchpad.
    Group<KernelSpec::ScratchpadBinding> reader_scratchpad_bindings;
    reader_scratchpad_bindings.push_back(
        KernelSpec::ScratchpadBinding{.scratchpad_spec_name = INDEX_SCRATCH_SP, .accessor_name = "in1"});
    if (use_local_cache) {
        // The weight cache: the reader fills it and reads tokens back out of it, with no hand-off to
        // another kernel, so it is the reader's private scratchpad too.
        reader_scratchpad_bindings.push_back(
            KernelSpec::ScratchpadBinding{.scratchpad_spec_name = WEIGHT_CACHE_SP, .accessor_name = "local_cache"});
    }

    // These defines and the weight cache's scratchpad binding share one condition, the embeddings type:
    // the reader names the cache handle only on the builds where the host binds it.
    KernelSpec::CompilerOptions::Defines embedding_defines{
        {enchantum::to_string(embeddings_type).data(), "1"},
        {enchantum::to_string(embeddings_index_type).data(), "1"},
    };

    KernelSpec::RuntimeArgSchema reader_runtime_arg_schema = split_runtime_arg_schema;
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_runtime_arg_schema.runtime_arg_names.push_back("pad_token");
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize_runtime_rows.cpp",
        .compiler_options = {.defines = embedding_defines},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .scratchpad_bindings = std::move(reader_scratchpad_bindings),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = WEIGHTS_PARAM, .accessor_name = "weights"},
            },
        .compile_time_args =
            {
                {"weight_block_size", weight_page_size},
                {"tiles_per_chunk", tiles_per_chunk},
                {"input_block_size_bytes", input_block_size_bytes},
                {"num_chunks", num_chunks},
                {"last_chunk_tiles", last_chunk_tiles},
                {"batch", batch_size},
                {"num_cores", num_cores},
            },
        .runtime_arg_schema = std::move(reader_runtime_arg_schema),
        .hw_config = ttnn::create_reader_datamovement_config(),
    });

    // -----------------------------------------------------------------------
    // Compute
    // -----------------------------------------------------------------------
    // Legacy compute config left every field at its default. ComputeHardwareConfig's common defaults
    // reproduce them (HiFi4, precise SFPU, 16-bit dest, double-buffered dest, no unpack-mode entries)
    // on every generation. A TT-1.x.x-only extra is unused on TT-2.x.x.
    ComputeHardwareConfig compute_hw{};

    spec.kernels.push_back(KernelSpec{
        .unique_id = COMPUTE,
        .source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/compute/tilize_chunked_runtime_rows.cpp",
        // O3 explicitly: Metal 2.0's type-agnostic CompilerOptions defaults to O2.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = WEIGHTS_STAGING,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = OUTPUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args =
            {
                {"tiles_per_chunk", tiles_per_chunk},
                {"num_chunks", num_chunks},
                {"last_chunk_tiles", last_chunk_tiles},
                {"batch", batch_size},
                {"num_cores", num_cores},
            },
        .runtime_arg_schema = split_runtime_arg_schema,
        .hw_config = compute_hw,
    });

    // -----------------------------------------------------------------------
    // Writer
    // -----------------------------------------------------------------------
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize_writer_runtime_rows.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "dst"},
            },
        .compile_time_args =
            {
                {"num_tiles_per_block", num_tiles_per_block},
                {"batch", batch_size},
                {"num_cores", num_cores},
            },
        .runtime_arg_schema = split_runtime_arg_schema,
        .hw_config = ttnn::create_writer_datamovement_config(),
    });

    // One work unit over the full grid: the three kernels agree on the split by construction.
    spec.work_units.push_back(WorkUnitSpec{
        .name = "all_cores",
        .kernels = {READER, WRITER, COMPUTE},
        .target_nodes = all_cores,
    });

    // -----------------------------------------------------------------------
    // Run args
    // -----------------------------------------------------------------------
    auto cores = corerange_to_cores(all_cores, std::nullopt, /*row_wise=*/false);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    reader_run_args.common_runtime_arg_values = {{kLengthArg, num_blocks_per_batch}};
    writer_run_args.common_runtime_arg_values = {{kLengthArg, num_blocks_per_batch}};
    compute_run_args.common_runtime_arg_values = {{kLengthArg, num_blocks_per_batch}};
    for (uint32_t i = 0; i < cores.size(); ++i) {
        AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, cores[i], {{"core_idx", i}});
        if (embeddings_type == EmbeddingsType::PADDED) {
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, cores[i], {{"pad_token", pad_token.value()}});
        }
        AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, cores[i], {{"core_idx", i}});
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, cores[i], {{"core_idx", i}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.kernel_run_args.push_back(std::move(compute_run_args));
    run_args.tensor_args.emplace(INPUT_PARAM, input_mesh_tensor);
    run_args.tensor_args.emplace(WEIGHTS_PARAM, weights_mesh_tensor);
    run_args.tensor_args.emplace(OUTPUT_PARAM, output_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// A hit re-binds the three tensors and rewrites the common block count the three kernels derive
// their per-core ranges from (the hash leaves the index length out).
tt::tt_metal::experimental::ProgramRunArgs EmbeddingsFusedRuntimeRowsProgramFactory::override_runtime_arguments(
    const EmbeddingParams& /*operation_attributes*/,
    const EmbeddingInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    ProgramRunArgs params;
    const uint32_t blocks_per_batch = fused_blocks_per_batch(tensor_args);
    for (const char* kernel : {"reader", "writer", "compute"}) {
        params.kernel_run_args.push_back(KernelRunArgs{
            .kernel = KernelSpecName{kernel}, .common_runtime_arg_values = {{kLengthArg, blocks_per_batch}}});
    }
    params.tensor_args.emplace(TensorParamName{"input"}, tensor_args.input_tensor_arg.mesh_tensor());
    params.tensor_args.emplace(TensorParamName{"weights"}, tensor_args.weight_arg.mesh_tensor());
    params.tensor_args.emplace(TensorParamName{"output"}, tensor_return_value.mesh_tensor());
    return params;
}

}  // namespace ttnn::prim
