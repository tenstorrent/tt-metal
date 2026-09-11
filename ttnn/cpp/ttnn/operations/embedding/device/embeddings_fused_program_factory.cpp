// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_fused_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts EmbeddingsFusedProgramFactory::create_program_artifacts(
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

    // Create Buffers
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

    // PADDED and BINARY serve some weight rows out of a locally cached copy instead of fetching them
    // per token; the other embeddings types have no such rows, so the cache is absent for them.
    const bool use_local_cache = embeddings_type == EmbeddingsType::PADDED || embeddings_type == EmbeddingsType::BINARY;

    // -----------------------------------------------------------------------
    // Resource names
    // -----------------------------------------------------------------------
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_group_1"};
    const KernelSpecName COMPUTE_G2{"compute_group_2"};

    const DFBSpecName WEIGHTS_STAGING{"weights_staging"};
    const DFBSpecName OUTPUT{"output"};
    // Reader-private staging regions (formerly reader self-loop DFBs; see the ScratchpadSpecs below).
    const ScratchpadSpecName INDEX_SCRATCH{"index_scratch"};
    const ScratchpadSpecName WEIGHT_CACHE{"weight_cache"};

    const TensorParamName INPUT_PARAM{"input"};
    const TensorParamName WEIGHTS_PARAM{"weights"};
    const TensorParamName OUTPUT_PARAM{"output"};

    ProgramSpec spec;
    spec.name = "embeddings_fused";

    // -----------------------------------------------------------------------
    // Dataflow buffers
    // -----------------------------------------------------------------------
    // The reader gathers each block's weight rows (one chunk at a time) into this row-major staging
    // buffer; the compute kernel tilizes them out of it into the output buffer.
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = WEIGHTS_STAGING,
        .entry_size = weights_single_tile_size,
        .num_entries = buffering * tiles_per_chunk,
        .data_format_metadata = weights_data_format,
    });

    // The reader's private page for the block of indices it is working through. Nothing hands it off
    // or synchronizes on it, so it is a scratchpad rather than a DFB. (As a DFB it was bound by the
    // reader as both PRODUCER and CONSUMER -- a DM self-loop, legal on Gen1 but rejected by the spec
    // validator on Gen2/Quasar, and its UInt32 format has no Quasar device format.) size_per_node is
    // the former DFB's whole allocation, entry_size * num_entries.
    spec.scratchpads.push_back(ScratchpadSpec{
        .unique_id = INDEX_SCRATCH,
        .size_per_node = TILE_HEIGHT * input_element_size_bytes * 1,
    });

    uint32_t output_dfb_total_size;
    if (output_sharded) {
        output_dfb_total_size = output.buffer()->aligned_size_per_bank();
    } else {
        output_dfb_total_size = buffering * tiles_per_chunk * output_single_tile_size;
    }
    // The output buffer's total size has to divide evenly by its tile-sized entry. When the output is
    // sharded the total is the shard's own aligned size per bank, which the op's validation makes a
    // whole number of tiles; assert it here so a disagreement fails loudly at construction rather than
    // as a framework rejection of the borrowed buffer's size.
    TT_FATAL(
        output_dfb_total_size % output_single_tile_size == 0,
        "Embedding output buffer size {} B must be divisible by its tile size {} B",
        output_dfb_total_size,
        output_single_tile_size);
    DataflowBufferSpec out_dfb{
        .unique_id = OUTPUT,
        .entry_size = output_single_tile_size,
        .num_entries = output_dfb_total_size / output_single_tile_size,
        .data_format_metadata = output_data_format,
    };
    if (output_sharded) {
        // The output buffer *is* the output shard: it is built on the output tensor's own SRAM, so the
        // tiles compute packs land in place and no writer kernel is created below.
        out_dfb.borrowed_from = OUTPUT_PARAM;
    }
    spec.dataflow_buffers.push_back(std::move(out_dfb));

    if (use_local_cache) {
        uint32_t cache_page_size = round_up_to_mul32(weight_page_size);
        // PADDED caches the single pad row; BINARY caches rows 0 and 1. The reader fills the cache and
        // replays tokens out of it with no hand-off to another kernel, so like the index page it is a
        // scratchpad (formerly a reader self-loop DFB of entry_size cache_page_size).
        spec.scratchpads.push_back(ScratchpadSpec{
            .unique_id = WEIGHT_CACHE,
            .size_per_node = cache_page_size * ((embeddings_type == EmbeddingsType::PADDED) ? 1u : 2u),
        });
    }
    uint32_t weight_block_size;
    if (output_sharded) {
        weight_block_size = output.shard_spec().value().shape[1] * weights_element_size_bytes;
    } else {
        weight_block_size = weight_page_size;
    }

    // TODO: Can increase size for larger reads
    uint32_t input_block_size_bytes = TILE_HEIGHT * input_element_size_bytes;

    // -----------------------------------------------------------------------
    // Tensor parameters
    //
    // The output parameter is declared in every configuration: the writer binds it when the output is
    // interleaved, and the borrowed output buffer resolves its address from it when the output is
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
        .dfb_spec_name = WEIGHTS_STAGING,
        .accessor_name = "in0",
        .endpoint_type = DFBEndpointType::PRODUCER,
    });
    // The index page never leaves the reader (see the ScratchpadSpec above).
    Group<ScratchpadBinding> reader_scratchpad_bindings;
    reader_scratchpad_bindings.push_back(ScratchpadBinding{
        .scratchpad_spec_name = INDEX_SCRATCH,
        .accessor_name = "indices",
    });
    if (use_local_cache) {
        // Likewise the weight cache: the reader fills it and reads tokens back out of it, with no
        // hand-off to another kernel.
        reader_scratchpad_bindings.push_back(ScratchpadBinding{
            .scratchpad_spec_name = WEIGHT_CACHE,
            .accessor_name = "local_cache",
        });
    }

    // These defines and the weight cache's scratchpad binding share one condition, the embeddings type.
    // That is what lets the reader name the cache handle at all: a scratch:: handle exists only on the
    // builds where the host binds it, so the reader's reference to it is compiled out under the same
    // defines on the builds where it is not.
    KernelSpec::CompilerOptions::Defines embedding_defines{
        {enchantum::to_string(embeddings_type).data(), "1"},
        {enchantum::to_string(embeddings_index_type).data(), "1"},
    };

    Group<std::string> reader_rta_names = {"input_start_id", "input_start_offset", "weight_offset", "num_blocks"};
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_rta_names.push_back("pad_token");
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embeddings_tilize.cpp",
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
                {"input_page_size", input_page_size},
                {"weight_block_size", weight_block_size},
                {"tiles_per_chunk", tiles_per_chunk},
                {"input_block_size_bytes", input_block_size_bytes},
                {"num_chunks", num_chunks},
                {"last_chunk_tiles", last_chunk_tiles},
            },
        .runtime_arg_schema = {.runtime_arg_names = std::move(reader_rta_names)},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // -----------------------------------------------------------------------
    // Compute
    //
    // Split across the two core groups, each with its own per_core_block_cnt compile-time arg, so the
    // two groups are two KernelSpecs of the same source rather than one spec with the count demoted to
    // a runtime arg. The source itself is config-selected: the chunked kernel when a block's tiles do
    // not fit the L1 budget in one go, the shared tilize kernel otherwise.
    // -----------------------------------------------------------------------
    const char* compute_kernel_path =
        use_chunked_processing ? "ttnn/cpp/ttnn/operations/embedding/device/kernels/compute/tilize_chunked.cpp"
                               : "ttnn/cpp/ttnn/kernel/compute/tilize_metal2.cpp";

    // Legacy compute config left every field at its default; ComputeGen1Config's defaults reproduce
    // them exactly (HiFi4, precise SFPU, 16-bit dest, double-buffered dest, no unpack-mode entries).
    ComputeHardwareConfig compute_hw = ComputeGen1Config{};
    // Gen2 (Quasar): KernelSpec::hw_config must hold the target generation's alternative -- the spec
    // validator rejects a ComputeGen1Config on Quasar. The Gen1 config sets no field, so the Gen2 one
    // is default-constructed too (same HiFi4 / precise SFPU / 16-bit dest / double-buffered dest; the
    // Gen2-only enable_2x_src_register is left at its default). WH/BH take the Gen1 config unchanged.
    if (device->arch() == tt::ARCH::QUASAR) {
        // TODO(#52269): Quasar unpack_modes are copied from Gen1 and not yet optimized for Quasar.
        compute_hw = ComputeGen2Config{};
    }

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t per_core_block_cnt) {
        Group<DFBBinding> compute_dfb_bindings;
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = WEIGHTS_STAGING,
            .accessor_name = "in",
            .endpoint_type = DFBEndpointType::CONSUMER,
        });
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = OUTPUT,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::PRODUCER,
        });
        if (output_sharded) {
            // With no writer kernel the compute kernel is the output buffer's only endpoint: it packs
            // tiles straight into the output shard and nothing drains them, so it holds both roles.
            compute_dfb_bindings.push_back(DFBBinding{
                .dfb_spec_name = OUTPUT,
                .accessor_name = "out",
                .endpoint_type = DFBEndpointType::CONSUMER,
            });
        }

        KernelSpec::CompileTimeArgs compute_compile_time_args;
        if (use_chunked_processing) {
            compute_compile_time_args = {
                {"per_core_block_cnt", per_core_block_cnt},
                {"tiles_per_chunk", tiles_per_chunk},
                {"num_chunks", num_chunks},
                {"last_chunk_tiles", last_chunk_tiles},
            };
        } else {
            // The shared tilize kernel processes each block as one chunk of tiles_per_chunk tiles.
            compute_compile_time_args = {
                {"per_core_block_cnt", per_core_block_cnt},
                {"per_core_block_tile_cnt", tiles_per_chunk},
            };
        }

        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel_path,
            // O3 explicitly: the legacy compute config set no opt_level and so resolved to O3, but
            // Metal 2.0's type-agnostic CompilerOptions defaults to O2. Leaving it unset would drop a
            // level on both compute specs' compile and link.
            .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = std::move(compute_dfb_bindings),
            .compile_time_args = std::move(compute_compile_time_args),
            .hw_config = compute_hw,
        };
    };

    // -----------------------------------------------------------------------
    // Writer
    //
    // A sharded output needs no writer: compute has already packed its tiles into the output shard.
    // TODO: We can use the second risc to do more work in parallel
    // -----------------------------------------------------------------------
    if (!output_sharded) {
        // Tilized writer
        spec.kernels.push_back(KernelSpec{
            .unique_id = WRITER,
            .source = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                      "writer_unary_interleaved_start_id_metal2.cpp",
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
            .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
            .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        });
    }

    // -----------------------------------------------------------------------
    // Work units: one per core group, each running the data-movement kernels together with that
    // group's compute spec. The two groups' node sets are disjoint and together cover all_cores.
    // -----------------------------------------------------------------------
    Group<KernelSpecName> data_movement_kernels = {READER};
    if (!output_sharded) {
        data_movement_kernels.push_back(WRITER);
    }

    if (num_blocks_per_core_group_1 > 0) {
        spec.kernels.push_back(make_compute(COMPUTE_G1, num_blocks_per_core_group_1));
        Group<KernelSpecName> group_1_kernels = data_movement_kernels;
        group_1_kernels.push_back(COMPUTE_G1);
        spec.work_units.push_back(WorkUnitSpec{
            .name = "core_group_1",
            .kernels = std::move(group_1_kernels),
            .target_nodes = core_group_1,
        });
    }

    if (num_blocks_per_core_group_2 > 0) {
        spec.kernels.push_back(make_compute(COMPUTE_G2, num_blocks_per_core_group_2));
        Group<KernelSpecName> group_2_kernels = data_movement_kernels;
        group_2_kernels.push_back(COMPUTE_G2);
        spec.work_units.push_back(WorkUnitSpec{
            .name = "core_group_2",
            .kernels = std::move(group_2_kernels),
            .target_nodes = core_group_2,
        });
    }

    // -----------------------------------------------------------------------
    // Run args
    // -----------------------------------------------------------------------
    auto cores = corerange_to_cores(all_cores, std::nullopt, row_major);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    uint32_t input_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;

        // Reader
        {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
                {{"input_start_id", input_offset / num_blocks_per_batch},
                 {"input_start_offset", input_offset % num_blocks_per_batch * input_block_size_bytes},
                 {"weight_offset", weight_offset},
                 {"num_blocks", local_num_blocks}});
            if (embeddings_type == EmbeddingsType::PADDED) {
                AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"pad_token", pad_token.value()}});
            }
        }

        // Writer
        if (!output_sharded) {
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
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
