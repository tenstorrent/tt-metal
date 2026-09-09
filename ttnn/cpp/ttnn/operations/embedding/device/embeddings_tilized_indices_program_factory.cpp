// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_tilized_indices_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts EmbeddingsTilizedIndicesProgramFactory::create_program_artifacts(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor_arg;
    const auto& weights = tensor_args.weight_arg;
    auto& output = tensor_return_value;
    const auto& embeddings_type = operation_attributes.embeddings_type;

    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& weights_mesh_tensor = weights.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    ////////////////////////////////////////////////////////////////////////////
    //                      Grayskull Device Setup
    ////////////////////////////////////////////////////////////////////////////
    // This should allocate a DRAM buffer on the device
    auto* device = a.device();

    ////////////////////////////////////////////////////////////////////////////
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    uint32_t input_element_size_bytes = a.element_size();
    uint32_t weights_element_size_bytes = weights.element_size();
    uint32_t output_element_size_bytes = output.element_size();

    // row major, page size is last dim
    uint32_t weight_page_size = weights.padded_shape()[-1] * weights_element_size_bytes;
    uint32_t output_page_size = output.padded_shape()[-1] * output_element_size_bytes;

    // weights shape is [1, 1, num_embeddings, num_dim]

    uint32_t batch_size = a.logical_shape()[0];  // num rows
    uint32_t num_cols = a.logical_shape()[-1];
    uint32_t volume = num_cols * batch_size;
    auto alignment = a.buffer()->alignment();

    // setup problem and grid size

    uint32_t problem_size = volume;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreSplitResult work = split_work_to_cores_aligned(compute_with_storage_grid_size, problem_size, FACE_HEIGHT);

    uint32_t num_cores = work.required_cores;
    CoreRangeSet all_cores = work.all_cores;
    CoreRangeSet core_group_1 = work.core_group_1;
    CoreRangeSet core_group_2 = work.core_group_2;
    uint32_t num_blocks_per_core_group_1 = work.units_per_core_group_1;
    uint32_t num_blocks_per_core_group_2 = work.units_per_core_group_2;

    uint32_t g1_numcores = core_group_1.num_cores();

    tt::DataFormat input_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    tt::DataFormat weights_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);

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
    spec.name = "embeddings_tilized_indices";

    // -----------------------------------------------------------------------
    // Dataflow buffers
    //
    // num_entries is the number of pages the legacy circular buffer held: its total size divided by
    // its page size.
    //
    // One buffer serves as both the reader's weight-staging area and the writer's output buffer, which
    // is what makes the two kernels a genuine producer/consumer pair over it.
    // -----------------------------------------------------------------------
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = OUTPUT,
        .entry_size = rounded_weight_page_size,
        .num_entries = 2,
        .data_format_metadata = weights_data_format,
    });

    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = INDEX_SCRATCH,
        .entry_size = FACE_HEIGHT * index_page_size,
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

    if (a.logical_shape()[-1] <= FACE_HEIGHT) {
        embedding_defines.insert({"ONLY_ONE_FACE_COLUMN", "1"});
    }

    spec.kernels.push_back(KernelSpec{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embedding_ind_tilized.cpp",
        .compiler_options = {.defines = std::move(embedding_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_PARAM, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = WEIGHTS_PARAM, .accessor_name = "weights"},
            },
        .compile_time_args =
            {
                {"weight_stick_size", weight_page_size},
                // width/length of a row
                {"row_length", num_cols},
            },
        .runtime_arg_schema =
            {.runtime_arg_names = {"tile_offset", "face_offset", "num_rows", "curr_col", "starting_index"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    });

    // -----------------------------------------------------------------------
    // Tilized writer
    // -----------------------------------------------------------------------
    spec.kernels.push_back(KernelSpec{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = OUTPUT,
                    .accessor_name = "out0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT_PARAM, .accessor_name = "dst"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"stick_size", "num_sticks", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    });

    spec.work_units.push_back(WorkUnitSpec{
        .name = "main",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    });

    // -----------------------------------------------------------------------
    // Run args
    // -----------------------------------------------------------------------
    uint32_t col_offset = 0;
    uint32_t weight_offset = 0;

    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    uint32_t row = 0;
    uint32_t tiles_per_tile_row = (num_cols + TILE_HEIGHT - 1) / TILE_HEIGHT;

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        col_offset = weight_offset % num_cols;
        row = weight_offset / num_cols;

        uint32_t local_num_blocks = i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;
        uint32_t r_f_offset = (((row % TILE_HEIGHT) / FACE_HEIGHT) * 2 * FACE_HW) + ((row % FACE_HEIGHT) * FACE_HEIGHT);
        // Offset by one face size if we are in the right half of the tile + where we are in the row
        uint32_t c_f_offset = ((col_offset % TILE_HEIGHT) / FACE_HEIGHT) * FACE_HW;
        uint32_t face_offset = r_f_offset + c_f_offset;
        uint32_t curr_tile = ((row / TILE_HEIGHT) * tiles_per_tile_row) + (col_offset / TILE_HEIGHT);

        // Reader
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"tile_offset", curr_tile},
             {"face_offset", face_offset},
             {"num_rows", local_num_blocks},
             {"curr_col", col_offset},
             // starting col in the face row; under PADDED the reader also hands this value to its
             // local weight cache as the pad token
             {"starting_index", static_cast<uint32_t>(col_offset % FACE_HEIGHT)}});

        // Writer
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"stick_size", output_page_size}, {"num_sticks", local_num_blocks}, {"start_id", weight_offset}});

        weight_offset += local_num_blocks;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    run_args.kernel_run_args.push_back(std::move(writer_run_args));
    run_args.tensor_args.emplace(INPUT_PARAM, input_mesh_tensor);
    run_args.tensor_args.emplace(WEIGHTS_PARAM, weights_mesh_tensor);
    run_args.tensor_args.emplace(OUTPUT_PARAM, output_mesh_tensor);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
