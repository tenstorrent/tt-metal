// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "embeddings_tilized_indices_program_factory.hpp"
#include "embedding_program_factory_common.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts EmbeddingsTilizedIndicesProgramFactory::create_program_artifacts(
    const EmbeddingParams& operation_attributes, const EmbeddingInputs& tensor_args, Tensor& tensor_return_value) {
    // Metal 2.0 named resource handles (function-local so the three factory TUs never collide under
    // unity build). c_0 doubles as the weights-in staging buffer and the output buffer.
    const DFBSpecName OUT_DFB{"out"};              // legacy c_0 (weights-in + output)
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
    //                      Application Setup
    ////////////////////////////////////////////////////////////////////////////

    auto* device = a.device();

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
    uint32_t num_blocks_per_core_group_1 = work.units_per_core_group_1;
    uint32_t num_blocks_per_core_group_2 = work.units_per_core_group_2;

    uint32_t g1_numcores = core_group_1.num_cores();

    // Data formats
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    tt::DataFormat weights_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(weights.dtype());

    uint32_t rounded_weight_page_size = tt::align(weight_page_size, alignment);

    ////////////////////////////////////////////////////////////////////////////
    //                 DataflowBuffers
    ////////////////////////////////////////////////////////////////////////////
    const bool has_weight_cache =
        embeddings_type == EmbeddingsType::PADDED || embeddings_type == EmbeddingsType::BINARY;

    std::vector<DataflowBufferSpec> dfbs;

    // c_0 weights-in staging + output (reader produces, writer consumes)
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = OUT_DFB,
        .entry_size = rounded_weight_page_size,
        .num_entries = 2,
        .data_format_metadata = weights_cb_data_format,
    });

    // c_1 index scratch (single toucher, reader-only)
    uint32_t index_page_size = round_up_to_mul32(input_element_size_bytes);
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IDX_DFB,
        .entry_size = FACE_HEIGHT * index_page_size,
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
    if (a.logical_shape()[-1] <= FACE_HEIGHT) {
        embedding_defines.emplace("ONLY_ONE_FACE_COLUMN", "1");
    }

    // reader
    Group<DFBBinding> reader_dfb_bindings;
    reader_dfb_bindings.push_back(
        DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER});
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

    Group<std::string> reader_rta_names = {"tile_offset", "face_offset", "num_rows", "curr_col", "starting_index"};
    if (embeddings_type == EmbeddingsType::PADDED) {
        reader_rta_names.push_back("pad_token");
    }

    KernelSpec reader{
        .unique_id = READER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/operations/embedding/device/kernels/dataflow/embedding_ind_tilized.cpp"},
        .compiler_options = {.defines = std::move(embedding_defines)},
        .dfb_bindings = std::move(reader_dfb_bindings),
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
             TensorBinding{.tensor_parameter_name = WEIGHTS, .accessor_name = "weights"}},
        .compile_time_args =
            {{"weight_stick_size", weight_page_size}, {"row_length", static_cast<uint32_t>(a.logical_shape()[-1])}},
        .runtime_arg_schema = {.runtime_arg_names = std::move(reader_rta_names)},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // writer: Metal 2.0 fork of the shared stick-layout writer.
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            std::filesystem::path{
                "ttnn/cpp/ttnn/kernel/dataflow/writer_unary_stick_layout_interleaved_start_id_metal2.cpp"},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"stick_size", "num_sticks", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                 Runtime args (per node)
    ////////////////////////////////////////////////////////////////////////////
    auto cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);

    uint32_t col_offset = 0;
    uint32_t weight_offset = 0;
    uint32_t row = 0;
    uint32_t tiles_per_tile_row = (num_cols + TILE_HEIGHT - 1) / TILE_HEIGHT;

    KernelRunArgs reader_kra{.kernel = READER};
    KernelRunArgs writer_kra{.kernel = WRITER};

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
            reader_kra.runtime_arg_values,
            core,
            {{"tile_offset", curr_tile},
             {"face_offset", face_offset},
             {"num_rows", local_num_blocks},
             {"curr_col", col_offset},
             {"starting_index", static_cast<uint32_t>(col_offset % FACE_HEIGHT)}});  // starting col in the face row
        if (embeddings_type == EmbeddingsType::PADDED) {
            AddRuntimeArgsForNode(reader_kra.runtime_arg_values, core, {{"pad_token", pad_token.value()}});
        }

        // Writer
        AddRuntimeArgsForNode(
            writer_kra.runtime_arg_values,
            core,
            {{"stick_size", static_cast<uint32_t>(output_page_size)},
             {"num_sticks", local_num_blocks},
             {"start_id", weight_offset}});

        weight_offset += local_num_blocks;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                 Assemble
    ////////////////////////////////////////////////////////////////////////////
    ProgramSpec spec{
        .name = "embedding_tilized_indices",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = input_mt.tensor_spec()},
             TensorParameter{.unique_id = WEIGHTS, .spec = weights_mt.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output_mt.tensor_spec()}},
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_kra));
    run_args.kernel_run_args.push_back(std::move(writer_kra));
    run_args.tensor_args = {{INPUT, input_mt}, {WEIGHTS, weights_mt}, {OUTPUT, output_mt}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
