// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/reshape_view/device/reshape_tiled_rows_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>
#include <tt-metalium/workload/run_args/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

namespace {

constexpr uint32_t kTileDim = 32;
constexpr uint32_t kFaceDim = 16;

const tt::tt_metal::KernelSpecName READER{"reader"};
const tt::tt_metal::KernelSpecName WRITER{"writer"};
const tt::tt_metal::TensorParamName INPUT_T{"input"};
const tt::tt_metal::TensorParamName OUTPUT_T{"output"};

// The per-call values every kernel takes as common runtime args.
tt::tt_metal::KernelRunArgs common_args(
    const tt::tt_metal::KernelSpecName& kernel, const Tensor& input, const Tensor& output) {
    const auto& in_shape = input.logical_shape();
    const auto& out_shape = output.logical_shape();
    const uint32_t num_output_pages = output.physical_volume() / (kTileDim * kTileDim);
    return tt::tt_metal::KernelRunArgs{
        .kernel = kernel,
        .common_runtime_arg_values = {
            {"num_output_pages", num_output_pages},
            {"in_rows", static_cast<uint32_t>(in_shape[-2])},
            {"out_rows", static_cast<uint32_t>(out_shape[-2])},
        }};
}

}  // namespace

bool tiled_rows_reshape_applies(
    const Tensor& input,
    const ttnn::Shape& input_shape_3d,
    const ttnn::Shape& output_shape_3d,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool has_sub_core_grid) {
    if (input.layout() != Layout::TILE || input.memory_config().is_sharded() || output_mem_config.is_sharded() ||
        has_sub_core_grid) {
        return false;
    }
    // The dtypes the native reshape takes (validate_on_program_cache_miss); routing sends others to codegen.
    const auto dtype = input.dtype();
    if (dtype != DataType::BFLOAT16 && dtype != DataType::FLOAT32 && dtype != DataType::INT32 &&
        dtype != DataType::UINT32) {
        return false;
    }
    const auto& tile = input.tensor_spec().tile();
    if (tile.get_tile_shape() != std::array<uint32_t, 2>{kTileDim, kTileDim} ||
        tile.get_face_shape() != std::array<uint32_t, 2>{kFaceDim, kFaceDim}) {
        return false;
    }
    return input_shape_3d.rank() == 3 && output_shape_3d.rank() == 3 && input_shape_3d[-1] == output_shape_3d[-1] &&
           input_shape_3d[-2] > 0 && output_shape_3d[-2] > 0;
}

bool ReshapeViewTiledRowsProgramFactory::applies(
    const ReshapeViewParams& operation_attributes, const ReshapeViewInputs& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& out_padded = operation_attributes.padded_output_shape;
    const auto& out_shape = operation_attributes.logical_output_shape;
    // The kernels assume the default tile padding: whole tiles of rows, the same padded width.
    return tiled_rows_reshape_applies(
               input,
               input.logical_shape(),
               out_shape,
               operation_attributes.output_mem_config,
               operation_attributes.sub_core_grid.has_value()) &&
           input.padded_shape()[-1] == out_padded[-1] &&
           input.padded_shape()[-2] == tt::round_up(input.logical_shape()[-2], kTileDim) &&
           out_padded[-2] == tt::round_up(out_shape[-2], kTileDim);
}

// Algorithm: the work is split over output tiles. For each output tile, the reader loads the input tile
// of each run of consecutive rows that come from one input tile (hostdevcommon/rows_common.hpp), and the
// writer copies those rows into a scratch tile, face row block by face row block, then writes the tile
// out. Both kernels derive the runs and their tile range from the common runtime args, so the program
// runs on the whole grid at every size.
ttnn::device_operation::ProgramArtifacts ReshapeViewTiledRowsProgramFactory::create_program_artifacts(
    const ReshapeViewParams& /*operation_attributes*/,
    const ReshapeViewInputs& tensor_args,
    Tensor& tensor_return_value) {
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    auto& output = tensor_return_value;

    TT_FATAL(
        input.logical_shape().volume() == output.logical_shape().volume(), "Requested shapes are not of equal volume");

    auto* device = input.device();
    const auto grid = device->compute_with_storage_grid_size();
    const uint32_t num_cores = grid.x * grid.y;
    const CoreRangeSet all_cores(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));

    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t tile_size_bytes = tt::tile_size(data_format);
    const uint32_t tiles_per_row = input.padded_shape()[-1] / kTileDim;

    const DFBSpecName IN_TILES{"in_tiles"};
    const ScratchpadSpecName WORKING{"working"};

    ProgramSpec spec;
    spec.name = "reshape_view_tiled_rows";
    spec.dataflow_buffers = {
        DataflowBufferSpec{
            .unique_id = IN_TILES,
            .entry_size = tile_size_bytes,
            .num_entries = 2,
            .data_format_metadata = data_format,
        },
    };
    spec.scratchpads = {
        ScratchpadSpec{.unique_id = WORKING, .size_per_node = tile_size_bytes},
    };

    const KernelSpec::RuntimeArgSchema schema = {
        .runtime_arg_names = {"core_idx"}, .common_runtime_arg_names = {"num_output_pages", "in_rows", "out_rows"}};

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/reader_reshape_tiled_rows.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN_TILES, .accessor_name = "in_tiles", .endpoint_type = DFBEndpointType::PRODUCER},
            },
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_T, .accessor_name = "input"}},
        .compile_time_args =
            {
                {"Tile_size_bytes", tile_size_bytes},
                {"tiles_per_row", tiles_per_row},
                {"num_cores", num_cores},
            },
        .runtime_arg_schema = schema,
        .hw_config = create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/reshape_view/device/device/dataflow/writer_reshape_tiled_rows.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = IN_TILES, .accessor_name = "in_tiles", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .scratchpad_bindings = {ScratchpadBinding{.scratchpad_spec_name = WORKING, .accessor_name = "working"}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT_T, .accessor_name = "output"}},
        .compile_time_args =
            {
                {"Tile_size_bytes", tile_size_bytes},
                {"element_sz_bytes", tt::datum_size(data_format)},
                {"tiles_per_row", tiles_per_row},
                {"num_cores", num_cores},
            },
        .runtime_arg_schema = schema,
        .hw_config = create_writer_datamovement_config(),
    };

    spec.kernels = {reader, writer};
    // The shapes vary per call; an interleaved tiled tensor's accessor does not depend on them.
    spec.tensor_parameters = {
        TensorParameter{
            .unique_id = INPUT_T,
            .spec = input.mesh_tensor().tensor_spec(),
            .relaxations = {.dynamic_tensor_shape = true}},
        TensorParameter{
            .unique_id = OUTPUT_T,
            .spec = output.mesh_tensor().tensor_spec(),
            .relaxations = {.dynamic_tensor_shape = true}},
    };
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = {READER, WRITER}, .target_nodes = all_cores}};

    KernelRunArgs reader_kra = common_args(READER, input, output);
    KernelRunArgs writer_kra = common_args(WRITER, input, output);
    uint32_t core_idx = 0;
    for (const auto& core : corerange_to_cores(all_cores, std::nullopt)) {
        AddRuntimeArgsForNode(reader_kra.runtime_arg_values, core, {{"core_idx", core_idx}});
        AddRuntimeArgsForNode(writer_kra.runtime_arg_values, core, {{"core_idx", core_idx}});
        ++core_idx;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_kra), std::move(writer_kra)};
    run_args.tensor_args = {
        {INPUT_T, TensorArgument{input.mesh_tensor()}},
        {OUTPUT_T, TensorArgument{output.mesh_tensor()}},
    };
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::ProgramRunArgs ReshapeViewTiledRowsProgramFactory::override_runtime_arguments(
    const ReshapeViewParams& /*operation_attributes*/,
    const ReshapeViewInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    tt::tt_metal::ProgramRunArgs params;
    params.kernel_run_args = {
        common_args(READER, tensor_args.input, tensor_return_value),
        common_args(WRITER, tensor_args.input, tensor_return_value),
    };
    params.tensor_args = {
        {INPUT_T, tt::tt_metal::TensorArgument{tensor_args.input.mesh_tensor()}},
        {OUTPUT_T, tt::tt_metal::TensorArgument{tensor_return_value.mesh_tensor()}},
    };
    return params;
}

}  // namespace ttnn::prim
