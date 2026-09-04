// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include <optional>
#include <span>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Spec-scope names. Prefixed because program-factory translation units are unity-built: the
// anonymous namespaces of every factory in this target merge into one scope.
const KernelSpecName TTA_READER{"reader"};
const KernelSpecName TTA_WRITER{"writer"};
const DFBSpecName TTA_TILES_DFB{"tiles"};
const DFBSpecName TTA_INDEX_DFB{"index"};
const TensorParamName TTA_INPUT{"input"};
const TensorParamName TTA_START{"start"};
const TensorParamName TTA_END{"end"};
const TensorParamName TTA_OUTPUT{"output"};

// Per-node work split, resolved identically on the cache-miss and cache-hit paths.
struct TensorArgsWorkSplit {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t num_tiles_per_core_group_1 = 0;
    uint32_t num_tiles_per_core_group_2 = 0;
    uint32_t num_dims = 0;
    // Per-dimension tile counts, indexed innermost-first.
    std::vector<uint32_t> num_unpadded_tiles_per_dim;
    std::vector<uint32_t> num_padded_tiles_per_dim;
    std::vector<uint32_t> accumulated_total_per_dim;
};

TensorArgsWorkSplit resolve_tensor_args_work_split(
    const SliceParams& args, const ttnn::Tensor& input_tensor, const ttnn::Tensor& output) {
    tt::tt_metal::IDevice* device = input_tensor.device();
    const uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    const std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    const uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    const uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    const uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    const uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    const uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    const uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    TensorArgsWorkSplit ws{
        .all_cores = all_cores,
        .cores = corerange_to_cores(all_cores),
        .core_group_1 = core_group_1,
        .core_group_2 = core_group_2,
        .num_tiles_per_core_group_1 = num_tiles_per_core_group_1,
        .num_tiles_per_core_group_2 = num_tiles_per_core_group_2,
        .num_dims = num_dims,
        .num_unpadded_tiles_per_dim = std::vector<uint32_t>(num_dims),
        .num_padded_tiles_per_dim = std::vector<uint32_t>(num_dims),
        .accumulated_total_per_dim = std::vector<uint32_t>(num_dims),
    };

    ws.accumulated_total_per_dim[0] = num_total_Xt;
    ws.accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;
    ws.num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    ws.num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    ws.num_padded_tiles_per_dim[0] = num_padded_Xt;
    ws.num_padded_tiles_per_dim[1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        const uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        const uint32_t num_total_dim = input_shape[-(i + 1)];
        ws.num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        ws.num_padded_tiles_per_dim[i] = (num_total_dim - num_unpadded_dim) * ws.accumulated_total_per_dim[i - 1];
        ws.accumulated_total_per_dim[i] = num_total_dim * ws.accumulated_total_per_dim[i - 1];
    }
    return ws;
}

// Per-node values for both kernels: the reader's start_id / num_tiles plus its per-dimension index
// vector, and the writer's num_pages / start_id. A node in neither core group is a no-op node; it is
// still part of both kernels' node set, so it must be given values, and it gets a page count of zero.
//
// One function serves both the cache-miss and the cache-hit path, which must agree exactly or a
// cache hit leaves stale per-node scalars behind (#52651).
void add_tensor_args_per_node_run_args(
    const TensorArgsWorkSplit& ws, KernelRunArgs& reader_run_args, KernelRunArgs& writer_run_args) {
    // The legacy factory pins this to zero: the real start offset for this factory is computed on
    // device from the start/end index tensors, which the host cannot see.
    constexpr uint32_t start_offset = 0;

    uint32_t num_tiles_written = 0;
    for (const auto& core : ws.cores) {
        uint32_t num_tiles_per_core = 0;
        bool active = true;
        if (ws.core_group_1.contains(core)) {
            num_tiles_per_core = ws.num_tiles_per_core_group_1;
        } else if (ws.core_group_2.contains(core)) {
            num_tiles_per_core = ws.num_tiles_per_core_group_2;
        } else {
            active = false;
        }

        uint32_t start_id = 0;
        std::vector<uint32_t> id_per_dim(ws.num_dims, 0);
        if (active) {
            id_per_dim[0] = num_tiles_written % ws.num_unpadded_tiles_per_dim[0];
            uint32_t unpadded_written = num_tiles_written / ws.num_unpadded_tiles_per_dim[0];
            start_id = id_per_dim[0] + start_offset;
            for (uint32_t j = 1; j < ws.num_dims; ++j) {
                id_per_dim[j] = unpadded_written % ws.num_unpadded_tiles_per_dim[j];
                unpadded_written = unpadded_written / ws.num_unpadded_tiles_per_dim[j];
                start_id += id_per_dim[j] * ws.accumulated_total_per_dim[j - 1];
            }
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"start_id", start_id}, {"num_tiles", num_tiles_per_core}});
        reader_run_args.advanced_options.runtime_varargs[core] = std::move(id_per_dim);

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_written}});

        if (active) {
            num_tiles_written += num_tiles_per_core;
        }
    }
}

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceTileTensorArgsProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& start_tensor = tensor_args.start_tensor.value();
    const auto& end_tensor = tensor_args.end_tensor.value();

    const auto& input = input_tensor.mesh_tensor();
    const auto& start = start_tensor.mesh_tensor();
    const auto& end = end_tensor.mesh_tensor();
    const auto& out = output.mesh_tensor();

    tt::tt_metal::IDevice* device = input_tensor.device();

    TT_FATAL(input_tensor.buffer() != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(start_tensor.buffer() != nullptr, "Start buffer should be allocated on device!");
    TT_FATAL(end_tensor.buffer() != nullptr, "End buffer should be allocated on device!");
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(dfb_data_format);
    constexpr uint32_t num_input_tiles = 2;

    const TensorArgsWorkSplit ws = resolve_tensor_args_work_split(args, input_tensor, output);

    const auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_width = tile_shape[1];
    const uint32_t tile_height = tile_shape[0];

    // --- Dataflow buffers ---
    // Staging FIFO between the reader and the writer.
    DataflowBufferSpec tiles_dfb{
        .unique_id = TTA_TILES_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };
    // Single-entry scratch the reader stages the start and end index tensors through, one at a time.
    // The reader is its only toucher and drives both FIFO roles on it, so it is bound as producer and
    // consumer of this buffer -- see its bindings below.
    DataflowBufferSpec index_dfb{
        .unique_id = TTA_INDEX_DFB,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = dfb_data_format,
    };

    // --- Tensor parameters ---
    TensorParameter input_param{.unique_id = TTA_INPUT, .spec = input.tensor_spec()};
    TensorParameter start_param{.unique_id = TTA_START, .spec = start.tensor_spec()};
    TensorParameter end_param{.unique_id = TTA_END, .spec = end.tensor_spec()};
    TensorParameter output_param{.unique_id = TTA_OUTPUT, .spec = out.tensor_spec()};

    // --- Reader kernel ---
    // Three per-dimension blocks ride the common vararg channel, in this order: unpadded tile counts,
    // padded tile counts, input shape. The reader indexes all three by dimension.
    KernelSpec reader{
        .unique_id = TTA_READER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
                                        "reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TTA_TILES_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // Self-loop: this kernel is the index buffer's only toucher and drives both roles.
                DFBBinding{
                    .dfb_spec_name = TTA_INDEX_DFB,
                    .accessor_name = "index",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = TTA_INDEX_DFB,
                    .accessor_name = "index",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TTA_INPUT, .accessor_name = "src"},
                TensorBinding{.tensor_parameter_name = TTA_START, .accessor_name = "start"},
                TensorBinding{.tensor_parameter_name = TTA_END, .accessor_name = "end"},
            },
        .compile_time_args =
            {
                {"num_dims", ws.num_dims},
                {"tile_width", tile_width},
                {"tile_height", tile_height},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_id", "num_tiles"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options =
            {
                .num_runtime_varargs = ws.num_dims,
                .num_common_runtime_varargs = ws.num_dims * 3,
            },
    };

    // --- Writer kernel ---
    // Borrowed from eltwise/unary, which already carries a Metal 2.0 fork beside the legacy file. The
    // fork's binding vocabulary (dfb::out, tensor::dst, num_pages / start_id) is this spec's
    // constraint, not a free choice: other ops bind the same fork.
    KernelSpec writer{
        .unique_id = TTA_WRITER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id_metal2.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TTA_TILES_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = TTA_OUTPUT, .accessor_name = "dst"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ProgramSpec spec{
        .name = "slice_tile_tensor_args",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(tiles_dfb), std::move(index_dfb)},
        .tensor_parameters =
            {std::move(input_param), std::move(start_param), std::move(end_param), std::move(output_param)},
        .work_units =
            {
                WorkUnitSpec{
                    .name = "slice_tile_tensor_args",
                    .kernels = {TTA_READER, TTA_WRITER},
                    .target_nodes = ws.all_cores,
                },
            },
    };

    // --- Run args ---
    KernelRunArgs reader_run_args{.kernel = TTA_READER};
    KernelRunArgs writer_run_args{.kernel = TTA_WRITER};

    // Common vararg blocks, in the order the reader walks them. Shape-derived and hash-keyed, so
    // unlike the per-node values they are set once here and deliberately not re-applied on a hit.
    auto& common = reader_run_args.advanced_options.common_runtime_varargs;
    common.reserve(ws.num_dims * 3);
    common.insert(common.end(), ws.num_unpadded_tiles_per_dim.begin(), ws.num_unpadded_tiles_per_dim.end());
    common.insert(common.end(), ws.num_padded_tiles_per_dim.begin(), ws.num_padded_tiles_per_dim.end());
    const auto& input_shape = input_tensor.padded_shape();
    for (uint32_t i = 0; i < ws.num_dims; ++i) {
        common.push_back(input_shape[static_cast<int32_t>(i)]);
    }

    add_tensor_args_per_node_run_args(ws, reader_run_args, writer_run_args);

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {TTA_INPUT, input},
                {TTA_START, start},
                {TTA_END, end},
                {TTA_OUTPUT, out},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::experimental::ProgramRunArgs SliceTileTensorArgsProgramFactory::override_runtime_arguments(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto& input_tensor = tensor_args.input;

    const TensorArgsWorkSplit ws = resolve_tensor_args_work_split(args, input_tensor, output);

    KernelRunArgs reader_run_args{.kernel = TTA_READER};
    KernelRunArgs writer_run_args{.kernel = TTA_WRITER};
    add_tensor_args_per_node_run_args(ws, reader_run_args, writer_run_args);

    return ProgramRunArgs{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {TTA_INPUT, input_tensor.mesh_tensor()},
                {TTA_START, tensor_args.start_tensor.value().mesh_tensor()},
                {TTA_END, tensor_args.end_tensor.value().mesh_tensor()},
                {TTA_OUTPUT, output.mesh_tensor()},
            },
    };
}

}  // namespace ttnn::prim
