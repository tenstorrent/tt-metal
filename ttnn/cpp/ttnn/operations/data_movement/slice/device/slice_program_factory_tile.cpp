// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"
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
const KernelSpecName TILE_READER{"reader"};
const KernelSpecName TILE_WRITER{"writer"};
const DFBSpecName TILE_DFB{"tiles"};
const TensorParamName TILE_INPUT{"input"};
const TensorParamName TILE_OUTPUT{"output"};

// Per-node work split, resolved identically on the cache-miss and cache-hit paths.
struct TileWorkSplit {
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

TileWorkSplit resolve_work_split(const SliceParams& args, const ttnn::Tensor& input, const ttnn::Tensor& output) {
    tt::tt_metal::IDevice* device = input.device();
    const uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    const std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    const uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    const uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    const uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    const uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    const uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    const uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    TileWorkSplit ws{
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
// Its other values are inert -- a zero page count is what stops either kernel entering its transfer
// loop -- so the writer's start_id is simply left at the running tile total there rather than being
// special-cased to zero.
//
// One function serves both the cache-miss and the cache-hit path, which must agree exactly or a
// cache hit leaves stale per-node scalars behind (#52651).
void add_per_node_run_args(
    const TileWorkSplit& ws, uint32_t start_offset, KernelRunArgs& reader_run_args, KernelRunArgs& writer_run_args) {
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

ttnn::device_operation::ProgramArtifacts SliceTileProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& input = input_tensor.mesh_tensor();
    const auto& out = output.mesh_tensor();
    tt::tt_metal::IDevice* device = input_tensor.device();

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    const tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(dfb_data_format);
    const uint32_t num_input_tiles = 2;

    const TileWorkSplit ws = resolve_work_split(args, input_tensor, output);

    // --- Dataflow buffer ---
    // Staging FIFO between the reader and the writer: the reader fills it a tile at a time from the
    // input tensor, the writer drains it to the output tensor.
    DataflowBufferSpec tiles_dfb{
        .unique_id = TILE_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };

    // --- Tensor parameters ---
    TensorParameter input_param{
        .unique_id = TILE_INPUT,
        .spec = input.tensor_spec(),
    };
    TensorParameter output_param{
        .unique_id = TILE_OUTPUT,
        .spec = out.tensor_spec(),
    };

    // --- Reader kernel ---
    // The per-dimension unpadded/padded tile counts and this node's per-dimension index vector are
    // variable-count blocks (one entry per dimension), so they ride the vararg channel; everything
    // else the reader takes is a named argument.
    KernelSpec reader{
        .unique_id = TILE_READER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
                                        "reader_unary_unpad_dims_interleaved_start_id.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE_DFB,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = TILE_INPUT,
                    .accessor_name = "src",
                },
            },
        .compile_time_args = {{"num_dims", ws.num_dims}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_id", "num_tiles"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options =
            {
                .num_runtime_varargs = ws.num_dims,
                .num_common_runtime_varargs = ws.num_dims * 2,
            },
    };

    // --- Writer kernel ---
    KernelSpec writer{
        .unique_id = TILE_WRITER,
        .source = std::filesystem::path("ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
                                        "writer_unary_interleaved_start_id.cpp"),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE_DFB,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = TILE_OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ProgramSpec spec{
        .name = "slice_tile",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(tiles_dfb)},
        .tensor_parameters = {std::move(input_param), std::move(output_param)},
        .work_units =
            {
                WorkUnitSpec{
                    .name = "slice_tile",
                    .kernels = {TILE_READER, TILE_WRITER},
                    .target_nodes = ws.all_cores,
                },
            },
    };

    // --- Run args ---
    KernelRunArgs reader_run_args{.kernel = TILE_READER};
    KernelRunArgs writer_run_args{.kernel = TILE_WRITER};

    // Common vararg block, in the order the reader walks it: all unpadded counts, then all padded
    // counts. Shape-derived and hash-keyed, so unlike the per-node values it is set once here and
    // is deliberately not re-applied on a cache hit.
    reader_run_args.advanced_options.common_runtime_varargs.reserve(ws.num_dims * 2);
    reader_run_args.advanced_options.common_runtime_varargs.insert(
        reader_run_args.advanced_options.common_runtime_varargs.end(),
        ws.num_unpadded_tiles_per_dim.begin(),
        ws.num_unpadded_tiles_per_dim.end());
    reader_run_args.advanced_options.common_runtime_varargs.insert(
        reader_run_args.advanced_options.common_runtime_varargs.end(),
        ws.num_padded_tiles_per_dim.begin(),
        ws.num_padded_tiles_per_dim.end());

    const uint32_t start_offset =
        ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, args.slice_start);
    add_per_node_run_args(ws, start_offset, reader_run_args, writer_run_args);

    ProgramRunArgs run_args{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {TILE_INPUT, input},
                {TILE_OUTPUT, out},
            },
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::experimental::ProgramRunArgs SliceTileProgramFactory::override_runtime_arguments(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto& input_tensor = tensor_args.input;

    const TileWorkSplit ws = resolve_work_split(args, input_tensor, output);

    KernelRunArgs reader_run_args{.kernel = TILE_READER};
    KernelRunArgs writer_run_args{.kernel = TILE_WRITER};

    const uint32_t start_offset =
        ttnn::operations::data_movement::get_tiled_start_offset(input_tensor, args.slice_start);
    add_per_node_run_args(ws, start_offset, reader_run_args, writer_run_args);

    return ProgramRunArgs{
        .kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)},
        .tensor_args =
            {
                {TILE_INPUT, input_tensor.mesh_tensor()},
                {TILE_OUTPUT, output.mesh_tensor()},
            },
    };
}

std::vector<tt::tt_metal::DynamicRuntimeArg> slice_tile_dynamic_args(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    const Tensor& output,
    uint32_t start_offset,
    uint32_t reader_kernel_idx,
    uint32_t writer_kernel_idx) {
    // Must reproduce create_descriptor's work split exactly; divergence leaves stale scalars in these slots.
    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();
    const uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    const std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    const uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    const uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    const uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    const uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;
    std::vector<uint32_t> num_unpadded_tiles_per_dim(num_dims);
    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        const uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        const uint32_t num_total_dim = input_shape[-(i + 1)];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    const auto cores = corerange_to_cores(all_cores);
    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
    dynamic_args.reserve(cores.size() * (2 + num_dims + 2));

    uint32_t num_tiles_written = 0;
    for (const auto& core : cores) {
        uint32_t num_tiles_per_core = 0;
        bool active = true;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            active = false;
        }

        uint32_t id0 = 0, start_id = 0;
        std::vector<uint32_t> id_per_dim(num_dims, 0);
        if (active) {
            id0 = num_tiles_written % num_unpadded_tiles_per_dim[0];
            uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
            start_id = id0 + start_offset;
            for (uint32_t j = 1; j < num_dims; ++j) {
                id_per_dim[j] = unpadded_written % num_unpadded_tiles_per_dim[j];
                unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
                start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
            }
        }

        dynamic_args.push_back({reader_kernel_idx, core, 0, start_id, false});
        dynamic_args.push_back({reader_kernel_idx, core, 1, num_tiles_per_core, false});
        dynamic_args.push_back({reader_kernel_idx, core, 2, id0, false});
        for (uint32_t j = 1; j < num_dims; ++j) {
            dynamic_args.push_back({reader_kernel_idx, core, 2 + j, id_per_dim[j], false});
        }
        // Writer slot 0 (dst buffer) is patched by patch_slot0; re-emit only slots 1 and 2.
        dynamic_args.push_back({writer_kernel_idx, core, 1, num_tiles_per_core, false});
        dynamic_args.push_back({writer_kernel_idx, core, 2, num_tiles_written, false});

        if (active) {
            num_tiles_written += num_tiles_per_core;
        }
    }
    return dynamic_args;
}

}  // namespace ttnn::prim
