// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Spec resource names are declared function-locally rather than at file scope: all five slice
// factories land in one unity-build translation unit, where two files declaring `READER` would
// collide.
struct TileSpecNames {
    KernelSpecName reader{"reader"};
    KernelSpecName writer{"writer"};
    DFBSpecName src0{"src0"};
    TensorParamName input{"input"};
    TensorParamName output{"output"};
};

}  // namespace

SliceTileWorkSplit slice_tile_work_split(
    const SliceParams& args, const SliceInputs& tensor_args, const Tensor& output, uint32_t start_offset) {
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

    SliceTileWorkSplit split;
    split.all_nodes = all_cores;
    const auto nodes = corerange_to_cores(all_cores);
    split.per_node.reserve(nodes.size());

    uint32_t num_tiles_written = 0;
    for (const auto& node : nodes) {
        SliceTilePerNodeArgs per_node;
        per_node.node = node;
        per_node.id_per_dim.assign(num_dims, 0);
        per_node.num_tiles_written = num_tiles_written;

        if (core_group_1.contains(node)) {
            per_node.active = true;
            per_node.num_tiles = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(node)) {
            per_node.active = true;
            per_node.num_tiles = num_tiles_per_core_group_2;
        }

        if (per_node.active) {
            // Per-dim indices for this node's starting position.
            per_node.id_per_dim[0] = num_tiles_written % num_unpadded_tiles_per_dim[0];
            uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
            per_node.start_id = per_node.id_per_dim[0] + start_offset;
            for (uint32_t j = 1; j < num_dims; ++j) {
                per_node.id_per_dim[j] = unpadded_written % num_unpadded_tiles_per_dim[j];
                unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
                per_node.start_id += per_node.id_per_dim[j] * accumulated_total_per_dim[j - 1];
            }
            num_tiles_written += per_node.num_tiles;
        }

        split.per_node.push_back(std::move(per_node));
    }
    return split;
}

ttnn::device_operation::ProgramArtifacts SliceTileProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const TileSpecNames names;

    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    const std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    const uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input, args.slice_start);
    const auto split = slice_tile_work_split(args, tensor_args, output, start_offset);

    // --- DFB ---
    constexpr uint32_t num_input_tiles = 2;
    const DataflowBufferSpec src0_dfb{
        .unique_id = names.src0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };

    // --- Tensor parameters ---
    const TensorParameter input_param{.unique_id = names.input, .spec = input.tensor_spec()};
    const TensorParameter output_param{.unique_id = names.output, .spec = output.tensor_spec()};

    // --- Reader common varargs: [num_unpadded_tiles_per_dim..., num_padded_tiles_per_dim...] ---
    const uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    const uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    const uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    const uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    const uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    const uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    AdvancedKernelRunArgs::Varargs reader_common_varargs(num_dims * 2);
    reader_common_varargs[0] = num_unpadded_Xt;
    reader_common_varargs[1] = num_unpadded_Yt;
    reader_common_varargs[num_dims] = num_padded_Xt;
    reader_common_varargs[num_dims + 1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        const uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        const uint32_t num_total_dim = input_shape[-(i + 1)];
        reader_common_varargs[i] = num_unpadded_dim;
        reader_common_varargs[num_dims + i] = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    // --- Kernels ---
    const KernelSpec reader{
        .unique_id = names.reader,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "reader_unary_unpad_dims_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.src0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = names.input, .accessor_name = "input"},
            },
        .compile_time_args = {{"num_dims", num_dims}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options =
            {
                .num_runtime_varargs = num_dims,
                .num_common_runtime_varargs = num_dims * 2,
            },
    };

    // Slice keeps its own copy of the unary writer rather than binding the shared eltwise one: the
    // DFB it drains is selected per instantiation so the fusion infrastructure can remap it, which
    // the shared copy cannot express.
    const KernelSpec writer{
        .unique_id = names.writer,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.src0,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = names.output, .accessor_name = "dst"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // --- Per-node run args ---
    KernelRunArgs reader_run_args{.kernel = names.reader};
    KernelRunArgs writer_run_args{.kernel = names.writer};
    reader_run_args.advanced_options.common_runtime_varargs = std::move(reader_common_varargs);

    for (const auto& per_node : split.per_node) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            per_node.node,
            {{"start_id", per_node.start_id}, {"num_tiles", per_node.num_tiles}});
        reader_run_args.advanced_options.runtime_varargs[per_node.node] = per_node.id_per_dim;

        // An inactive node writes nothing, so its start_id is never read; the value emitted here is
        // 0, which is what the legacy descriptor emitted. (The cache-hit path below re-emits the
        // running tile count instead — a legacy divergence, preserved.)
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            per_node.node,
            {{"num_pages", per_node.num_tiles}, {"start_id", per_node.active ? per_node.num_tiles_written : 0u}});
    }

    ProgramSpec spec{
        .name = "slice_tile",
        .kernels = {reader, writer},
        .dataflow_buffers = {src0_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "slice",
            .kernels = {names.reader, names.writer},
            .target_nodes = split.all_nodes,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{names.input, input.mesh_tensor()}, {names.output, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs SliceTileProgramFactory::override_runtime_arguments(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const TileSpecNames names;

    const uint32_t start_offset =
        ttnn::operations::data_movement::get_tiled_start_offset(tensor_args.input, args.slice_start);
    const auto split = slice_tile_work_split(args, tensor_args, output, start_offset);

    KernelRunArgs reader_run_args{.kernel = names.reader};
    KernelRunArgs writer_run_args{.kernel = names.writer};
    for (const auto& per_node : split.per_node) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            per_node.node,
            {{"start_id", per_node.start_id}, {"num_tiles", per_node.num_tiles}});
        reader_run_args.advanced_options.runtime_varargs[per_node.node] = per_node.id_per_dim;
        // Unconditionally the running tile count, including on an inactive node — where it differs
        // from the value the cache-miss path emitted (0). Inert either way: num_pages is 0 there, so
        // the writer's loop never runs and start_id is never read.
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            per_node.node,
            {{"num_pages", per_node.num_tiles}, {"start_id", per_node.num_tiles_written}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    // On this concept the framework refreshes nothing on its own, so every tensor binding is
    // re-supplied here; that is what re-points the input and output base addresses on a cache hit.
    run_args.tensor_args = {{names.input, tensor_args.input.mesh_tensor()}, {names.output, output.mesh_tensor()}};
    return run_args;
}

}  // namespace ttnn::prim
