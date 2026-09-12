// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"

#include "ttnn/operations/data_movement/slice/device/slice_metal2_names.hpp"

#include <optional>
#include <span>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts SliceTileProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    using namespace ttnn::prim::slice_metal2;

    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    TT_ASSERT(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    // --- Dataflow buffer ---
    uint32_t num_input_tiles = 2;

    DataflowBufferSpec dfb_in{
        .unique_id = TILE_IN,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };

    // The reader walks a per-dimension tile-index odometer, incrementing it in place as it advances
    // through the source. The host seeds it with this core's starting position (the `id_per_dim`
    // vararg block below); the kernel copies that seed into this scratchpad and mutates it there.
    ScratchpadSpec id_per_dim_scratch{
        .unique_id = TILE_ID_PER_DIM,
        .size_per_node = num_dims * static_cast<uint32_t>(sizeof(uint32_t)),
    };

    // Reader common runtime args: [num_unpadded_per_dim..., num_padded_per_dim...]
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    // The source base address is a tensor binding, not an argument, so this vector holds only the
    // per-dim values.
    std::vector<uint32_t> reader_common_dims(num_dims * 2);
    std::span<uint32_t> reader_common_dims_view{reader_common_dims};
    auto num_unpadded_tiles_per_dim = reader_common_dims_view.subspan(0, num_dims);
    auto num_padded_tiles_per_dim = reader_common_dims_view.subspan(num_dims, num_dims);
    num_unpadded_tiles_per_dim[0] = num_unpadded_Xt;
    num_unpadded_tiles_per_dim[1] = num_unpadded_Yt;
    num_padded_tiles_per_dim[0] = num_padded_Xt;
    num_padded_tiles_per_dim[1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(num_dims); ++i) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_tiles_per_dim[i] = num_unpadded_dim;
        num_padded_tiles_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input, args.slice_start);

    KernelSpec reader{
        .unique_id = TILE_READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "reader_unary_unpad_dims_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE_IN,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .scratchpad_bindings =
            {
                ScratchpadBinding{
                    .scratchpad_spec_name = TILE_ID_PER_DIM,
                    .accessor_name = "id_per_dim",
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src",
                },
            },
        .compile_time_args = {{"num_dims", num_dims}},
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"start_id", "num_tiles"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options =
            {
                // Per-core: the id_per_dim seed.
                .num_runtime_varargs = num_dims,
                // Broadcast: num_unpadded_tiles then num_padded_tiles, one entry per dimension each.
                .num_common_runtime_varargs = num_dims * 2,
            },
    };

    KernelSpec writer{
        .unique_id = TILE_WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TILE_IN,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = OUTPUT,
                    .accessor_name = "dst",
                },
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"num_pages", "start_id"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Reader per-core: start_id, num_tiles, and the id_per_dim seed.
    // Writer per-core: num_pages, start_id.
    KernelRunArgs reader_run_args{.kernel = TILE_READER};
    KernelRunArgs writer_run_args{.kernel = TILE_WRITER};
    reader_run_args.advanced_options.common_runtime_varargs = reader_common_dims;

    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"start_id", 0u}, {"num_tiles", 0u}});
            reader_run_args.advanced_options.runtime_varargs[core] = std::vector<uint32_t>(num_dims, 0u);
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_pages", 0u}, {"start_id", 0u}});
            continue;
        }

        // Compute per-dim indices for this core's starting position
        std::vector<uint32_t> id_per_dim(num_dims);
        id_per_dim[0] = num_tiles_written % num_unpadded_tiles_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
        uint32_t start_id = id_per_dim[0] + start_offset;
        for (uint32_t j = 1; j < num_dims; ++j) {
            id_per_dim[j] = unpadded_written % num_unpadded_tiles_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"start_id", start_id}, {"num_tiles", num_tiles_per_core}});
        reader_run_args.advanced_options.runtime_varargs[core] = std::move(id_per_dim);

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_written}});

        num_tiles_written += num_tiles_per_core;
    }

    ProgramSpec spec{
        .name = "slice_tile",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(dfb_in)},
        .scratchpads = {std::move(id_per_dim_scratch)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {TILE_READER, TILE_WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input.mesh_tensor()},
        {OUTPUT, output.mesh_tensor()},
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
    return slice_program_run_args(SliceTileProgramFactory{}, args, tensor_args, output);
}

tt::tt_metal::experimental::Group<tt::tt_metal::experimental::KernelRunArgs> slice_tile_run_args(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    const Tensor& output,
    uint32_t start_offset,
    const tt::tt_metal::experimental::KernelSpecName& reader_kernel,
    const tt::tt_metal::experimental::KernelSpecName& writer_kernel) {
    // Must reproduce create_program_artifacts's work split exactly; divergence leaves stale scalars in these slots.
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

    KernelRunArgs reader_run_args{.kernel = reader_kernel};
    KernelRunArgs writer_run_args{.kernel = writer_kernel};

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

        uint32_t start_id = 0;
        std::vector<uint32_t> id_per_dim(num_dims, 0);
        if (active) {
            id_per_dim[0] = num_tiles_written % num_unpadded_tiles_per_dim[0];
            uint32_t unpadded_written = num_tiles_written / num_unpadded_tiles_per_dim[0];
            start_id = id_per_dim[0] + start_offset;
            for (uint32_t j = 1; j < num_dims; ++j) {
                id_per_dim[j] = unpadded_written % num_unpadded_tiles_per_dim[j];
                unpadded_written = unpadded_written / num_unpadded_tiles_per_dim[j];
                start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
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
    return {std::move(reader_run_args), std::move(writer_run_args)};
}

}  // namespace ttnn::prim
