// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_metal2_names.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.hpp"

#include <optional>
#include <span>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// This factory's two dataflow buffers. The identifiers are factory-prefixed for the unity build.
// The staging one is a single-entry scratch the reader both fills and drains, twice: once for the
// start tensor and once for the end tensor.
const DFBSpecName TTA_TILES{"tiles"};
const DFBSpecName TTA_STAGING{"staging"};

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceTileTensorArgsProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    using slice_metal2::END;
    using slice_metal2::INPUT;
    using slice_metal2::OUTPUT;
    using slice_metal2::READER;
    using slice_metal2::START;
    using slice_metal2::WRITER;

    const auto& input_tensor = tensor_args.input;
    const auto& start_tensor = tensor_args.start_tensor.value();
    const auto& end_tensor = tensor_args.end_tensor.value();
    tt::tt_metal::IDevice* device = input_tensor.device();

    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& start_mesh_tensor = start_tensor.mesh_tensor();
    const auto& end_mesh_tensor = end_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    TT_FATAL(input_tensor.buffer() != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(start_tensor.buffer() != nullptr, "Start buffer should be allocated on device!");
    TT_FATAL(end_tensor.buffer() != nullptr, "End buffer should be allocated on device!");
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    constexpr uint32_t num_input_tiles = 2;
    DataflowBufferSpec tiles_dfb{
        .unique_id = TTA_TILES,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };
    DataflowBufferSpec staging_dfb{
        .unique_id = TTA_STAGING,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = dfb_data_format,
    };

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_tensor.padded_shape().rank());
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_height = tile_shape[0];

    // Reader common args: three num_dims-long blocks, in this order:
    //   num_unpadded_tiles_per_dim, num_padded_tiles_per_dim, input_shape
    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    std::vector<uint32_t> reader_common_dims(num_dims * 3);
    std::span<uint32_t> reader_common_dims_view{reader_common_dims};
    auto num_unpadded_tiles_per_dim = reader_common_dims_view.subspan(0, num_dims);
    auto num_padded_tiles_per_dim = reader_common_dims_view.subspan(num_dims, num_dims);
    auto input_shape_args = reader_common_dims_view.subspan(num_dims * 2, num_dims);
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
    for (int32_t i = 0; i < static_cast<int32_t>(num_dims); ++i) {
        input_shape_args[i] = input_shape[i];
    }

    // This factory's slice bounds come off the device, so the per-core start index carries no
    // host-computed offset; the reader derives that from the start tensor it reads.
    constexpr uint32_t start_offset = 0;

    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TTA_TILES,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                // The staging buffer is touched only by this reader, which runs the full
                // producer/consumer handshake on it, so it is bound at both endpoints.
                DFBBinding{
                    .dfb_spec_name = TTA_STAGING,
                    .accessor_name = "staging",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = TTA_STAGING,
                    .accessor_name = "staging",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src",
                },
                TensorBinding{
                    .tensor_parameter_name = START,
                    .accessor_name = "start",
                },
                TensorBinding{
                    .tensor_parameter_name = END,
                    .accessor_name = "end",
                },
            },
        .compile_time_args =
            {
                {"num_dims", num_dims},
                {"tile_width", tile_width},
                {"tile_height", tile_height},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options =
            {
                // The per-dim walk counters, one per dimension, advanced as the kernel reads.
                .num_runtime_varargs = num_dims,
                // The three per-dim blocks, identical on every node.
                .num_common_runtime_varargs = num_dims * 3,
            },
    };

    // The writer is the eltwise/unary donor's Metal 2.0 fork, which already exists beside the
    // original. Its binding vocabulary (dfb::out, tensor::dst, args::num_pages, args::start_id) is
    // this factory's constraint rather than a free choice. It gates OUT_SHARDED and BACKWARDS, and
    // slice defines neither, so no compiler_options.defines entry is needed.
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = TTA_TILES,
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
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    reader_run_args.advanced_options.common_runtime_varargs =
        AdvancedKernelRunArgs::Varargs(reader_common_dims.begin(), reader_common_dims.end());

    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core (num_tiles == 0, so the loops never run, but the args are bound for a
            // uniform layout across nodes)
            AddRuntimeArgsForNode(reader_run_args.runtime_arg_values, core, {{"start_id", 0}, {"num_tiles", 0}});
            reader_run_args.advanced_options.runtime_varargs[core] = AdvancedKernelRunArgs::Varargs(num_dims, 0);
            AddRuntimeArgsForNode(writer_run_args.runtime_arg_values, core, {{"num_pages", 0}, {"start_id", 0}});
            continue;
        }

        AdvancedKernelRunArgs::Varargs id_per_dim(num_dims, 0);
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
        .name = "slice_tile_tensor_args",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(tiles_dfb), std::move(staging_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = START, .spec = start_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = END, .spec = end_mesh_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {READER, WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_mesh_tensor},
        {START, start_mesh_tensor},
        {END, end_mesh_tensor},
        {OUTPUT, output_mesh_tensor},
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
    // This factory's start index carries no host-side offset, so the per-core walk is re-derived with
    // a zero start_offset, exactly as the ported-from refresh did.
    ProgramRunArgs run_args = slice_tile_per_core_run_args(args, tensor_args, output, /*start_offset=*/0);
    run_args.tensor_args = {
        {slice_metal2::INPUT, tensor_args.input.mesh_tensor()},
        {slice_metal2::START, tensor_args.start_tensor.value().mesh_tensor()},
        {slice_metal2::END, tensor_args.end_tensor.value().mesh_tensor()},
        {slice_metal2::OUTPUT, output.mesh_tensor()},
    };
    return run_args;
}

}  // namespace ttnn::prim
