// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory_tile.hpp"

#include <optional>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts SliceTileProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    tt::tt_metal::IDevice* device = input.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());

    // Resource names
    const DFBSpecName C0{"c0"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // --- DataflowBuffer (legacy CB c_0) ---
    constexpr uint32_t num_input_tiles = 2;
    DataflowBufferSpec c0_dfb{
        .unique_id = C0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };

    // --- Common reader args: num_unpadded_per_dim..., num_padded_per_dim... (read in a loop
    //     by a runtime-varying index → common runtime varargs). ---
    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    std::vector<uint32_t> num_unpadded_tiles_per_dim(num_dims);
    std::vector<uint32_t> num_padded_tiles_per_dim(num_dims);
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

    // Common runtime varargs layout: [0, num_dims) num_unpadded, [num_dims, 2*num_dims) num_padded.
    std::vector<uint32_t> reader_common_varargs;
    reader_common_varargs.reserve(2 * num_dims);
    reader_common_varargs.insert(
        reader_common_varargs.end(), num_unpadded_tiles_per_dim.begin(), num_unpadded_tiles_per_dim.end());
    reader_common_varargs.insert(
        reader_common_varargs.end(), num_padded_tiles_per_dim.begin(), num_padded_tiles_per_dim.end());

    uint32_t start_offset = ttnn::operations::experimental::quasar::get_tiled_start_offset(input, args.slice_start);

    // --- Reader KernelSpec ---
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "reader_unary_unpad_dims_interleaved_start_id.cpp",
        .compiler_options = {},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C0, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "in"}},
        .compile_time_args = {{"num_dims", num_dims}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = num_dims, .num_common_runtime_varargs = 2 * num_dims},
    };

    // --- Writer KernelSpec ---
    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id.cpp",
        .compiler_options = {},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C0, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // --- Per-core runtime args ---
    // Reader per-core: named (start_id, num_tiles) + id_per_dim runtime varargs.
    // Writer per-core: named (num_pages, start_id).
    Group<KernelRunArgs::NodeRuntimeArgs> reader_node_args;
    Group<KernelRunArgs::NodeRuntimeArgs> writer_node_args;
    AdvancedKernelRunArgs reader_run_advanced;

    uint32_t num_tiles_written = 0;
    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            reader_node_args.push_back({.node = core, .args = {{"start_id", 0}, {"num_tiles", 0}}});
            reader_run_advanced.runtime_varargs.emplace(core, std::vector<uint32_t>(num_dims, 0));
            writer_node_args.push_back({.node = core, .args = {{"num_pages", 0}, {"start_id", 0}}});
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

        reader_node_args.push_back({.node = core, .args = {{"start_id", start_id}, {"num_tiles", num_tiles_per_core}}});
        reader_run_advanced.runtime_varargs.emplace(core, std::move(id_per_dim));

        writer_node_args.push_back(
            {.node = core, .args = {{"num_pages", num_tiles_per_core}, {"start_id", num_tiles_written}}});

        num_tiles_written += num_tiles_per_core;
    }

    // --- TensorParameters ---
    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // --- Assemble ProgramSpec ---
    ProgramSpec spec;
    spec.name = "slice_tile";
    spec.kernels = {reader, writer};
    spec.dataflow_buffers = {c0_dfb};
    spec.tensor_parameters = {input_param, output_param};
    spec.work_units = {WorkUnitSpec{
        .name = "slice_tile_wu",
        .kernels = {READER, WRITER},
        .target_nodes = all_cores,
    }};

    // --- Assemble ProgramRunArgs ---
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{
            .kernel = READER,
            .runtime_arg_values = std::move(reader_node_args),
            .common_runtime_arg_values = {},
            .advanced_options =
                AdvancedKernelRunArgs{
                    .runtime_varargs = std::move(reader_run_advanced.runtime_varargs),
                    .common_runtime_varargs = std::move(reader_common_varargs)},
        },
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = std::move(writer_node_args),
        },
    };
    run_args.tensor_args.emplace(INPUT, input.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
