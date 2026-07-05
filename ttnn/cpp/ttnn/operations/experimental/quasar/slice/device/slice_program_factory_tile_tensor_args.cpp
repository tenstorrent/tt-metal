// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory_tile_tensor_args.hpp"

#include <optional>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts SliceTileTensorArgsProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& start_tensor = tensor_args.start_tensor.value();
    const auto& end_tensor = tensor_args.end_tensor.value();
    tt::tt_metal::IDevice* device = input_tensor.device();

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    std::uint32_t num_dims = static_cast<std::uint32_t>(input_tensor.padded_shape().rank());
    auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    uint32_t tile_width = tile_shape[1];
    uint32_t tile_height = tile_shape[0];

    // Resource names
    const DFBSpecName C0{"c0"};  // src0 (legacy CB c_0)
    const DFBSpecName C1{"c1"};  // tensor scratch (legacy CB c_1)
    const TensorParamName INPUT{"input"};
    const TensorParamName START{"start"};
    const TensorParamName END{"end"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // --- DataflowBuffers ---
    constexpr uint32_t num_input_tiles = 2;
    DataflowBufferSpec c0_dfb{
        .unique_id = C0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };
    DataflowBufferSpec c1_dfb{
        .unique_id = C1,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = cb_data_format,
    };

    // --- Common reader args: num_unpadded_per_dim..., num_padded_per_dim..., input_shape...
    //     (each read in a loop by a runtime-varying index → common runtime varargs). ---
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

    // Common runtime varargs layout:
    //   [0, num_dims)            num_unpadded
    //   [num_dims, 2*num_dims)   num_padded
    //   [2*num_dims, 3*num_dims) input_shape
    std::vector<uint32_t> reader_common_varargs;
    reader_common_varargs.reserve(3 * num_dims);
    reader_common_varargs.insert(
        reader_common_varargs.end(), num_unpadded_tiles_per_dim.begin(), num_unpadded_tiles_per_dim.end());
    reader_common_varargs.insert(
        reader_common_varargs.end(), num_padded_tiles_per_dim.begin(), num_padded_tiles_per_dim.end());
    for (int32_t i = 0; i < static_cast<int32_t>(num_dims); ++i) {
        reader_common_varargs.push_back(input_shape[i]);
    }

    // --- Reader KernelSpec ---
    // c_1 (tensor scratch) is a real produce-then-consume scratch FIFO on the reader → self-loop binding.
    KernelSpec reader{
        .unique_id = READER,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/"
            "reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp",
        .compiler_options = {},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = C0, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = C1, .accessor_name = "cb_tensor", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = C1, .accessor_name = "cb_tensor", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "in"},
             TensorBinding{.tensor_parameter_name = START, .accessor_name = "start"},
             TensorBinding{.tensor_parameter_name = END, .accessor_name = "end"}},
        .compile_time_args = {{"num_dims", num_dims}, {"tile_width", tile_width}, {"tile_height", tile_height}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
        .advanced_options = {.num_runtime_varargs = num_dims, .num_common_runtime_varargs = 3 * num_dims},
    };

    // --- Writer KernelSpec (shared writer_unary_interleaved_start_id.cpp) ---
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
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    // --- Per-core runtime args ---
    constexpr uint32_t start_offset = 0;  // tensor-args path computes the start offset in-kernel
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
    TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.tensor_spec()};
    TensorParameter start_param{.unique_id = START, .spec = start_tensor.tensor_spec()};
    TensorParameter end_param{.unique_id = END, .spec = end_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // --- Assemble ProgramSpec ---
    ProgramSpec spec;
    spec.name = "slice_tile_tensor_args";
    spec.kernels = {reader, writer};
    spec.dataflow_buffers = {c0_dfb, c1_dfb};
    spec.tensor_parameters = {input_param, start_param, end_param, output_param};
    spec.work_units = {WorkUnitSpec{
        .name = "slice_tile_tensor_args_wu",
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
    run_args.tensor_args.emplace(INPUT, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(START, start_tensor.mesh_tensor());
    run_args.tensor_args.emplace(END, end_tensor.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
