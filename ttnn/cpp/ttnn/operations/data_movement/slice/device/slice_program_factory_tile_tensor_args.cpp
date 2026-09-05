// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.hpp"
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

// Function-local for the same unity-build reason as the other slice factories.
struct TensorArgsSpecNames {
    KernelSpecName reader{"reader"};
    KernelSpecName writer{"writer"};
    DFBSpecName src0{"src0"};
    // Single-entry staging buffer the reader fills and drains itself, once for the start tensor and
    // once for the end tensor.
    DFBSpecName tensor_stage{"tensor_stage"};
    TensorParamName input{"input"};
    TensorParamName start{"start"};
    TensorParamName end{"end"};
    // The writer is the eltwise/unary Metal 2.0 fork; `output` is bound to its own `tensor::dst`.
    TensorParamName output{"output"};
};

// The start offset is computed on device from the start tensor, so the host contributes none.
constexpr uint32_t kHostStartOffset = 0;

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceTileTensorArgsProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const TensorArgsSpecNames names;

    const auto& input_tensor = tensor_args.input;
    const auto& start_tensor = tensor_args.start_tensor.value();
    const auto& end_tensor = tensor_args.end_tensor.value();
    tt::tt_metal::IDevice* device = input_tensor.device();

    TT_FATAL(input_tensor.buffer() != nullptr, "Input buffer should be allocated on device!");
    TT_FATAL(start_tensor.buffer() != nullptr, "Start buffer should be allocated on device!");
    TT_FATAL(end_tensor.buffer() != nullptr, "End buffer should be allocated on device!");
    TT_FATAL(output.buffer() != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t single_tile_size = tt::tile_size(dfb_data_format);

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    const std::uint32_t num_dims = static_cast<std::uint32_t>(input_shape.rank());
    const auto tile_shape = input_tensor.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_width = tile_shape[1];
    const uint32_t tile_height = tile_shape[0];

    const auto split = slice_tile_work_split(args, tensor_args, output, kHostStartOffset);

    // --- DFBs ---
    constexpr uint32_t num_input_tiles = 2;
    const DataflowBufferSpec src0_dfb{
        .unique_id = names.src0,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = dfb_data_format,
    };
    // One toucher that already runs both halves of the handshake (reserve/push/wait/pop, twice), so
    // the reader binds both endpoints.
    const DataflowBufferSpec tensor_stage_dfb{
        .unique_id = names.tensor_stage,
        .entry_size = single_tile_size,
        .num_entries = 1,
        .data_format_metadata = dfb_data_format,
    };

    // --- Tensor parameters ---
    const TensorParameter input_param{.unique_id = names.input, .spec = input_tensor.tensor_spec()};
    const TensorParameter start_param{.unique_id = names.start, .spec = start_tensor.tensor_spec()};
    const TensorParameter end_param{.unique_id = names.end, .spec = end_tensor.tensor_spec()};
    const TensorParameter output_param{.unique_id = names.output, .spec = output.tensor_spec()};

    // --- Reader common varargs ---
    //   [num_unpadded_tiles_per_dim..., num_padded_tiles_per_dim..., input_shape...]
    const uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    const uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    const uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    const uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    const uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    const uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    AdvancedKernelRunArgs::Varargs reader_common_varargs(num_dims * 3);
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
    for (int32_t i = 0; i < static_cast<int32_t>(num_dims); ++i) {
        reader_common_varargs[num_dims * 2 + i] = input_shape[i];
    }

    // --- Kernels ---
    const KernelSpec reader{
        .unique_id = names.reader,
        .source =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
            "reader_unary_unpad_dims_interleaved_start_id_tensor_args.cpp",
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.src0,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = names.tensor_stage,
                    .accessor_name = "tensor_stage",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = names.tensor_stage,
                    .accessor_name = "tensor_stage",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = names.input, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = names.start, .accessor_name = "start"},
                TensorBinding{.tensor_parameter_name = names.end, .accessor_name = "end"},
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
                .num_runtime_varargs = num_dims,
                .num_common_runtime_varargs = num_dims * 3,
            },
    };

    // Reuse the Metal 2.0 fork that lives beside the eltwise/unary original. Its interface is fixed
    // by the ops already bound to it: DFB `out`, tensor `dst`, named args `num_pages` / `start_id`.
    // It gates on OUT_SHARDED / BACKWARDS, and slice sets no defines, so neither fires.
    const KernelSpec writer{
        .unique_id = names.writer,
        .source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
            "writer_unary_interleaved_start_id_metal2.cpp",
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
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            per_node.node,
            {{"num_pages", per_node.num_tiles}, {"start_id", per_node.active ? per_node.num_tiles_written : 0u}});
    }

    ProgramSpec spec{
        .name = "slice_tile_tensor_args",
        .kernels = {reader, writer},
        .dataflow_buffers = {src0_dfb, tensor_stage_dfb},
        .tensor_parameters = {input_param, start_param, end_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "slice",
            .kernels = {names.reader, names.writer},
            .target_nodes = split.all_nodes,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {names.input, input_tensor.mesh_tensor()},
        {names.start, start_tensor.mesh_tensor()},
        {names.end, end_tensor.mesh_tensor()},
        {names.output, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs SliceTileTensorArgsProgramFactory::override_runtime_arguments(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const TensorArgsSpecNames names;

    const auto split = slice_tile_work_split(args, tensor_args, output, kHostStartOffset);

    KernelRunArgs reader_run_args{.kernel = names.reader};
    KernelRunArgs writer_run_args{.kernel = names.writer};
    for (const auto& per_node : split.per_node) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            per_node.node,
            {{"start_id", per_node.start_id}, {"num_tiles", per_node.num_tiles}});
        reader_run_args.advanced_options.runtime_varargs[per_node.node] = per_node.id_per_dim;
        // See SliceTileProgramFactory::override_runtime_arguments: the running tile count goes out
        // even on an inactive node, matching the legacy refresh path rather than the legacy build.
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            per_node.node,
            {{"num_pages", per_node.num_tiles}, {"start_id", per_node.num_tiles_written}});
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {names.input, tensor_args.input.mesh_tensor()},
        {names.start, tensor_args.start_tensor.value().mesh_tensor()},
        {names.end, tensor_args.end_tensor.value().mesh_tensor()},
        {names.output, output.mesh_tensor()}};
    return run_args;
}

}  // namespace ttnn::prim
