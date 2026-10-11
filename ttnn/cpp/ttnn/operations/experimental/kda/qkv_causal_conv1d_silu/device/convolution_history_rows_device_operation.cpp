// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "convolution_history_rows_device_operation.hpp"

#include <algorithm>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/experimental/kda/factory/chronology_binding.hpp"
#include "ttnn/operations/experimental/kda/factory/kda_factory_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental::prim {

namespace {
constexpr uint32_t history_table_rows = 6;
// Channel tiles per worker: enough cores that the 1 KiB face reads overlap, few enough to stay latency-bound.
constexpr uint32_t tiles_per_worker = 8;
}  // namespace

ConvolutionHistoryRowsOperation::program_factory_t ConvolutionHistoryRowsOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return ConvolutionHistoryRowsProgramFactory{};
}

void ConvolutionHistoryRowsOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    using namespace kda_factory_detail;
    constexpr std::string_view operation_name = "convolution_history_rows";
    check_actual_start(in.input, in.actual_start, operation_name);
    if (in.actual_end) {
        check_actual_start(in.input, *in.actual_end, operation_name);
    }
    check_allocated_device_tensor(in.input, operation_name, "input");
    check_layout(in.input, Layout::TILE, operation_name, "input");
    check_dtype(in.input, DataType::BFLOAT16, operation_name, "input");
    check_interleaved(in.input, operation_name, "input");
    const auto& shape = in.input.logical_shape();
    TT_FATAL(shape.rank() == 3 && shape[0] == 1, "{}: input must be [1,T,W]", operation_name);
    TT_FATAL(
        shape[1] >= tt::constants::TILE_HEIGHT && shape[1] % tt::constants::TILE_HEIGHT == 0,
        "{}: T must be positive and tile aligned",
        operation_name);
    TT_FATAL(
        attrs.width > 0 && attrs.width % tt::constants::TILE_WIDTH == 0 &&
            attrs.column_offset % tt::constants::TILE_WIDTH == 0 && attrs.column_offset + attrs.width <= shape[2],
        "{}: the tile-aligned column window [{}, {}) must lie within the input width {}",
        operation_name,
        attrs.column_offset,
        attrs.column_offset + attrs.width,
        shape[2]);
    TT_FATAL(
        attrs.sequence_parallel_axis < in.input.device()->shape().dims(),
        "{}: invalid sequence_parallel_axis",
        operation_name);
}

ConvolutionHistoryRowsOperation::spec_return_value_t ConvolutionHistoryRowsOperation::compute_output_specs(
    const operation_attributes_t& attrs, const tensor_args_t&) {
    return {TensorSpec(
        Shape({1, history_table_rows, attrs.width}),
        TensorLayout(DataType::BFLOAT16, PageConfig(Layout::ROW_MAJOR), ttnn::DRAM_MEMORY_CONFIG))};
}

ConvolutionHistoryRowsOperation::tensor_return_value_t ConvolutionHistoryRowsOperation::create_output_tensors(
    const operation_attributes_t& attrs, const tensor_args_t& in) {
    return {create_device_tensor(compute_output_specs(attrs, in)[0], in.input.device())};
}

ttnn::device_operation::MeshWorkloadArtifacts ConvolutionHistoryRowsProgramFactory::create_mesh_workload_artifacts(
    const ConvolutionHistoryRowsParams& attrs,
    const ConvolutionHistoryRowsInputs& in,
    std::vector<Tensor>& outputs,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    using namespace tt::tt_metal::experimental;
    const auto& input = in.input.mesh_tensor();
    const auto& output = outputs[0];
    const auto& device = input.device();
    const uint32_t Wt = attrs.width / tt::constants::TILE_WIDTH;
    const uint32_t local_rows = in.input.logical_shape()[1];
    const auto grid = device.compute_with_storage_grid_size();
    const uint32_t workers = std::min<uint32_t>(grid.x * grid.y, (Wt + tiles_per_worker - 1) / tiles_per_worker);
    const uint32_t per_worker = (Wt + workers - 1) / workers;
    const CoreRangeSet core_set = tt::tt_metal::num_cores_to_corerangeset(workers, grid, /*row_wise=*/true);
    const auto cores = tt::tt_metal::corerange_to_cores(core_set, std::nullopt, true);

    const KernelSpecName reader_name{"reader"};
    const ScratchpadSpecName scratch_name{"scratch"};
    const TensorParamName input_tensor{"input"};
    const TensorParamName output_tensor{"output"};
    const TensorParamName actual_start_tensor{"actual_start"};

    KernelSpec reader{
        .unique_id = reader_name,
        .source =
            "ttnn/cpp/ttnn/operations/experimental/kda/qkv_causal_conv1d_silu/device/kernels/dataflow/"
            "reader_convolution_history_rows.cpp",
        .scratchpad_bindings = {{scratch_name, "scratch"}},
        .tensor_bindings =
            {
                TensorBinding{input_tensor, "input"},
                TensorBinding{output_tensor, "output"},
                TensorBinding{actual_start_tensor, "actual_start"},
            },
        .compile_time_args =
            {{"input_wt", input.padded_shape()[-1] / tt::constants::TILE_WIDTH},
             {"input_ct_offset", attrs.column_offset / tt::constants::TILE_WIDTH},
             {"max_tiles", per_worker}},
        .runtime_arg_schema = {.runtime_arg_names = {"ct_start", "ct_count"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    KernelRunArgs reader_run_args{.kernel = reader_name};
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const uint32_t begin = std::min(Wt, i * per_worker);
        const uint32_t end = std::min(Wt, begin + per_worker);
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, cores[i], {{"ct_start", begin}, {"ct_count", end - begin}});
    }
    constexpr uint32_t tile_bytes = 2048;
    ProgramSpec spec{
        .name = "convolution_history_rows",
        .kernels = {},
        .scratchpads = {{.unique_id = scratch_name, .size_per_node = per_worker * (tile_bytes / 2 + 64) + 128}},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = input_tensor, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = output_tensor, .spec = output.mesh_tensor().tensor_spec()},
                TensorParameter{.unique_id = actual_start_tensor, .spec = in.actual_start.mesh_tensor().tensor_spec()},
            },
        .work_units = {WorkUnitSpec{.name = "main", .kernels = {reader_name}, .target_nodes = core_set}},
    };
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {input_tensor, input},
        {output_tensor, output.mesh_tensor()},
        {actual_start_tensor, in.actual_start.mesh_tensor()},
    };
    kda_factory_detail::bind_actual_end(spec, run_args, in.actual_end, reader);
    spec.kernels = {std::move(reader)};
    run_args.kernel_run_args.push_back(std::move(reader_run_args));
    return kda_factory_detail::chronology_workload(
        ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)},
        tensor_coords,
        device,
        attrs.sequence_parallel_axis,
        local_rows,
        reader_name);
}

Tensor convolution_history_rows(
    const Tensor& input,
    const Tensor& actual_start,
    uint32_t width,
    uint32_t column_offset,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis) {
    return ttnn::device_operation::launch<ConvolutionHistoryRowsOperation>(
        ConvolutionHistoryRowsParams{
            .width = width, .column_offset = column_offset, .sequence_parallel_axis = sequence_parallel_axis},
        ConvolutionHistoryRowsInputs{.input = input, .actual_start = actual_start, .actual_end = actual_end})[0];
}

}  // namespace ttnn::experimental::prim
