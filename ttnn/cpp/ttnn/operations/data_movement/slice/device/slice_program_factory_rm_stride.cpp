// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"

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
struct StrideSpecNames {
    KernelSpecName reader{"reader"};
    KernelSpecName writer{"writer"};
    // The accessor names are the kernels' own vocabulary: the reader fills what it calls its output
    // buffer, and the writer drains what it calls its input buffer. Same DFB either way.
    DFBSpecName in_dfb{"in_dfb"};
    TensorParamName input{"input"};
    TensorParamName output{"output"};
};

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceRmStrideProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const StrideSpecNames names;

    const auto& input_tensor = tensor_args.input;
    tt::tt_metal::IDevice* device = input_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    const uint32_t element_size = input_tensor.element_size();

    // Calculate total output rows based on tensor rank
    const uint32_t total_output_rows = output_shape.volume() / output_shape[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), total_output_rows)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_output_rows);

    // Select kernels based on tensor rank
    const bool using_4d_kernels = input_shape.rank() <= 4;
    const std::string reader_kernel_path =
        using_4d_kernels
            ? "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp";
    const std::string writer_kernel_path =
        using_4d_kernels
            ? "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp"
            : "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp";

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    const uint32_t actual_input_w = input_shape[-1];
    const uint32_t input_bytes_per_row = actual_input_w * element_size;
    const uint32_t dfb_entry_size = input_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    const uint32_t dfb_entry_size_aligned = tt::round_up(dfb_entry_size, alignment);

    const DataflowBufferSpec in_dfb{
        .unique_id = names.in_dfb,
        .entry_size = dfb_entry_size_aligned,
        .num_entries = 2,
        .data_format_metadata = dfb_data_format,
    };

    const TensorParameter input_param{.unique_id = names.input, .spec = input_tensor.tensor_spec()};
    const TensorParameter output_param{.unique_id = names.output, .spec = output.tensor_spec()};

    const uint32_t tensor_rank = input_shape.rank();

    // Both kernel pairs still declare (and ignore) a compile-time element size; the runtime
    // `element_size` is what they actually use. Kept so the port changes no behaviour.
    KernelSpec::CompileTimeArgs shared_compile_time_args{{"compile_time_element_size", element_size}};

    KernelSpec reader{
        .unique_id = names.reader,
        .source = reader_kernel_path,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.in_dfb,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = names.input, .accessor_name = "src"},
            },
        .compile_time_args = shared_compile_time_args,
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    KernelSpec writer{
        .unique_id = names.writer,
        .source = writer_kernel_path,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = names.in_dfb,
                    .accessor_name = "in",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = names.output, .accessor_name = "dst"},
            },
        .compile_time_args = shared_compile_time_args,
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    if (using_4d_kernels) {
        // A fixed run of distinct fields on both sides: every one is a named argument.
        reader.runtime_arg_schema.runtime_arg_names = {
            "tensor_rank",
            "input_w",
            "input_h",
            "input_d",
            "input_n",
            "output_w",
            "output_h",
            "output_d",
            "output_n",
            "slice_start_w",
            "slice_end_w",
            "slice_step_w",
            "slice_start_h",
            "slice_end_h",
            "slice_step_h",
            "slice_start_d",
            "slice_end_d",
            "slice_step_d",
            "slice_start_n",
            "slice_end_n",
            "slice_step_n",
            "element_size",
            "num_rows_for_this_core",
            "start_row_for_this_core"};
        writer.runtime_arg_schema.runtime_arg_names = {
            "tensor_rank",
            "output_w",
            "output_h",
            "output_d",
            "output_n",
            "element_size",
            "num_rows_for_this_core",
            "start_row_for_this_core"};
    } else {
        // The dimension / slice-parameter blocks are tensor_rank long, so they travel as varargs.
        reader.runtime_arg_schema.runtime_arg_names = {
            "tensor_rank", "element_size", "num_rows_for_this_core", "start_row_for_this_core"};
        reader.advanced_options.num_runtime_varargs = tensor_rank * 5;
        writer.runtime_arg_schema.runtime_arg_names = {
            "tensor_rank", "element_size", "num_rows_for_this_core", "start_row_for_this_core"};
        writer.advanced_options.num_runtime_varargs = tensor_rank;
    }

    // Per-node work distribution. Note this is NOT the split_work_to_cores group split: the factory
    // re-derives its own even spread with a one-row remainder, and the port preserves that.
    const uint32_t base_rows_per_core = total_output_rows / num_cores;
    const uint32_t extra_rows = total_output_rows % num_cores;

    const auto& slice_start = args.slice_start;
    const auto& slice_end = args.slice_end;
    const auto& slice_step = args.step;

    KernelRunArgs reader_run_args{.kernel = names.reader};
    KernelRunArgs writer_run_args{.kernel = names.writer};

    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;

    for (const auto& node : corerange_to_cores(all_cores)) {
        uint32_t rows_for_this_core = base_rows_per_core;
        if (extra_rows_remaining > 0) {
            rows_for_this_core += 1;
            extra_rows_remaining -= 1;
        }

        if (using_4d_kernels) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                node,
                {{"tensor_rank", tensor_rank},
                 {"input_w", input_shape[-1]},
                 {"input_h", input_shape[-2]},
                 {"input_d", input_shape[-3]},
                 {"input_n", input_shape[-4]},
                 {"output_w", output_shape[-1]},
                 {"output_h", output_shape[-2]},
                 {"output_d", output_shape[-3]},
                 {"output_n", output_shape[-4]},
                 {"slice_start_w", slice_start[-1]},
                 {"slice_end_w", slice_end[-1]},
                 {"slice_step_w", slice_step[-1]},
                 {"slice_start_h", slice_start[-2]},
                 {"slice_end_h", slice_end[-2]},
                 {"slice_step_h", slice_step[-2]},
                 {"slice_start_d", slice_start[-3]},
                 {"slice_end_d", slice_end[-3]},
                 {"slice_step_d", slice_step[-3]},
                 {"slice_start_n", slice_start[-4]},
                 {"slice_end_n", slice_end[-4]},
                 {"slice_step_n", slice_step[-4]},
                 {"element_size", element_size},
                 {"num_rows_for_this_core", rows_for_this_core},
                 {"start_row_for_this_core", row_start_id}});

            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                node,
                {{"tensor_rank", tensor_rank},
                 {"output_w", output_shape[-1]},
                 {"output_h", output_shape[-2]},
                 {"output_d", output_shape[-3]},
                 {"output_n", output_shape[-4]},
                 {"element_size", element_size},
                 {"num_rows_for_this_core", rows_for_this_core},
                 {"start_row_for_this_core", row_start_id}});
        } else {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                node,
                {{"tensor_rank", tensor_rank},
                 {"element_size", element_size},
                 {"num_rows_for_this_core", rows_for_this_core},
                 {"start_row_for_this_core", row_start_id}});
            // Five tensor_rank-long blocks, in the order the kernel walks them.
            AdvancedKernelRunArgs::Varargs reader_varargs;
            reader_varargs.reserve(tensor_rank * 5);
            reader_varargs.insert(reader_varargs.end(), input_shape.cbegin(), input_shape.cend());
            reader_varargs.insert(reader_varargs.end(), output_shape.cbegin(), output_shape.cend());
            reader_varargs.insert(reader_varargs.end(), slice_start.cbegin(), slice_start.cend());
            reader_varargs.insert(reader_varargs.end(), slice_end.cbegin(), slice_end.cend());
            reader_varargs.insert(reader_varargs.end(), slice_step.cbegin(), slice_step.cend());
            reader_run_args.advanced_options.runtime_varargs[node] = std::move(reader_varargs);

            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                node,
                {{"tensor_rank", tensor_rank},
                 {"element_size", element_size},
                 {"num_rows_for_this_core", rows_for_this_core},
                 {"start_row_for_this_core", row_start_id}});
            writer_run_args.advanced_options.runtime_varargs[node] =
                AdvancedKernelRunArgs::Varargs(output_shape.cbegin(), output_shape.cend());
        }

        row_start_id += rows_for_this_core;
    }

    ProgramSpec spec{
        .name = "slice_rm_stride",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {in_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "slice",
            .kernels = {names.reader, names.writer},
            .target_nodes = all_cores,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{names.input, input_tensor.mesh_tensor()}, {names.output, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs SliceRmStrideProgramFactory::override_runtime_arguments(
    const SliceParams& /*args*/,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const StrideSpecNames names;

    // The legacy refresh for this factory re-pointed the two buffer addresses and nothing else; both
    // now travel as tensor bindings, so re-supplying them is the whole job.
    ProgramRunArgs run_args;
    run_args.tensor_args = {{names.input, tensor_args.input.mesh_tensor()}, {names.output, output.mesh_tensor()}};
    return run_args;
}

}  // namespace ttnn::prim
