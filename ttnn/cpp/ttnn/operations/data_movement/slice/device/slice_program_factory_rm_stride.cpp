// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"

#include "ttnn/operations/data_movement/slice/device/slice_metal2_names.hpp"

#include <optional>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim {

ttnn::device_operation::ProgramArtifacts SliceRmStrideProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    using namespace ttnn::prim::slice_metal2;

    const auto& input_tensor = tensor_args.input;
    tt::tt_metal::IDevice* device = input_tensor.device();

    const auto& input_shape = input_tensor.padded_shape();
    const auto& output_shape = output.padded_shape();
    uint32_t element_size = input_tensor.element_size();

    // Calculate total output rows based on tensor rank
    uint32_t total_output_rows = output_shape.volume() / output_shape[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), total_output_rows)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_output_rows);

    // Select kernels based on tensor rank
    std::string reader_kernel_path;
    std::string writer_kernel_path;
    if (input_shape.rank() <= 4) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp";
    } else {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp";
    }

    tt::DataFormat dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t actual_input_w = input_shape[-1];
    uint32_t input_bytes_per_row = actual_input_w * element_size;
    uint32_t dfb_entry_size = input_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t dfb_entry_size_aligned = tt::round_up(dfb_entry_size, alignment);

    const uint32_t tensor_rank = input_shape.rank();
    const bool using_4d_kernels = input_shape.rank() <= 4;

    DataflowBufferSpec dfb_in{
        .unique_id = STRIDE_IN,
        .entry_size = dfb_entry_size_aligned,
        .num_entries = 2,
        .data_format_metadata = dfb_data_format,
    };

    // The *_nd pair reads five rank-long blocks (reader) and one (writer); the *_4d pair spells the
    // same values out as a fixed run of named scalars, so it takes no varargs at all.
    const uint32_t reader_num_varargs = using_4d_kernels ? 0u : (5u * tensor_rank);
    const uint32_t writer_num_varargs = using_4d_kernels ? 0u : tensor_rank;

    Group<std::string> reader_arg_names;
    Group<std::string> writer_arg_names;
    if (using_4d_kernels) {
        reader_arg_names = {
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
        writer_arg_names = {
            "tensor_rank",
            "output_w",
            "output_h",
            "output_d",
            "output_n",
            "element_size",
            "num_rows_for_this_core",
            "start_row_for_this_core"};
    } else {
        reader_arg_names = {"tensor_rank", "element_size", "num_rows_for_this_core", "start_row_for_this_core"};
        writer_arg_names = reader_arg_names;
    }

    KernelSpec reader{
        .unique_id = STRIDE_READER,
        .source = reader_kernel_path,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STRIDE_IN,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = INPUT,
                    .accessor_name = "src",
                },
            },
        .compile_time_args = {{"compile_time_element_size", element_size}},
        .runtime_arg_schema = {.runtime_arg_names = reader_arg_names},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = reader_num_varargs},
    };

    KernelSpec writer{
        .unique_id = STRIDE_WRITER,
        .source = writer_kernel_path,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STRIDE_IN,
                    .accessor_name = "in",
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
        .compile_time_args = {{"compile_time_element_size", element_size}},
        .runtime_arg_schema = {.runtime_arg_names = writer_arg_names},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = writer_num_varargs},
    };

    // Calculate runtime arguments
    uint32_t base_rows_per_core = total_output_rows / num_cores;
    uint32_t extra_rows = total_output_rows % num_cores;

    const auto& slice_start = args.slice_start;
    const auto& slice_end = args.slice_end;
    const auto& slice_step = args.step;

    auto all_cores_vec = corerange_to_cores(all_cores);

    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;

    KernelRunArgs reader_run_args{.kernel = STRIDE_READER};
    KernelRunArgs writer_run_args{.kernel = STRIDE_WRITER};

    for (const auto& core : all_cores_vec) {
        uint32_t rows_for_this_core = base_rows_per_core;
        if (extra_rows_remaining > 0) {
            rows_for_this_core += 1;
            extra_rows_remaining -= 1;
        }

        if (using_4d_kernels) {
            AddRuntimeArgsForNode(
                reader_run_args.runtime_arg_values,
                core,
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
                core,
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
                core,
                {{"tensor_rank", tensor_rank},
                 {"element_size", element_size},
                 {"num_rows_for_this_core", rows_for_this_core},
                 {"start_row_for_this_core", row_start_id}});

            std::vector<uint32_t> reader_varargs;
            reader_varargs.reserve(5 * tensor_rank);
            reader_varargs.insert(reader_varargs.end(), input_shape.cbegin(), input_shape.cend());
            reader_varargs.insert(reader_varargs.end(), output_shape.cbegin(), output_shape.cend());
            reader_varargs.insert(reader_varargs.end(), slice_start.cbegin(), slice_start.cend());
            reader_varargs.insert(reader_varargs.end(), slice_end.cbegin(), slice_end.cend());
            reader_varargs.insert(reader_varargs.end(), slice_step.cbegin(), slice_step.cend());
            reader_run_args.advanced_options.runtime_varargs[core] = std::move(reader_varargs);

            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"tensor_rank", tensor_rank},
                 {"element_size", element_size},
                 {"num_rows_for_this_core", rows_for_this_core},
                 {"start_row_for_this_core", row_start_id}});

            writer_run_args.advanced_options.runtime_varargs[core] =
                std::vector<uint32_t>(output_shape.cbegin(), output_shape.cend());
        }

        row_start_id += rows_for_this_core;
    }

    ProgramSpec spec{
        .name = "slice_rm_stride",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(dfb_in)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_tensor.tensor_spec()},
                TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {STRIDE_READER, STRIDE_WRITER},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {INPUT, input_tensor.mesh_tensor()},
        {OUTPUT, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::experimental::ProgramRunArgs SliceRmStrideProgramFactory::override_runtime_arguments(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    return slice_program_run_args(SliceRmStrideProgramFactory{}, args, tensor_args, output);
}

}  // namespace ttnn::prim
