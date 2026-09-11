// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_metal2_names.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"

#include <optional>
#include <string>
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

// The one dataflow buffer this factory declares: the reader stages each sliced row here and the
// writer drains it. The identifier is factory-prefixed because sibling slice factories declare their
// own buffers and this target is a unity build.
const DFBSpecName STRIDE_ROW{"row"};

}  // namespace

ttnn::device_operation::ProgramArtifacts SliceRmStrideProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    using slice_metal2::INPUT;
    using slice_metal2::OUTPUT;
    using slice_metal2::READER;
    using slice_metal2::WRITER;

    const auto& input_tensor = tensor_args.input;
    tt::tt_metal::IDevice* device = input_tensor.device();
    const auto& input_mesh_tensor = input_tensor.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();

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
    uint32_t dfb_page_size = input_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t dfb_page_size_aligned = tt::round_up(dfb_page_size, alignment);

    DataflowBufferSpec row_dfb{
        .unique_id = STRIDE_ROW,
        .entry_size = dfb_page_size_aligned,
        .num_entries = 2,
        .data_format_metadata = dfb_data_format,
    };

    // Calculate runtime arguments
    uint32_t tensor_rank = input_shape.rank();
    uint32_t base_rows_per_core = total_output_rows / num_cores;
    uint32_t extra_rows = total_output_rows % num_cores;

    const bool using_4d_kernels = input_shape.rank() <= 4;
    const auto& slice_start = args.slice_start;
    const auto& slice_end = args.slice_end;
    const auto& slice_step = args.step;

    // The 4D kernels read a fixed run of scalars, so every one is a named argument. The
    // N-dimensional kernels read rank-length blocks whose length is only known at run time, so those
    // blocks are runtime varargs and only the leading scalars are named.
    KernelSpec::RuntimeArgSchema reader_schema;
    KernelSpec::RuntimeArgSchema writer_schema;
    if (using_4d_kernels) {
        reader_schema.runtime_arg_names = {
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
        writer_schema.runtime_arg_names = {
            "tensor_rank",
            "output_w",
            "output_h",
            "output_d",
            "output_n",
            "element_size",
            "num_rows_for_this_core",
            "start_row_for_this_core"};
    } else {
        reader_schema.runtime_arg_names = {
            "tensor_rank", "element_size", "num_rows_for_this_core", "start_row_for_this_core"};
        writer_schema.runtime_arg_names = reader_schema.runtime_arg_names;
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_path,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STRIDE_ROW,
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
        // `compile_time_element_size` is read into a local the kernel never uses. It is carried
        // across unchanged: dropping it would change the kernel's argument set, which is a
        // functional change rather than a translation.
        .compile_time_args = {{"compile_time_element_size", element_size}},
        .runtime_arg_schema = reader_schema,
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = using_4d_kernels ? 0 : (5 * tensor_rank)},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_path,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = STRIDE_ROW,
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
        .runtime_arg_schema = writer_schema,
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
        .advanced_options = {.num_runtime_varargs = using_4d_kernels ? 0 : tensor_rank},
    };

    auto all_cores_vec = corerange_to_cores(all_cores);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;

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
            AddRuntimeArgsForNode(
                writer_run_args.runtime_arg_values,
                core,
                {{"tensor_rank", tensor_rank},
                 {"element_size", element_size},
                 {"num_rows_for_this_core", rows_for_this_core},
                 {"start_row_for_this_core", row_start_id}});

            // The five blocks are shape-derived and so identical on every node. They stay per-node
            // runtime varargs because that is how the legacy factory dispatched them; promoting them
            // to common runtime varargs would change dispatch semantics.
            AdvancedKernelRunArgs::Varargs reader_varargs;
            reader_varargs.reserve(5 * tensor_rank);
            reader_varargs.insert(reader_varargs.end(), input_shape.cbegin(), input_shape.cend());
            reader_varargs.insert(reader_varargs.end(), output_shape.cbegin(), output_shape.cend());
            reader_varargs.insert(reader_varargs.end(), slice_start.cbegin(), slice_start.cend());
            reader_varargs.insert(reader_varargs.end(), slice_end.cbegin(), slice_end.cend());
            reader_varargs.insert(reader_varargs.end(), slice_step.cbegin(), slice_step.cend());
            reader_run_args.advanced_options.runtime_varargs[core] = std::move(reader_varargs);

            writer_run_args.advanced_options.runtime_varargs[core] =
                AdvancedKernelRunArgs::Varargs(output_shape.cbegin(), output_shape.cend());
        }

        row_start_id += rows_for_this_core;
    }

    ProgramSpec spec{
        .name = "slice_rm_stride",
        .kernels = {std::move(reader), std::move(writer)},
        .dataflow_buffers = {std::move(row_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = INPUT, .spec = input_mesh_tensor.tensor_spec()},
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
        {OUTPUT, output_mesh_tensor},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

tt::tt_metal::experimental::ProgramRunArgs SliceRmStrideProgramFactory::override_runtime_arguments(
    const SliceParams& /*args*/,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Every other argument this factory emits is shape-derived, and both tensor specs and the slice
    // params are folded into compute_program_hash(), so a cache hit can only differ in where the two
    // tensors live. Re-binding them is the whole of the per-dispatch refresh, matching the set the
    // ported-from override re-pointed.
    ProgramRunArgs run_args;
    run_args.tensor_args = {
        {slice_metal2::INPUT, tensor_args.input.mesh_tensor()},
        {slice_metal2::OUTPUT, output.mesh_tensor()},
    };
    return run_args;
}

}  // namespace ttnn::prim
