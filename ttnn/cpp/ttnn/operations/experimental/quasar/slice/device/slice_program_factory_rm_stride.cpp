// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory_rm_stride.hpp"

#include <optional>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts SliceRmStrideProgramFactory::create_program_artifacts(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
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

    const uint32_t tensor_rank = input_shape.rank();
    const bool using_4d_kernels = tensor_rank <= 4;

    // Select kernels based on tensor rank
    std::string reader_kernel_path;
    std::string writer_kernel_path;
    if (using_4d_kernels) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/reader_multicore_slice_4d.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/writer_multicore_slice_4d.cpp";
    } else {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/reader_multicore_slice_nd.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/experimental/quasar/slice/device/kernels/dataflow/writer_multicore_slice_nd.cpp";
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t actual_input_w = input_shape[-1];
    uint32_t input_bytes_per_row = actual_input_w * element_size;
    uint32_t cb_page_size = input_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t cb_page_size_aligned = tt::round_up(cb_page_size, alignment);

    // Resource names
    const DFBSpecName C0{"c0"};  // src->dst staging FIFO (legacy CB in_cb = c_0)
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};

    // --- DataflowBuffer (legacy in_cb, double-buffered) ---
    DataflowBufferSpec c0_dfb{
        .unique_id = C0,
        .entry_size = cb_page_size_aligned,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    };

    // --- Reader / Writer KernelSpecs ---
    // The 4D path keeps tensor_rank / element_size as per-core RTAs (faithful to legacy);
    // the ND path makes them CTAs and reads the per-program dim/slice arrays from common varargs.
    KernelSpec reader;
    reader.unique_id = READER;
    reader.source = reader_kernel_path;
    reader.dfb_bindings = {
        DFBBinding{.dfb_spec_name = C0, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER}};
    reader.tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "in"}};
    reader.hw_config = ttnn::create_reader_datamovement_config(device->arch());

    KernelSpec writer;
    writer.unique_id = WRITER;
    writer.source = writer_kernel_path;
    writer.dfb_bindings = {
        DFBBinding{.dfb_spec_name = C0, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER}};
    writer.tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"}};
    writer.hw_config = ttnn::create_writer_datamovement_config(device->arch());

    const auto& slice_start = args.slice_start;
    const auto& slice_end = args.slice_end;
    const auto& slice_step = args.step;

    // Common runtime varargs for the ND path (identical for all cores).
    std::vector<uint32_t> reader_common_varargs;
    std::vector<uint32_t> writer_common_varargs;

    if (using_4d_kernels) {
        reader.compile_time_args = {{"compile_time_element_size", element_size}};
        reader.runtime_arg_schema = {
            .runtime_arg_names = {
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
                "start_row_for_this_core"}};

        writer.compile_time_args = {{"compile_time_element_size", element_size}};
        writer.runtime_arg_schema = {
            .runtime_arg_names = {
                "tensor_rank",
                "output_w",
                "output_h",
                "output_d",
                "output_n",
                "element_size",
                "num_rows_for_this_core",
                "start_row_for_this_core"}};
    } else {
        reader.compile_time_args = {{"tensor_rank", tensor_rank}, {"element_size", element_size}};
        reader.runtime_arg_schema = {.runtime_arg_names = {"num_rows_for_this_core", "start_row_for_this_core"}};
        reader.advanced_options = {.num_common_runtime_varargs = 5 * tensor_rank};

        writer.compile_time_args = {{"tensor_rank", tensor_rank}, {"element_size", element_size}};
        writer.runtime_arg_schema = {.runtime_arg_names = {"num_rows_for_this_core", "start_row_for_this_core"}};
        writer.advanced_options = {.num_common_runtime_varargs = tensor_rank};

        // Reader common-vararg layout: input_dims, output_dims, slice_starts, slice_ends, slice_steps.
        reader_common_varargs.reserve(5 * tensor_rank);
        reader_common_varargs.insert(reader_common_varargs.end(), input_shape.cbegin(), input_shape.cend());
        reader_common_varargs.insert(reader_common_varargs.end(), output_shape.cbegin(), output_shape.cend());
        reader_common_varargs.insert(reader_common_varargs.end(), slice_start.cbegin(), slice_start.cend());
        reader_common_varargs.insert(reader_common_varargs.end(), slice_end.cbegin(), slice_end.cend());
        reader_common_varargs.insert(reader_common_varargs.end(), slice_step.cbegin(), slice_step.cend());

        // Writer common-vararg layout: output_dims.
        writer_common_varargs.assign(output_shape.cbegin(), output_shape.cend());
    }

    // --- Per-core runtime args ---
    Group<KernelRunArgs::NodeRuntimeArgs> reader_node_args;
    Group<KernelRunArgs::NodeRuntimeArgs> writer_node_args;

    uint32_t base_rows_per_core = total_output_rows / num_cores;
    uint32_t extra_rows = total_output_rows % num_cores;
    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;

    for (const auto& core : corerange_to_cores(all_cores)) {
        uint32_t rows_for_this_core = base_rows_per_core;
        if (extra_rows_remaining > 0) {
            rows_for_this_core += 1;
            extra_rows_remaining -= 1;
        }

        if (using_4d_kernels) {
            reader_node_args.push_back(
                {.node = core,
                 .args = {
                     {"tensor_rank", tensor_rank},
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
                     {"start_row_for_this_core", row_start_id}}});

            writer_node_args.push_back(
                {.node = core,
                 .args = {
                     {"tensor_rank", tensor_rank},
                     {"output_w", output_shape[-1]},
                     {"output_h", output_shape[-2]},
                     {"output_d", output_shape[-3]},
                     {"output_n", output_shape[-4]},
                     {"element_size", element_size},
                     {"num_rows_for_this_core", rows_for_this_core},
                     {"start_row_for_this_core", row_start_id}}});
        } else {
            reader_node_args.push_back(
                {.node = core,
                 .args = {{"num_rows_for_this_core", rows_for_this_core}, {"start_row_for_this_core", row_start_id}}});
            writer_node_args.push_back(
                {.node = core,
                 .args = {{"num_rows_for_this_core", rows_for_this_core}, {"start_row_for_this_core", row_start_id}}});
        }

        row_start_id += rows_for_this_core;
    }

    // --- TensorParameters ---
    TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // --- Assemble ProgramSpec ---
    ProgramSpec spec;
    spec.name = "slice_rm_stride";
    spec.kernels = {reader, writer};
    spec.dataflow_buffers = {c0_dfb};
    spec.tensor_parameters = {input_param, output_param};
    spec.work_units = {WorkUnitSpec{
        .name = "slice_rm_stride_wu",
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
            .advanced_options = AdvancedKernelRunArgs{.common_runtime_varargs = std::move(reader_common_varargs)},
        },
        KernelRunArgs{
            .kernel = WRITER,
            .runtime_arg_values = std::move(writer_node_args),
            .common_runtime_arg_values = {},
            .advanced_options = AdvancedKernelRunArgs{.common_runtime_varargs = std::move(writer_common_varargs)},
        },
    };
    run_args.tensor_args.emplace(INPUT, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
