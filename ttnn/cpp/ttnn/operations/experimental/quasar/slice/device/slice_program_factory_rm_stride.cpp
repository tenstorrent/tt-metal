// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/quasar/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/slice/device/slice_program_factory_rm_stride.hpp"

#include <optional>
#include <vector>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/spec_run_args.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramSpecArtifacts SliceRmStrideProgramFactory::create_program_spec(
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

    // Select kernels based on tensor rank
    const bool using_4d_kernels = input_shape.rank() <= 4;
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

    // Metal 2.0 named resource handles (locals, prefixed to avoid unity-build name collisions).
    const DFBSpecName C0{"slice_rmstride_c0"};  // legacy c_0: sliced-row stream (reader produces, writer consumes)
    const TensorParamName INPUT{"slice_rmstride_input"};
    const TensorParamName OUTPUT{"slice_rmstride_output"};
    const KernelSpecName READER{"slice_rmstride_reader"};
    const KernelSpecName WRITER{"slice_rmstride_writer"};

    // --- DataflowBuffer (legacy CB index 0, double-buffered) ---
    // total_size = 2 * cb_page_size_aligned -> entry_size = cb_page_size_aligned, num_entries = 2.
    DataflowBufferSpec c0_dfb{
        .unique_id = C0,
        .entry_size = cb_page_size_aligned,
        .num_entries = 2,
        .data_format_metadata = cb_data_format,
    };

    // --- TensorParameters ---
    // The reader binds INPUT, the writer binds OUTPUT. Both replace the legacy explicit
    // buffer-address RTA + TensorAccessorArgs CTAs.
    TensorParameter input_param{.unique_id = INPUT, .spec = input_tensor.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // Runtime-arg distribution: preserved EXACTLY from legacy.
    // base/extra rows use num_cores (cores with work); iterate all_cores_vec so idle cores
    // also receive args (rows_for_this_core == 0).
    uint32_t tensor_rank = input_shape.rank();
    uint32_t base_rows_per_core = total_output_rows / num_cores;
    uint32_t extra_rows = total_output_rows % num_cores;

    const auto& slice_start = args.slice_start;
    const auto& slice_end = args.slice_end;
    const auto& slice_step = args.step;

    auto all_cores_vec = corerange_to_cores(all_cores);

    // ------------------------------------------------------------------------
    // Build the kernel specs + run args. element_size is carried only as a named RTA
    // (the legacy compile_time slot1 CTA copy was never read in the kernel body).
    // ------------------------------------------------------------------------
    KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path{reader_kernel_path},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C0, .accessor_name = "cb_out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "in"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::READER},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path{writer_kernel_path},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = C0, .accessor_name = "cb_in", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "out"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    ProgramRunArgs run_args;

    if (using_4d_kernels) {
        // 4D path: fixed-length runtime arg lists -> all named RTAs (same order the kernel reads).
        reader_spec.runtime_arg_schema = {
            .runtime_arg_names = {
                "tensor_rank",  "input_w",      "input_h",      "input_d",      "input_n",
                "output_w",     "output_h",     "output_d",     "output_n",     "slice_start_w",
                "slice_end_w",  "slice_step_w", "slice_start_h", "slice_end_h", "slice_step_h",
                "slice_start_d", "slice_end_d", "slice_step_d", "slice_start_n", "slice_end_n",
                "slice_step_n", "element_size", "num_rows_for_this_core", "row_start_id"}};
        writer_spec.runtime_arg_schema = {
            .runtime_arg_names = {
                "tensor_rank", "output_w", "output_h", "output_d", "output_n", "element_size",
                "num_rows_for_this_core", "row_start_id"}};

        KernelRunArgs reader_run{.kernel = READER};
        KernelRunArgs writer_run{.kernel = WRITER};
        reader_run.runtime_arg_values.reserve(all_cores_vec.size());
        writer_run.runtime_arg_values.reserve(all_cores_vec.size());

        uint32_t row_start_id = 0;
        uint32_t extra_rows_remaining = extra_rows;
        for (const auto& core : all_cores_vec) {
            uint32_t rows_for_this_core = base_rows_per_core;
            if (extra_rows_remaining > 0) {
                rows_for_this_core += 1;
                extra_rows_remaining -= 1;
            }

            const NodeCoord node = core;
            reader_run.runtime_arg_values.push_back(
                {node,
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
                  {"row_start_id", row_start_id}}});

            writer_run.runtime_arg_values.push_back(
                {node,
                 {{"tensor_rank", tensor_rank},
                  {"output_w", output_shape[-1]},
                  {"output_h", output_shape[-2]},
                  {"output_d", output_shape[-3]},
                  {"output_n", output_shape[-4]},
                  {"element_size", element_size},
                  {"num_rows_for_this_core", rows_for_this_core},
                  {"row_start_id", row_start_id}}});

            row_start_id += rows_for_this_core;
        }

        run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    } else {
        // ND path: the shape/slice arrays are core-INVARIANT (identical across cores), and each
        // kernel reads them once (not per-core-varying) -> common runtime varargs.
        // Per-core scalars (num_rows_for_this_core, row_start_id) stay named RTAs.
        //
        // Reader common varargs layout (length 5*tensor_rank):
        //   [0*R, 1*R) input_shape, [1*R, 2*R) output_shape, [2*R, 3*R) slice_start,
        //   [3*R, 4*R) slice_end,   [4*R, 5*R) slice_step.
        // Writer common varargs layout (length tensor_rank): output_shape.
        reader_spec.runtime_arg_schema = {
            .runtime_arg_names = {"tensor_rank", "element_size", "num_rows_for_this_core", "row_start_id"}};
        reader_spec.advanced_options.num_common_runtime_varargs = 5 * tensor_rank;

        writer_spec.runtime_arg_schema = {
            .runtime_arg_names = {"tensor_rank", "element_size", "num_rows_for_this_core", "row_start_id"}};
        writer_spec.advanced_options.num_common_runtime_varargs = tensor_rank;

        std::vector<uint32_t> reader_common_varargs;
        reader_common_varargs.reserve(5 * tensor_rank);
        reader_common_varargs.insert(reader_common_varargs.end(), input_shape.cbegin(), input_shape.cend());
        reader_common_varargs.insert(reader_common_varargs.end(), output_shape.cbegin(), output_shape.cend());
        reader_common_varargs.insert(reader_common_varargs.end(), slice_start.cbegin(), slice_start.cend());
        reader_common_varargs.insert(reader_common_varargs.end(), slice_end.cbegin(), slice_end.cend());
        reader_common_varargs.insert(reader_common_varargs.end(), slice_step.cbegin(), slice_step.cend());

        std::vector<uint32_t> writer_common_varargs(output_shape.cbegin(), output_shape.cend());

        KernelRunArgs reader_run{.kernel = READER};
        KernelRunArgs writer_run{.kernel = WRITER};
        reader_run.runtime_arg_values.reserve(all_cores_vec.size());
        writer_run.runtime_arg_values.reserve(all_cores_vec.size());

        uint32_t row_start_id = 0;
        uint32_t extra_rows_remaining = extra_rows;
        for (const auto& core : all_cores_vec) {
            uint32_t rows_for_this_core = base_rows_per_core;
            if (extra_rows_remaining > 0) {
                rows_for_this_core += 1;
                extra_rows_remaining -= 1;
            }

            const NodeCoord node = core;
            reader_run.runtime_arg_values.push_back(
                {node,
                 {{"tensor_rank", tensor_rank},
                  {"element_size", element_size},
                  {"num_rows_for_this_core", rows_for_this_core},
                  {"row_start_id", row_start_id}}});
            writer_run.runtime_arg_values.push_back(
                {node,
                 {{"tensor_rank", tensor_rank},
                  {"element_size", element_size},
                  {"num_rows_for_this_core", rows_for_this_core},
                  {"row_start_id", row_start_id}}});

            row_start_id += rows_for_this_core;
        }

        run_args.kernel_run_args = {
            KernelRunArgs{
                .kernel = READER,
                .runtime_arg_values = std::move(reader_run.runtime_arg_values),
                .advanced_options = AdvancedKernelRunArgs{.common_runtime_varargs = std::move(reader_common_varargs)},
            },
            KernelRunArgs{
                .kernel = WRITER,
                .runtime_arg_values = std::move(writer_run.runtime_arg_values),
                .advanced_options = AdvancedKernelRunArgs{.common_runtime_varargs = std::move(writer_common_varargs)},
            },
        };
    }

    // --- Assemble ProgramSpec ---
    ProgramSpec spec{
        .name = "slice_rm_stride",
        .kernels = {reader_spec, writer_spec},
        .dataflow_buffers = {c0_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "slice_rm_stride_wu",
            .kernels = {READER, WRITER},
            .target_nodes = all_cores,
        }},
    };

    run_args.tensor_args.emplace(INPUT, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(OUTPUT, output.mesh_tensor());

    return ttnn::device_operation::ProgramSpecArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

}  // namespace ttnn::prim::qsr
