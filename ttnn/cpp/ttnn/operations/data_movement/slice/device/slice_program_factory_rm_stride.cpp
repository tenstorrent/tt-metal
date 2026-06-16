// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"

#include <algorithm>
#include <optional>
#include <filesystem>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "ttnn/metal2_artifacts.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor SliceRmStrideProgramFactory::create_descriptor(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    tt::tt_metal::IDevice* device = input_tensor.device();
    ProgramDescriptor desc;

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

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t actual_input_w = input_shape[-1];
    uint32_t input_bytes_per_row = actual_input_w * element_size;
    uint32_t cb_page_size = input_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t cb_page_size_aligned = tt::round_up(cb_page_size, alignment);
    uint32_t cb_total_size = 2 * cb_page_size_aligned;

    constexpr uint8_t in_cb = 0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_total_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = in_cb,
            .data_format = cb_data_format,
            .page_size = cb_page_size_aligned,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*input_tensor.buffer()).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {in_cb, element_size};
    TensorAccessorArgs(*output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = writer_kernel_path;
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Calculate runtime arguments
    uint32_t tensor_rank = input_shape.rank();
    uint32_t base_rows_per_core = total_output_rows / num_cores;
    uint32_t extra_rows = total_output_rows % num_cores;

    const bool using_4d_kernels = input_shape.rank() <= 4;
    const auto& slice_start = args.slice_start;
    const auto& slice_end = args.slice_end;
    const auto& slice_step = args.step;

    auto all_cores_vec = corerange_to_cores(all_cores);
    reader_desc.runtime_args.reserve(all_cores_vec.size());
    writer_desc.runtime_args.reserve(all_cores_vec.size());

    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;

    auto* input_buffer = input_tensor.buffer();
    auto* output_buffer = output.buffer();

    for (const auto& core : all_cores_vec) {
        uint32_t rows_for_this_core = base_rows_per_core;
        if (extra_rows_remaining > 0) {
            rows_for_this_core += 1;
            extra_rows_remaining -= 1;
        }

        if (using_4d_kernels) {
            reader_desc.emplace_runtime_args(
                core, {input_buffer,    tensor_rank,      input_shape[-1],  input_shape[-2],    input_shape[-3],
                       input_shape[-4], output_shape[-1], output_shape[-2], output_shape[-3],   output_shape[-4],
                       slice_start[-1], slice_end[-1],    slice_step[-1],   slice_start[-2],    slice_end[-2],
                       slice_step[-2],  slice_start[-3],  slice_end[-3],    slice_step[-3],     slice_start[-4],
                       slice_end[-4],   slice_step[-4],   element_size,     rows_for_this_core, row_start_id});

            writer_desc.emplace_runtime_args(
                core,
                {output_buffer,
                 tensor_rank,
                 output_shape[-1],
                 output_shape[-2],
                 output_shape[-3],
                 output_shape[-4],
                 element_size,
                 rows_for_this_core,
                 row_start_id});
        } else {
            KernelDescriptor::RTArgList reader_args;
            reader_args.push_back(input_buffer);
            reader_args.push_back(tensor_rank);
            reader_args.push_back(element_size);
            reader_args.push_back(rows_for_this_core);
            reader_args.push_back(row_start_id);
            reader_args.append(std::vector<uint32_t>(input_shape.cbegin(), input_shape.cend()));
            reader_args.append(std::vector<uint32_t>(output_shape.cbegin(), output_shape.cend()));
            reader_args.append(std::vector<uint32_t>(slice_start.cbegin(), slice_start.cend()));
            reader_args.append(std::vector<uint32_t>(slice_end.cbegin(), slice_end.cend()));
            reader_args.append(std::vector<uint32_t>(slice_step.cbegin(), slice_step.cend()));
            reader_desc.emplace_runtime_args(core, reader_args);

            KernelDescriptor::RTArgList writer_args;
            writer_args.push_back(output_buffer);
            writer_args.push_back(tensor_rank);
            writer_args.push_back(element_size);
            writer_args.push_back(rows_for_this_core);
            writer_args.push_back(row_start_id);
            writer_args.append(std::vector<uint32_t>(output_shape.cbegin(), output_shape.cend()));
            writer_desc.emplace_runtime_args(core, writer_args);
        }

        row_start_id += rows_for_this_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

// Metal 2.0 (ProgramSpec) port of the ROW_MAJOR strided factory. Mirrors create_descriptor's work
// split, kernel selection, and per-core runtime-arg computation exactly, expressed with the Metal
// 2.0 host API and pointing at the forked *_m2 kernels.
//
// Case-1 port (no Case-2 bridge needed):
//   - src/dst addresses move from clean Buffer* RTAs into TensorParameter/TensorBinding (ta::src /
//     ta::dst). The legacy kernels built TensorAccessor(args, addr) with no page-size override, so
//     the binding-token accessor's default aligned page size matches exactly; both feed
//     noc_async_*_sharded with identical behavior.
//   - The single in CB (index 0) becomes one DataflowBufferSpec, reader=PRODUCER / writer=CONSUMER,
//     sharing one WorkUnitSpec on all_cores (local-DFB rule).
//   - element_size leading CTA -> named CTA compile_time_element_size.
//   - 4D kernels: fixed positional RTAs -> named RTAs. ND kernels: leading scalar RTAs -> named
//     RTAs, the per-dim arrays -> runtime varargs (same concatenated layout).
ttnn::device_operation::ProgramArtifacts SliceRmStrideSpecProgramFactory::create_program_spec(
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
    (void)core_group_1;
    (void)core_group_2;
    (void)num_rows_per_core_group_1;
    (void)num_rows_per_core_group_2;

    const bool using_4d_kernels = input_shape.rank() <= 4;

    // Select kernels based on tensor rank (forked *_m2 kernels).
    std::string reader_kernel_path;
    std::string writer_kernel_path;
    if (using_4d_kernels) {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_4d_m2.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_4d_m2.cpp";
    } else {
        reader_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd_m2.cpp";
        writer_kernel_path =
            "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd_m2.cpp";
    }

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t actual_input_w = input_shape[-1];
    uint32_t input_bytes_per_row = actual_input_w * element_size;
    uint32_t cb_page_size = input_bytes_per_row;

    auto src_buffer_alignment = input_tensor.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t cb_page_size_aligned = tt::round_up(cb_page_size, alignment);
    uint32_t cb_total_size = 2 * cb_page_size_aligned;

    m2::ProgramSpec spec;
    spec.name = "slice_rm_stride";

    // --- DFB (was: in CB index 0, double-buffered single-row FIFO) ---
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = m2::DFBSpecName{"cb_in_out"},
            .entry_size = cb_page_size_aligned,
            .num_entries = cb_total_size / cb_page_size_aligned,
            .data_format_metadata = cb_data_format,
        },
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{reader_kernel_path},
        .compiler_options = {},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_in_out"},
                    .accessor_name = "cb_in_out",
                    .endpoint_type = m2::DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"},
            },
        .compile_time_args = {{"compile_time_element_size", element_size}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::READER},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{writer_kernel_path},
        .compiler_options = {},
        .dfb_bindings =
            {
                m2::DFBBinding{
                    .dfb_spec_name = m2::DFBSpecName{"cb_in_out"},
                    .accessor_name = "cb_in_out",
                    .endpoint_type = m2::DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"},
            },
        .compile_time_args = {{"compile_time_element_size", element_size}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementRoleHint::WRITER},
    };

    if (using_4d_kernels) {
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
        // ND: leading scalar RTAs are named; the per-dim arrays travel as varargs.
        reader.runtime_arg_schema = {
            .runtime_arg_names = {"tensor_rank", "element_size", "num_rows_for_this_core", "start_row_for_this_core"}};
        // 5 arrays of length input_shape.rank() (input/output dims, slice start/end/step).
        reader.advanced_options.num_runtime_varargs = 5 * input_shape.rank();
        writer.runtime_arg_schema = {
            .runtime_arg_names = {"tensor_rank", "element_size", "num_rows_for_this_core", "start_row_for_this_core"}};
        // 1 array of length output_shape.rank() (output dims).
        writer.advanced_options.num_runtime_varargs = output_shape.rank();
    }

    spec.kernels = {reader, writer};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = input_tensor.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()},
    };
    spec.work_units = std::vector<m2::WorkUnitSpec>{
        m2::WorkUnitSpec{
            .name = "slice_rm_stride",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
            .target_nodes = all_cores,
        },
    };

    // --- ProgramRunArgs (per-core; mirrors create_descriptor's base+extra row distribution) ---
    uint32_t tensor_rank = input_shape.rank();
    uint32_t base_rows_per_core = total_output_rows / num_cores;
    uint32_t extra_rows = total_output_rows % num_cores;

    const auto& slice_start = args.slice_start;
    const auto& slice_end = args.slice_end;
    const auto& slice_step = args.step;

    m2::ProgramRunArgs run;
    m2::KernelRunArgs reader_run{.kernel = m2::KernelSpecName{"reader"}};
    m2::KernelRunArgs writer_run{.kernel = m2::KernelSpecName{"writer"}};

    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;

    for (const auto& core : corerange_to_cores(all_cores)) {
        const m2::NodeCoord node{core};
        uint32_t rows_for_this_core = base_rows_per_core;
        if (extra_rows_remaining > 0) {
            rows_for_this_core += 1;
            extra_rows_remaining -= 1;
        }

        if (using_4d_kernels) {
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
                  {"start_row_for_this_core", row_start_id}}});

            writer_run.runtime_arg_values.push_back(
                {node,
                 {{"tensor_rank", tensor_rank},
                  {"output_w", output_shape[-1]},
                  {"output_h", output_shape[-2]},
                  {"output_d", output_shape[-3]},
                  {"output_n", output_shape[-4]},
                  {"element_size", element_size},
                  {"num_rows_for_this_core", rows_for_this_core},
                  {"start_row_for_this_core", row_start_id}}});
        } else {
            reader_run.runtime_arg_values.push_back(
                {node,
                 {{"tensor_rank", tensor_rank},
                  {"element_size", element_size},
                  {"num_rows_for_this_core", rows_for_this_core},
                  {"start_row_for_this_core", row_start_id}}});
            m2::AdvancedKernelRunArgs::Varargs reader_varargs;
            reader_varargs.reserve(5 * tensor_rank);
            reader_varargs.insert(reader_varargs.end(), input_shape.cbegin(), input_shape.cend());
            reader_varargs.insert(reader_varargs.end(), output_shape.cbegin(), output_shape.cend());
            reader_varargs.insert(reader_varargs.end(), slice_start.cbegin(), slice_start.cend());
            reader_varargs.insert(reader_varargs.end(), slice_end.cbegin(), slice_end.cend());
            reader_varargs.insert(reader_varargs.end(), slice_step.cbegin(), slice_step.cend());
            reader_run.advanced_options.runtime_varargs[node] = std::move(reader_varargs);

            writer_run.runtime_arg_values.push_back(
                {node,
                 {{"tensor_rank", tensor_rank},
                  {"element_size", element_size},
                  {"num_rows_for_this_core", rows_for_this_core},
                  {"start_row_for_this_core", row_start_id}}});
            m2::AdvancedKernelRunArgs::Varargs writer_varargs(output_shape.cbegin(), output_shape.cend());
            writer_run.advanced_options.runtime_varargs[node] = std::move(writer_varargs);
        }

        row_start_id += rows_for_this_core;
    }

    run.kernel_run_args = {reader_run, writer_run};
    run.tensor_args = {
        {m2::TensorParamName{"src"}, input_tensor.mesh_tensor()},
        {m2::TensorParamName{"dst"}, output.mesh_tensor()},
    };

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run)};
}

}  // namespace ttnn::prim
