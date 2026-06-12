// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

// Always the nd kernels (they handle all ranks).
constexpr const char* READER_KERNEL =
    "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_multicore_slice_nd_m2.cpp";
constexpr const char* WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_multicore_slice_nd_m2.cpp";

// Enqueue-invariant work geometry: a pure function of shape + slice params (all in the cache key).
struct SliceRmStrideGeometry {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t rank = 0;
    uint32_t element_size = 0;
    uint32_t cb_page_size_aligned = 0;
    tt::DataFormat data_format = tt::DataFormat::Invalid;

    // Op-level arrays, broadcast to every core (read-only on device).
    // Reader common varargs order: [input_dims, output_dims, slice_starts, slice_ends, slice_steps].
    std::vector<uint32_t> input_shape;
    std::vector<uint32_t> output_shape;
    std::vector<uint32_t> slice_start;
    std::vector<uint32_t> slice_end;
    std::vector<uint32_t> slice_step;

    // Per-core values (parallel to `cores`).
    std::vector<uint32_t> num_rows;   // rows_for_this_core
    std::vector<uint32_t> start_row;  // row_start_id
};

SliceRmStrideGeometry compute_geometry(
    const SliceParams& args, const SliceInputs& tensor_args, const Tensor& output) {
    const auto& input = tensor_args.input;
    IDevice* device = input.device();

    SliceRmStrideGeometry g;
    g.data_format = datatype_to_dataformat_converter(input.dtype());
    g.element_size = input.element_size();

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    g.rank = static_cast<uint32_t>(input_shape.rank());

    // Total output rows based on tensor rank.
    uint32_t total_output_rows = output_shape.volume() / output_shape[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), total_output_rows)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, total_output_rows);
    g.all_cores = all_cores;

    // CB sizing: one page = one full input row, aligned to max(src, dst) buffer alignment.
    uint32_t actual_input_w = input_shape[-1];
    uint32_t input_bytes_per_row = actual_input_w * g.element_size;
    uint32_t cb_page_size = input_bytes_per_row;

    auto src_buffer_alignment = input.buffer()->alignment();
    auto dst_buffer_alignment = output.buffer()->alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);
    g.cb_page_size_aligned = tt::round_up(cb_page_size, alignment);

    // Op-level arrays (identical across cores).
    g.input_shape.assign(input_shape.cbegin(), input_shape.cend());
    g.output_shape.assign(output_shape.cbegin(), output_shape.cend());
    g.slice_start.assign(args.slice_start.cbegin(), args.slice_start.cend());
    g.slice_end.assign(args.slice_end.cbegin(), args.slice_end.cend());
    g.slice_step.assign(args.step.cbegin(), args.step.cend());

    // Per-core row split (matches the legacy base/extra distribution over all cores).
    g.cores = corerange_to_cores(all_cores);
    uint32_t base_rows_per_core = total_output_rows / num_cores;
    uint32_t extra_rows = total_output_rows % num_cores;

    g.num_rows.reserve(g.cores.size());
    g.start_row.reserve(g.cores.size());

    uint32_t row_start_id = 0;
    uint32_t extra_rows_remaining = extra_rows;
    for (size_t i = 0; i < g.cores.size(); ++i) {
        uint32_t rows_for_this_core = base_rows_per_core;
        if (extra_rows_remaining > 0) {
            rows_for_this_core += 1;
            extra_rows_remaining -= 1;
        }
        g.num_rows.push_back(rows_for_this_core);
        g.start_row.push_back(row_start_id);
        row_start_id += rows_for_this_core;
    }
    return g;
}

m2::NodeCoord node_of(const CoreCoord& c) {
    return m2::NodeCoord{static_cast<std::size_t>(c.x), static_cast<std::size_t>(c.y)};
}

}  // namespace

m2::ProgramSpec SliceRmStrideProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto g = compute_geometry(args, tensor_args, output);

    m2::DataflowBufferSpec cb{
        .unique_id = m2::DFBSpecName{"cb"},
        .entry_size = g.cb_page_size_aligned,
        .num_entries = 2,
        .data_format_metadata = g.data_format,
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{READER_KERNEL},
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"cb"}, "cb_out")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"}},
        .compile_time_args = {{"rank", g.rank}, {"element_size", g.element_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "start_row"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::READER},
        .advanced_options =
            m2::KernelAdvancedOptions{.num_runtime_varargs = 0, .num_common_runtime_varargs = 5 * g.rank},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{WRITER_KERNEL},
        .dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"cb"}, "cb_in")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"}},
        .compile_time_args = {{"rank", g.rank}, {"element_size", g.element_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "start_row"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::WRITER},
        .advanced_options =
            m2::KernelAdvancedOptions{.num_runtime_varargs = 0, .num_common_runtime_varargs = g.rank},
    };

    m2::ProgramSpec spec;
    spec.name = "slice_rm_stride";
    spec.kernels = {std::move(reader), std::move(writer)};
    spec.dataflow_buffers = {std::move(cb)};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"src"}, .spec = tensor_args.input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"dst"}, .spec = output.tensor_spec()}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "slice",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}},
        .target_nodes = g.all_cores}};
    return spec;
}

m2::ProgramRunArgs SliceRmStrideProgramFactory::create_invariant_run_args(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto g = compute_geometry(args, tensor_args, output);

    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    // Reader common varargs: [input_dims, output_dims, slice_starts, slice_ends, slice_steps].
    auto& reader_common = reader_args.advanced_options.common_runtime_varargs;
    reader_common.insert(reader_common.end(), g.input_shape.begin(), g.input_shape.end());
    reader_common.insert(reader_common.end(), g.output_shape.begin(), g.output_shape.end());
    reader_common.insert(reader_common.end(), g.slice_start.begin(), g.slice_start.end());
    reader_common.insert(reader_common.end(), g.slice_end.begin(), g.slice_end.end());
    reader_common.insert(reader_common.end(), g.slice_step.begin(), g.slice_step.end());

    // Writer common varargs: [output_dims].
    auto& writer_common = writer_args.advanced_options.common_runtime_varargs;
    writer_common.insert(writer_common.end(), g.output_shape.begin(), g.output_shape.end());

    for (size_t i = 0; i < g.cores.size(); ++i) {
        const auto node = node_of(g.cores[i]);
        reader_args.runtime_arg_values.push_back(
            {node, {{"num_rows", g.num_rows[i]}, {"start_row", g.start_row[i]}}});
        writer_args.runtime_arg_values.push_back(
            {node, {{"num_rows", g.num_rows[i]}, {"start_row", g.start_row[i]}}});
    }

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    return run_args;
}

m2::ProgramRunArgs SliceRmStrideProgramFactory::create_per_enqueue_args(
    const SliceParams& /*args*/,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    m2::ProgramRunArgs run_args;
    run_args.tensor_args.emplace(
        m2::TensorParamName{"src"}, m2::ProgramRunArgs::TensorArgument{std::cref(tensor_args.input.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"dst"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});
    return run_args;
}

}  // namespace ttnn::prim
