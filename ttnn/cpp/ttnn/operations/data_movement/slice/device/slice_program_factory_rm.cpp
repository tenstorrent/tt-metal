// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

#include <algorithm>
#include <filesystem>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

constexpr const char* RM_READER_KERNEL =
    "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
    "slice_reader_unary_unpad_dims_rm_interleaved_start_id_m2.cpp";
constexpr const char* RM_WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
    "slice_writer_unary_stick_layout_interleaved_start_id_m2.cpp";

constexpr uint32_t MAX_READ_SIZE = 4096;

// Enqueue-invariant work geometry: a pure function of shape + slice params (all in the cache key).
struct SliceRmGeometry {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t num_dims = 0;
    tt::DataFormat data_format = tt::DataFormat::Invalid;

    // Op-level byte sizes (identical across cores -> named CTAs).
    uint32_t padded_stick_size = 0;          // input row size in bytes (== src accessor page size)
    uint32_t unpadded_stick_size = 0;        // output row size in bytes (== dst accessor page size)
    uint32_t stick_size_offset = 0;          // aligned CB stride between consecutive sticks
    uint32_t misalignment = 0;               // slice column begin misalignment vs src alignment
    uint32_t column_offset_bytes = 0;        // begins_bytes - misalignment (source page offset_bytes)

    // CB sizing (program-scope, sized for the max-work core group).
    uint32_t cb_entry_size = 0;              // == cb_page_size
    uint32_t cb_num_entries = 0;             // == num_read_per_barrier_max * 2

    // Common, broadcast to every core (read-only on device): [num_unpadded..., num_padded...].
    std::vector<uint32_t> num_unpadded_per_dim;
    std::vector<uint32_t> num_padded_per_dim;

    // Per-core values (parallel to `cores`).
    std::vector<uint32_t> start_id;
    std::vector<uint32_t> num_sticks_per_core;
    std::vector<uint32_t> num_sticks_per_core_read;
    std::vector<uint32_t> num_read_per_barrier;
    std::vector<std::vector<uint32_t>> id_per_dim;  // [core][num_dims], the running-index seed
    std::vector<uint32_t> writer_start_id;          // == cumulative num_sticks_written
};

SliceRmGeometry rm_compute_geometry(const SliceParams& args, const SliceInputs& tensor_args, const Tensor& output) {
    const auto& input = tensor_args.input;
    IDevice* device = input.device();

    SliceRmGeometry g;
    g.data_format = datatype_to_dataformat_converter(input.dtype());

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    g.num_dims = static_cast<uint32_t>(input_shape.rank());

    uint32_t num_unpadded_sticks = output.physical_volume() / output_shape[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);
    g.all_cores = all_cores;

    g.padded_stick_size = input_shape[-1] * input.element_size();
    g.unpadded_stick_size = output_shape[-1] * input.element_size();

    g.num_unpadded_per_dim.resize(g.num_dims);
    g.num_padded_per_dim.resize(g.num_dims);
    std::vector<uint32_t> accumulated_total_per_dim(g.num_dims);

    // TODO: Remove first element of these arrays and update kernel accordingly
    g.num_unpadded_per_dim[0] = 1;
    g.num_padded_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;

    for (uint32_t i = 1; i < g.num_dims; ++i) {
        uint32_t num_unpadded_dim = output_shape[-(static_cast<int32_t>(i) + 1)];
        uint32_t num_total_dim = input_shape[-(static_cast<int32_t>(i) + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        g.num_unpadded_per_dim[i] = num_unpadded_dim;
        g.num_padded_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    auto src_buffer_alignment = input.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    const auto single_alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    uint32_t begins_bytes = args.slice_start[-1] * input.element_size();
    g.misalignment = begins_bytes % src_buffer_alignment;
    // Source page offset_bytes (read from nearest aligned address): begins_bytes - misalignment.
    g.column_offset_bytes = begins_bytes - g.misalignment;

    g.stick_size_offset = tt::round_up(g.unpadded_stick_size, single_alignment);

    // CB page size: doubled alignment when misaligned (extra room for the memmove fixup).
    auto cb_alignment = single_alignment;
    if (g.misalignment != 0) {
        cb_alignment *= 2;
    }
    const uint32_t cb_page_size = tt::round_up(g.unpadded_stick_size, cb_alignment);

    // CB capacity is sized from the larger core group (max num_read_per_barrier).
    const uint32_t num_input_pages =
        std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2);
    uint32_t num_read_per_barrier_max = 0;
    if (num_input_pages != 0) {
        auto num_sticks_per_core_pad32 = num_input_pages + ((32 - num_input_pages % 32) % 32);
        uint32_t num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, g.stick_size_offset, MAX_READ_SIZE);
        num_read_per_barrier_max = num_sticks_per_core_pad32 / num_sticks_per_core_read;
    }
    g.cb_entry_size = cb_page_size;
    g.cb_num_entries = num_read_per_barrier_max * 2;

    uint32_t start_offset = ttnn::operations::data_movement::get_rm_start_offset(input, args.slice_start);

    g.cores = corerange_to_cores(all_cores);
    uint32_t num_sticks_written = 0;
    for (const auto& core : g.cores) {
        uint32_t num_sticks_per_core;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        } else {
            // no-op core
            g.start_id.push_back(0);
            g.num_sticks_per_core.push_back(0);
            g.num_sticks_per_core_read.push_back(0);
            g.num_read_per_barrier.push_back(0);
            g.id_per_dim.emplace_back(g.num_dims, 0);
            g.writer_start_id.push_back(0);
            continue;
        }

        uint32_t num_sticks_per_core_read = 0, num_read_per_barrier = 0;
        auto num_sticks_per_core_pad32 = num_sticks_per_core + ((32 - num_sticks_per_core % 32) % 32);
        num_sticks_per_core_read =
            tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, g.stick_size_offset, MAX_READ_SIZE);
        num_read_per_barrier = num_sticks_per_core_pad32 / num_sticks_per_core_read;

        std::vector<uint32_t> ids(g.num_dims);
        ids[0] = num_sticks_written % g.num_unpadded_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / g.num_unpadded_per_dim[0];
        uint32_t start_id = ids[0] + start_offset;
        for (uint32_t j = 1; j < g.num_dims; ++j) {
            ids[j] = unpadded_written % g.num_unpadded_per_dim[j];
            unpadded_written = unpadded_written / g.num_unpadded_per_dim[j];
            start_id += ids[j] * accumulated_total_per_dim[j - 1];
        }

        g.start_id.push_back(start_id);
        g.num_sticks_per_core.push_back(num_sticks_per_core);
        g.num_sticks_per_core_read.push_back(num_sticks_per_core_read);
        g.num_read_per_barrier.push_back(num_read_per_barrier);
        g.id_per_dim.push_back(std::move(ids));
        g.writer_start_id.push_back(num_sticks_written);
        num_sticks_written += num_sticks_per_core;
    }
    return g;
}

m2::NodeCoord rm_node_of(const CoreCoord& c) {
    return m2::NodeCoord{static_cast<std::size_t>(c.x), static_cast<std::size_t>(c.y)};
}

}  // namespace

m2::ProgramSpec SliceRmProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto g = rm_compute_geometry(args, tensor_args, output);

    m2::DataflowBufferSpec cb{
        .unique_id = m2::DFBSpecName{"cb"},
        .entry_size = g.cb_entry_size,
        .num_entries = g.cb_num_entries,
        .data_format_metadata = g.data_format,
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{RM_READER_KERNEL},
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"cb"}, "cb_in")},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"}},
        .compile_time_args =
            {{"num_dims", g.num_dims},
             {"unpadded_stick_size", g.unpadded_stick_size},
             {"stick_size_offset", g.stick_size_offset},
             {"misalignment", g.misalignment},
             {"column_offset_bytes", g.column_offset_bytes}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"start_id", "num_sticks_per_core", "num_sticks_per_core_read", "num_read_per_barrier"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::READER},
        .advanced_options =
            m2::KernelAdvancedOptions{
                .num_runtime_varargs = g.num_dims, .num_common_runtime_varargs = 2 * g.num_dims},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{RM_WRITER_KERNEL},
        .dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"cb"}, "cb_out")},
        .tensor_bindings =
            {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"}},
        .compile_time_args = {{"stick_size", g.unpadded_stick_size}, {"stick_size_offset", g.stick_size_offset}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_sticks_per_core", "num_sticks_per_core_read", "num_read_per_barrier", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::WRITER},
    };

    m2::ProgramSpec spec;
    spec.name = "slice_rm";
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

m2::ProgramRunArgs SliceRmProgramFactory::create_invariant_run_args(
    const SliceParams& /*args*/, const SliceInputs& /*tensor_args*/, Tensor& /*output*/) {
    // Nothing is enqueue-invariant here: the reader carries varargs (which cannot be designated
    // invariant — gap #6), so on a cache hit every kernel with varargs must appear in the
    // per-enqueue UpdateProgramRunArgs. We therefore re-apply all run args per dispatch (the
    // framework's "++ by default") and keep this empty.
    return m2::ProgramRunArgs{};
}

m2::ProgramRunArgs SliceRmProgramFactory::create_per_enqueue_args(
    const SliceParams& args,
    const SliceInputs& tensor_args,
    Tensor& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    const auto g = rm_compute_geometry(args, tensor_args, output);

    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    // Common varargs (broadcast): [num_unpadded..., num_padded...].
    auto& common = reader_args.advanced_options.common_runtime_varargs;
    common.insert(common.end(), g.num_unpadded_per_dim.begin(), g.num_unpadded_per_dim.end());
    common.insert(common.end(), g.num_padded_per_dim.begin(), g.num_padded_per_dim.end());

    for (size_t i = 0; i < g.cores.size(); ++i) {
        const auto node = rm_node_of(g.cores[i]);
        reader_args.runtime_arg_values.push_back(
            {node,
             {{"start_id", g.start_id[i]},
              {"num_sticks_per_core", g.num_sticks_per_core[i]},
              {"num_sticks_per_core_read", g.num_sticks_per_core_read[i]},
              {"num_read_per_barrier", g.num_read_per_barrier[i]}}});
        reader_args.advanced_options.runtime_varargs.emplace(
            node, m2::AdvancedKernelRunArgs::Varargs(g.id_per_dim[i].begin(), g.id_per_dim[i].end()));
        writer_args.runtime_arg_values.push_back(
            {node,
             {{"num_sticks_per_core", g.num_sticks_per_core[i]},
              {"num_sticks_per_core_read", g.num_sticks_per_core_read[i]},
              {"num_read_per_barrier", g.num_read_per_barrier[i]},
              {"start_id", g.writer_start_id[i]}}});
    }

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    run_args.tensor_args.emplace(
        m2::TensorParamName{"src"}, m2::ProgramRunArgs::TensorArgument{std::cref(tensor_args.input.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"dst"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});
    return run_args;
}

}  // namespace ttnn::prim
