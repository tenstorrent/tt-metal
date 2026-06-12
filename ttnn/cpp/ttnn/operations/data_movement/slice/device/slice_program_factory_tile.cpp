// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"

#include <filesystem>
#include <optional>
#include <vector>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace ttnn::prim {

namespace {

constexpr const char* TILE_READER_KERNEL =
    "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/reader_slice_tile_m2.cpp";
constexpr const char* TILE_WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/writer_slice_tile_m2.cpp";

// Enqueue-invariant work geometry: a pure function of shape + slice params (all in the cache key).
struct SliceTileGeometry {
    CoreRangeSet all_cores;
    std::vector<CoreCoord> cores;
    uint32_t num_dims = 0;
    uint32_t single_tile_size = 0;
    tt::DataFormat data_format = tt::DataFormat::Invalid;

    // Common, broadcast to every core (read-only on device): [num_unpadded..., num_padded...].
    std::vector<uint32_t> num_unpadded_per_dim;
    std::vector<uint32_t> num_padded_per_dim;

    // Per-core values (parallel to `cores`).
    std::vector<uint32_t> reader_start_id;
    std::vector<uint32_t> reader_num_tiles;
    std::vector<std::vector<uint32_t>> id_per_dim;  // [core][num_dims], the running-index seed
    std::vector<uint32_t> writer_start_id;          // == cumulative num_tiles_written
};

SliceTileGeometry tile_compute_geometry(const SliceParams& args, const SliceInputs& tensor_args, const Tensor& output) {
    const auto& input = tensor_args.input;
    IDevice* device = input.device();

    SliceTileGeometry g;
    g.data_format = datatype_to_dataformat_converter(input.dtype());
    g.single_tile_size = tt::tile_size(g.data_format);

    uint32_t num_unpadded_tiles = output.physical_volume() / TILE_HW;
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        args.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(args.sub_core_grids.value(), num_unpadded_tiles)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_tiles);
    g.all_cores = all_cores;

    const auto& input_shape = input.padded_shape();
    const auto& output_shape = output.padded_shape();
    g.num_dims = static_cast<uint32_t>(input_shape.rank());

    uint32_t num_unpadded_Xt = output_shape[-1] / TILE_WIDTH;
    uint32_t num_total_Xt = input_shape[-1] / TILE_WIDTH;
    uint32_t num_padded_Xt = num_total_Xt - num_unpadded_Xt;
    uint32_t num_unpadded_Yt = output_shape[-2] / TILE_HEIGHT;
    uint32_t num_total_Yt = input_shape[-2] / TILE_HEIGHT;
    uint32_t num_padded_Yt = (num_total_Yt - num_unpadded_Yt) * num_total_Xt;

    std::vector<uint32_t> accumulated_total_per_dim(g.num_dims);
    accumulated_total_per_dim[0] = num_total_Xt;
    accumulated_total_per_dim[1] = num_total_Yt * num_total_Xt;

    g.num_unpadded_per_dim.resize(g.num_dims);
    g.num_padded_per_dim.resize(g.num_dims);
    g.num_unpadded_per_dim[0] = num_unpadded_Xt;
    g.num_unpadded_per_dim[1] = num_unpadded_Yt;
    g.num_padded_per_dim[0] = num_padded_Xt;
    g.num_padded_per_dim[1] = num_padded_Yt;
    for (int32_t i = 2; i < static_cast<int32_t>(g.num_dims); ++i) {
        uint32_t num_unpadded_dim = output_shape[-(i + 1)];
        uint32_t num_total_dim = input_shape[-(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        g.num_unpadded_per_dim[i] = num_unpadded_dim;
        g.num_padded_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t start_offset = ttnn::operations::data_movement::get_tiled_start_offset(input, args.slice_start);

    g.cores = corerange_to_cores(all_cores);
    uint32_t num_tiles_written = 0;
    for (const auto& core : g.cores) {
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            // no-op core
            g.reader_start_id.push_back(0);
            g.reader_num_tiles.push_back(0);
            g.id_per_dim.emplace_back(g.num_dims, 0);
            g.writer_start_id.push_back(0);
            continue;
        }

        std::vector<uint32_t> ids(g.num_dims);
        ids[0] = num_tiles_written % g.num_unpadded_per_dim[0];
        uint32_t unpadded_written = num_tiles_written / g.num_unpadded_per_dim[0];
        uint32_t start_id = ids[0] + start_offset;
        for (uint32_t j = 1; j < g.num_dims; ++j) {
            ids[j] = unpadded_written % g.num_unpadded_per_dim[j];
            unpadded_written = unpadded_written / g.num_unpadded_per_dim[j];
            start_id += ids[j] * accumulated_total_per_dim[j - 1];
        }
        g.reader_start_id.push_back(start_id);
        g.reader_num_tiles.push_back(num_tiles_per_core);
        g.id_per_dim.push_back(std::move(ids));
        g.writer_start_id.push_back(num_tiles_written);
        num_tiles_written += num_tiles_per_core;
    }
    return g;
}

m2::NodeCoord tile_node_of(const CoreCoord& c) {
    return m2::NodeCoord{static_cast<std::size_t>(c.x), static_cast<std::size_t>(c.y)};
}

}  // namespace

m2::ProgramSpec SliceTileProgramFactory::create_program_spec(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto g = tile_compute_geometry(args, tensor_args, output);

    m2::DataflowBufferSpec cb{
        .unique_id = m2::DFBSpecName{"cb"},
        .entry_size = g.single_tile_size,
        .num_entries = 2,
        .data_format_metadata = g.data_format,
    };

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{TILE_READER_KERNEL},
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"cb"}, "cb_in")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"src"}, .accessor_name = "src"}},
        .compile_time_args = {{"num_dims", g.num_dims}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::READER},
        .advanced_options =
            m2::KernelAdvancedOptions{
                .num_runtime_varargs = g.num_dims, .num_common_runtime_varargs = 2 * g.num_dims},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{TILE_WRITER_KERNEL},
        .dfb_bindings = {m2::ConsumerOf(m2::DFBSpecName{"cb"}, "cb_out")},
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = m2::TensorParamName{"dst"}, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.role = m2::DataMovementHardwareConfig::RoleHint::WRITER},
    };

    m2::ProgramSpec spec;
    spec.name = "slice_tile";
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

m2::ProgramRunArgs SliceTileProgramFactory::create_invariant_run_args(
    const SliceParams& args, const SliceInputs& tensor_args, Tensor& output) {
    const auto g = tile_compute_geometry(args, tensor_args, output);

    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    // Common varargs (broadcast): [num_unpadded..., num_padded...].
    auto& common = reader_args.advanced_options.common_runtime_varargs;
    common.insert(common.end(), g.num_unpadded_per_dim.begin(), g.num_unpadded_per_dim.end());
    common.insert(common.end(), g.num_padded_per_dim.begin(), g.num_padded_per_dim.end());

    for (size_t i = 0; i < g.cores.size(); ++i) {
        const auto node = tile_node_of(g.cores[i]);
        reader_args.runtime_arg_values.push_back(
            {node, {{"start_id", g.reader_start_id[i]}, {"num_tiles", g.reader_num_tiles[i]}}});
        reader_args.advanced_options.runtime_varargs.emplace(
            node, m2::AdvancedKernelRunArgs::Varargs(g.id_per_dim[i].begin(), g.id_per_dim[i].end()));
        writer_args.runtime_arg_values.push_back(
            {node, {{"num_pages", g.reader_num_tiles[i]}, {"start_id", g.writer_start_id[i]}}});
    }

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));
    return run_args;
}

m2::ProgramRunArgs SliceTileProgramFactory::create_per_enqueue_args(
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
