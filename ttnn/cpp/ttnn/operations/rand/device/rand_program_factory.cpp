// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <bit>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <limits>
#include <random>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/tensor/types.hpp"
#include "rand_device_operation.hpp"

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;
using immutable_info_t = RandDeviceOperation::RandProgramFactory::immutable_info_t;

namespace {

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed() -> uint32_t { return distribution(rng); }

constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/writer_uniform.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/compute_uniform.cpp";

// Work split over the output tiles — a pure function of (grid, tile count). create_program_spec
// derives it from the ImmutableInfo; create_per_enqueue_args re-derives the identical split from the
// output tensor, so the per-core seed assignment lines up with the work-split start_id / num_tiles for
// the same cores.
struct RandWorkSplit {
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

RandWorkSplit compute_work_split(const CoreCoord& grid, uint32_t units_to_divide) {
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    return {all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2, std::move(cores)};
}

uint32_t units_for_core(const RandWorkSplit& ws, const CoreCoord& core) {
    if (ws.core_group_1.contains(core)) {
        return ws.units_per_core_group_1;
    }
    if (ws.core_group_2.contains(core)) {
        return ws.units_per_core_group_2;
    }
    TT_THROW("Core not in specified core ranges");
}

}  // namespace

// extract_immutable_info — the cache key + sole input to the spec/static builders. The structural
// projection of the request: output layout + compute grid. Excludes seed/from/to (those are dynamic).
RandDeviceOperation::RandProgramFactory::immutable_info_t
RandDeviceOperation::RandProgramFactory::extract_immutable_info(
    const operation_attributes_t& attrs, [[maybe_unused]] const tensor_args_t& tensor_args) {
    TensorSpec output_spec(attrs.shape, TensorLayout(attrs.dtype, PageConfig(attrs.layout), attrs.memory_config));
    return immutable_info_t{
        .output_spec = std::move(output_spec), .grid = attrs.device->compute_with_storage_grid_size()};
}

// create_program_spec — the immutable blueprint only (DFBs, kernels, work-units, schemas), derived from
// the ImmutableInfo. The per-core work split it references is supplied as enqueue-invariant run-args by
// create_invariant_run_args below.
m2::ProgramSpec RandDeviceOperation::RandProgramFactory::create_program_spec(const immutable_info_t& info) {
    const uint32_t units = info.output_spec.padded_shape().volume() / constants::TILE_HW;
    const auto ws = compute_work_split(info.grid, units);
    const DataType output_dtype = info.output_spec.data_type();
    const auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    constexpr auto fp32_format = tt::DataFormat::Float32;

    m2::DataflowBufferSpec intermed_dfb{
        .unique_id = m2::DFBSpecName{"rand_tiles"},
        .entry_size = tile_size(fp32_format),
        .num_entries = 2,
        .data_format_metadata = fp32_format,
    };
    m2::DataflowBufferSpec out_dfb{
        .unique_id = m2::DFBSpecName{"rand_out"},
        .entry_size = tile_size(out_data_format),
        .num_entries = 1,
        .data_format_metadata = out_data_format,
    };

    m2::KernelSpec compute_kernel{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(m2::DFBSpecName{"rand_tiles"}, "rand_tiles")},
        .runtime_arg_schema = {.runtime_arg_names = {"seed", "from", "to", "start_id", "num_tiles"}},
        .hw_config =
            m2::ComputeHardwareConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
                .math_approx_mode = true,
            },
        .advanced_options = m2::KernelAdvancedOptions{.enqueue_invariant_runtime_args = {"start_id", "num_tiles"}},
    };

    m2::KernelSpec::CompilerOptions writer_opts;
    switch (output_dtype) {
        case DataType::BFLOAT16: writer_opts.defines.emplace("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_opts.defines.emplace("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default: TT_THROW("RandDeviceOperation: unsupported output dtype for writer kernel");
    }
    m2::KernelSpec writer_kernel{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{WRITER_KERNEL_PATH},
        .compiler_options = std::move(writer_opts),
        .dfb_bindings =
            {m2::ConsumerOf(m2::DFBSpecName{"rand_tiles"}, "rand_tiles"),
             m2::ProducerOf(m2::DFBSpecName{"rand_out"}, "rand_out"),
             m2::ConsumerOf(m2::DFBSpecName{"rand_out"}, "rand_out")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
        .advanced_options = m2::KernelAdvancedOptions{.enqueue_invariant_runtime_args = {"start_id", "num_tiles"}},
    };

    m2::ProgramSpec spec;
    spec.name = "rand";
    spec.kernels = {std::move(compute_kernel), std::move(writer_kernel)};
    spec.dataflow_buffers = {std::move(intermed_dfb), std::move(out_dfb)};
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = info.output_spec}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "rand_work",
        .kernels = {m2::KernelSpecName{"compute"}, m2::KernelSpecName{"writer"}},
        .target_nodes = ws.all_cores}};

    return spec;
}

// create_invariant_run_args — the enqueue-invariant per-core work split (start_id / num_tiles), declared
// invariant in the spec so it is set once on a cache miss and retained across hits. Derives ONLY from the
// ImmutableInfo (the same projection as the spec), so it can never see the seed.
m2::ProgramRunArgs RandDeviceOperation::RandProgramFactory::create_invariant_run_args(const immutable_info_t& info) {
    const uint32_t units = info.output_spec.padded_shape().volume() / constants::TILE_HW;
    const auto ws = compute_work_split(info.grid, units);

    m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel = m2::KernelSpecName{"compute"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};
    uint32_t tile_offset = 0;
    for (const auto& core : ws.cores) {
        const uint32_t n = units_for_core(ws, core);
        compute_args.runtime_arg_values.push_back({core, {{"start_id", tile_offset}, {"num_tiles", n}}});
        writer_args.runtime_arg_values.push_back({core, {{"start_id", tile_offset}, {"num_tiles", n}}});
        tile_offset += n;
    }
    m2::ProgramRunArgs invariant_args;
    invariant_args.kernel_run_args.push_back(std::move(compute_args));
    invariant_args.kernel_run_args.push_back(std::move(writer_args));
    return invariant_args;
}

// create_per_enqueue_args — the per-enqueue values: per-core seed + from/to range, and the output tensor.
// Re-applied on every dispatch via UpdateProgramRunArgs. The per-coordinate seed offset gives distinct
// streams across a sharded mesh.
m2::ProgramRunArgs RandDeviceOperation::RandProgramFactory::create_per_enqueue_args(
    const operation_attributes_t& attrs,
    [[maybe_unused]] const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    const uint32_t units = output.physical_volume() / constants::TILE_HW;
    const auto grid = output.device()->compute_with_storage_grid_size();
    const auto ws = compute_work_split(grid, units);

    // Per-device seed offset on a sharded mesh, so each device produces a distinct stream.
    uint32_t device_seed_offset = 0;
    const auto& shard_mask = attrs.mesh_dim_is_sharded;
    if (!shard_mask.empty()) {
        const ttnn::MeshCoordinate mesh_coordinate =
            mesh_dispatch_coordinate.value_or(ttnn::MeshCoordinate::zero_coordinate(attrs.device->shape().dims()));
        const auto& mesh_shape = attrs.device->shape();
        size_t shard_linear_idx = 0;
        size_t shard_stride = 1;
        for (int i = static_cast<int>(shard_mask.size()) - 1; i >= 0; --i) {
            if (shard_mask[i]) {
                shard_linear_idx += mesh_coordinate[i] * shard_stride;
                shard_stride *= mesh_shape[i];
            }
        }
        device_seed_offset = static_cast<uint32_t>(shard_linear_idx) * static_cast<uint32_t>(ws.cores.size());
    }

    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(attrs.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(attrs.to - eps);

    m2::ProgramRunArgs::KernelRunArgs compute_args{.kernel = m2::KernelSpecName{"compute"}};
    for (int i = 0; i < static_cast<int>(ws.cores.size()); ++i) {
        const uint32_t seed =
            attrs.seed != 0 ? attrs.seed + static_cast<uint32_t>(i) + device_seed_offset : get_random_seed();
        compute_args.runtime_arg_values.push_back(
            {ws.cores[i], {{"seed", seed}, {"from", from_bits}, {"to", to_bits}}});
    }

    m2::ProgramRunArgs args;
    args.kernel_run_args.push_back(std::move(compute_args));
    args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});
    return args;
}

}  // namespace ttnn::operations::rand
