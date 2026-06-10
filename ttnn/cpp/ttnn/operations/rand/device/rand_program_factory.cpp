// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <bit>
#include <ctime>
#include <filesystem>
#include <limits>
#include <random>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/metal2_artifacts.hpp"
#include "rand_device_operation.hpp"

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed() -> uint32_t { return distribution(rng); }

constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/writer_uniform.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/compute_uniform.cpp";

// Work split over the output tiles. Derived only from the output tensor + the device grid (the
// immutable program structure), so create_program_spec (which bakes the per-core start_id/num_tiles as
// enqueue-invariant) and create_per_enqueue_run_args (which walks the same core list to assign seeds)
// produce a consistent core enumeration.
struct RandWorkSplit {
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

RandWorkSplit compute_rand_work_split(RandDeviceOperation::tensor_return_value_t& output) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
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

// Per-core seed. seed == 0 means "non-deterministic": draw a fresh host seed each call (so repeated
// calls differ). seed != 0 means "deterministic": seed + core_index, so the same seed reproduces the
// same output (and a different seed changes it). The legacy per-DEVICE seed offset for sharded meshes
// is not expressible on the single-program ProgramSpec path (one spec stamped across the mesh) — that
// case needs a multi-program workload concept; see create_per_enqueue_run_args.
uint32_t rand_seed_for_core(const RandDeviceOperation::operation_attributes_t& attrs, int i) {
    return attrs.seed != 0 ? attrs.seed + static_cast<uint32_t>(i) : get_random_seed();
}

}  // namespace

RandDeviceOperation::program_factory_t RandDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return RandProgramFactory{};
}

// ===================================================================================
// create_program_spec — the immutable spec + the ENQUEUE-INVARIANT run-args (per-core work split).
// Runs on cache miss. start_id / num_tiles are declared enqueue-invariant so the hit path may omit
// them (they are retained from this miss-time SetProgramRunArgs).
// ===================================================================================
ttnn::device_operation::ProgramArtifacts RandDeviceOperation::RandProgramFactory::create_program_spec(
    const operation_attributes_t& /*attributes*/, const tensor_args_t& /*tensor_args*/, tensor_return_value_t& output) {
    const auto ws = compute_rand_work_split(output);
    const DataType output_dtype = output.dtype();
    const auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    constexpr auto fp32_format = tt::DataFormat::Float32;

    // Two DFBs: an fp32 intermediate produced by compute, and an output-dtype scratch the writer
    // narrows into (bound both producer and consumer on the writer so it has both endpoints).
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
    // The output tensor is per-enqueue (its address is refreshed each dispatch), so it is NOT declared
    // enqueue-invariant.
    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "rand_work",
        .kernels = {m2::KernelSpecName{"compute"}, m2::KernelSpecName{"writer"}},
        .target_nodes = ws.all_cores}};

    // Enqueue-invariant run-args: the per-core work split, identical for every dispatch sharing this
    // cache entry. Applied once on miss and retained thereafter.
    m2::ProgramRunArgs::KernelRunArgs compute_inv{.kernel = m2::KernelSpecName{"compute"}};
    m2::ProgramRunArgs::KernelRunArgs writer_inv{.kernel = m2::KernelSpecName{"writer"}};
    uint32_t tile_offset = 0;
    for (const auto& core : ws.cores) {
        const uint32_t units = units_for_core(ws, core);
        compute_inv.runtime_arg_values.push_back({core, {{"start_id", tile_offset}, {"num_tiles", units}}});
        writer_inv.runtime_arg_values.push_back({core, {{"start_id", tile_offset}, {"num_tiles", units}}});
        tile_offset += units;
    }
    m2::ProgramRunArgs invariant_run_args;
    invariant_run_args.kernel_run_args.push_back(std::move(compute_inv));
    invariant_run_args.kernel_run_args.push_back(std::move(writer_inv));

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = std::move(invariant_run_args)};
}

// ===================================================================================
// create_per_enqueue_run_args — the PER-ENQUEUE values: per-core seed + from/to, and the output
// tensor binding. Merged with the invariant set on cache miss; re-applied alone (UpdateProgramRunArgs)
// on every cache hit, so a new seed yields new output while the cached program is reused.
// ===================================================================================
m2::ProgramRunArgs RandDeviceOperation::RandProgramFactory::create_per_enqueue_run_args(
    const operation_attributes_t& attributes, const tensor_args_t& /*tensor_args*/, tensor_return_value_t& output) {
    const auto ws = compute_rand_work_split(output);
    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(attributes.to - eps);

    m2::ProgramRunArgs::KernelRunArgs compute_dyn{.kernel = m2::KernelSpecName{"compute"}};
    for (int i = 0; i < static_cast<int>(ws.cores.size()); ++i) {
        const uint32_t seed = rand_seed_for_core(attributes, i);
        compute_dyn.runtime_arg_values.push_back({ws.cores[i], {{"seed", seed}, {"from", from_bits}, {"to", to_bits}}});
    }
    // (The writer kernel's only per-enqueue input is the output tensor binding below; its start_id /
    // num_tiles are enqueue-invariant, so it needs no per-enqueue KernelRunArgs entry.)

    m2::ProgramRunArgs args;
    args.kernel_run_args.push_back(std::move(compute_dyn));
    args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});
    return args;
}

}  // namespace ttnn::operations::rand
