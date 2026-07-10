// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <bit>
#include <ctime>
#include <limits>
#include <random>
#include <vector>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>

#include "ttnn/tensor/types.hpp"
#include "rand_program_factory.hpp"

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/writer_rand.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/compute_rand.cpp";

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());
uint32_t get_random_seed() { return distribution(rng); }

// Work split shared by create_program_artifacts (spec build) and create_program_run_args (hit-patch)
// so both derive the identical core list. NOTE: device_seed_offset is 0 here — the M2 adapter stamps
// one program's run args across the whole mesh, so per-DEVICE unique seeding (sharded mesh) is not yet
// supported on this path (needs per-coord run args). Single-device / replicated seeding is correct.
struct RandWorkSplit {
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

RandWorkSplit compute_rand_work_split(const Tensor& output) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    return {all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2, std::move(cores)};
}

// Per-core seed; shared so miss-build and hit-patch produce identical values.
uint32_t rand_seed_for_core(const RandOperationAttributes& attrs, int i) {
    return attrs.seed != 0 ? attrs.seed + static_cast<uint32_t>(i) : get_random_seed();
}

// Single source of truth for the per-enqueue ProgramRunArgs. Built on cache miss (initial values) AND
// re-derived on every cache hit (fresh seed/from/to) — so seed/from/to are re-applied, never frozen.
ProgramRunArgs build_run_args(const RandOperationAttributes& attrs, const Tensor& output) {
    const KernelSpecName COMPUTE_KERNEL{"compute"};
    const KernelSpecName WRITER_KERNEL{"writer"};
    const TensorParamName OUTPUT{"output"};

    auto ws = compute_rand_work_split(output);

    constexpr float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(attrs.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(attrs.to - eps);

    KernelRunArgs compute_run{.kernel = COMPUTE_KERNEL};
    KernelRunArgs writer_run{.kernel = WRITER_KERNEL};
    compute_run.runtime_arg_values.reserve(ws.cores.size());
    writer_run.runtime_arg_values.reserve(ws.cores.size());

    uint32_t tile_offset = 0;
    for (int i = 0; i < static_cast<int>(ws.cores.size()); ++i) {
        const auto& core = ws.cores[i];
        uint32_t units_per_core = 0;
        if (ws.core_group_1.contains(core)) {
            units_per_core = ws.units_per_core_group_1;
        } else if (ws.core_group_2.contains(core)) {
            units_per_core = ws.units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        const uint32_t seed = rand_seed_for_core(attrs, i);
        const NodeCoord node = core;
        compute_run.runtime_arg_values.push_back(
            {node,
             {{"seed", seed},
              {"from_bits", from_bits},
              {"to_bits", to_bits},
              {"start_id", tile_offset},
              {"num_tiles", units_per_core}}});
        writer_run.runtime_arg_values.push_back({node, {{"start_id", tile_offset}, {"num_tiles", units_per_core}}});
        tile_offset += units_per_core;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(compute_run), std::move(writer_run)};
    run_args.tensor_args = {{OUTPUT, ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())}}};
    return run_args;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts RandProgramFactory::create_program_artifacts(
    const RandOperationAttributes& operation_attributes, const RandTensorArgs& /*tensor_args*/, Tensor& output) {
    const DFBSpecName CB_INTERMED{"cb_intermed"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName COMPUTE_KERNEL{"compute"};
    const KernelSpecName WRITER_KERNEL{"writer"};

    auto ws = compute_rand_work_split(output);

    const tt::DataFormat out_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t dtype_tile_size = tt::tile_size(out_data_format);
    const uint32_t intermed_tile_size = tt::tile_size(tt::DataFormat::Float32);
    constexpr uint32_t intermed_num_tiles = 2;

    // Single intermediate FIFO: compute (PRODUCER) generates random Float32 tiles, writer (CONSUMER)
    // streams them to the output tensor. rand always produces Float32 (rand.cpp typecasts afterward).
    std::vector<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = CB_INTERMED,
        .entry_size = intermed_tile_size,
        .num_entries = intermed_num_tiles,
        .data_format_metadata = tt::DataFormat::Float32,
    });

    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    KernelSpec compute_spec{
        .unique_id = COMPUTE_KERNEL,
        .source = std::filesystem::path{COMPUTE_KERNEL_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_INTERMED,
            .accessor_name = "cb_intermed",
            .endpoint_type = DFBBinding::EndpointType::PRODUCER}},
        // seed/from_bits/to_bits are per-enqueue (excluded from the key, re-applied every dispatch).
        .runtime_arg_schema = {.runtime_arg_names = {"seed", "from_bits", "to_bits", "start_id", "num_tiles"}},
        .hw_config =
            ComputeHardwareConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = true,  // false can push generated values out of [from, to)
                .dst_full_sync_en = false,
                .math_approx_mode = true},
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source = std::filesystem::path{WRITER_KERNEL_PATH},
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = CB_INTERMED,
            .accessor_name = "cb_intermed",
            .endpoint_type = DFBBinding::EndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args = {{"page_size", dtype_tile_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = DataMovementHardwareConfig{.role = DataMovementRoleHint::WRITER},
    };

    WorkUnitSpec wu{
        .name = "rand",
        .kernels = {COMPUTE_KERNEL, WRITER_KERNEL},
        .target_nodes = ws.all_cores,
    };

    ProgramSpec spec{
        .name = "rand",
        .kernels = {std::move(compute_spec), std::move(writer_spec)},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = {output_param},
        .work_units = {wu},
    };

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec), .run_params = build_run_args(operation_attributes, output)};
}

ProgramRunArgs RandProgramFactory::create_program_run_args(
    const RandOperationAttributes& operation_attributes, const RandTensorArgs& /*tensor_args*/, Tensor& output) {
    // Re-derive the per-enqueue run args (fresh seed/from/to) from the live attributes; the framework
    // re-applies them via SetProgramRunArgs on every cache hit, so a value excluded from the key is
    // re-applied instead of frozen. Shares build_run_args with create_program_artifacts.
    return build_run_args(operation_attributes, output);
}

}  // namespace ttnn::operations::rand
