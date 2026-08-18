// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <ctime>
#include <limits>
#include <random>
#include <string>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "uniform_device_operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::uniform {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {
std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution<int32_t> distribution(1, std::numeric_limits<int32_t>::max());

uint32_t get_random_seed() { return distribution(rng); }

// Work split used by create_program_artifacts (cache miss) and override_runtime_arguments (cache
// hit) so both derive the identical core list.
struct UniformWorkSplit {
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
};

UniformWorkSplit uniform_work_split(Tensor& output) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);
    return {
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
        std::move(cores)};
}

// Per-core work assignment, single-sourced so create_program_artifacts and
// override_runtime_arguments can never drift on core-group selection or tile_offset accumulation.
struct UniformCoreWork {
    CoreCoord core;
    uint32_t units_per_core;
    uint32_t tile_offset;
};

std::vector<UniformCoreWork> uniform_core_layout(const UniformWorkSplit& ws) {
    std::vector<UniformCoreWork> layout;
    layout.reserve(ws.cores.size());
    uint32_t tile_offset = 0;
    for (const auto& core : ws.cores) {
        uint32_t units_per_core;
        if (ws.core_group_1.contains(core)) {
            units_per_core = ws.units_per_core_group_1;
        } else if (ws.core_group_2.contains(core)) {
            units_per_core = ws.units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        layout.push_back({core, units_per_core, tile_offset});
        tile_offset += units_per_core;
    }
    return layout;
}

// Per-core seed; shared so the miss-build and the hit-patch derive it identically.
uint32_t uniform_seed_for_core(const UniformDeviceOperation::operation_attributes_t& attrs, int i) {
    return attrs.seed != 0 ? attrs.seed + i : get_random_seed();
}

// [from, to) as the bit patterns the compute kernel expects; shared so eps cannot drift between the
// miss-build and the hit-patch.
struct UniformRange {
    uint32_t f2u_from;
    uint32_t f2u_to;
};

UniformRange uniform_range(const UniformDeviceOperation::operation_attributes_t& attrs) {
    constexpr float eps = 1e-6f;
    // -eps make sure that generated number is < attrs.to
    return {std::bit_cast<uint32_t>(attrs.from), std::bit_cast<uint32_t>(attrs.to - eps)};
}

const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};
const DFBSpecName INTERMED{"intermed"};
const DFBSpecName DST{"dst"};
// The op is in-place, so its single tensor is both the input and the output. Named for the
// kernels' own vocabulary (`dst_addr`), since the forked kernels' binding names outlive this op.
const TensorParamName OUTPUT{"dst"};

// Per-core runtime args, keyed by name. Shared by the cache-miss build and the cache-hit patch so
// the two can never disagree on which kernel gets which value.
void add_uniform_run_args(
    KernelRunArgs& writer_run_args,
    KernelRunArgs& compute_run_args,
    const UniformDeviceOperation::operation_attributes_t& attrs,
    const std::vector<UniformCoreWork>& layout) {
    const auto [f2u_from, f2u_to] = uniform_range(attrs);

    for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
        const auto& [core, units_per_core, tile_offset] = layout[i];

        // Each core has its own seed to increase the number of generated random numbers
        const uint32_t seed = uniform_seed_for_core(attrs, i);

        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values,
            core,
            {{"seed", seed},
             {"f2u_from", f2u_from},
             {"f2u_to", f2u_to},
             {"start_id", tile_offset},
             {"num_tiles", units_per_core}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"start_id", tile_offset}, {"num_tiles", units_per_core}});
    }
}
}  // namespace

static constexpr const char* WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/uniform/device/kernels/writer_uniform_metal2.cpp";
static constexpr const char* COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/uniform/device/kernels/compute_uniform_metal2.cpp";

ttnn::device_operation::ProgramArtifacts UniformDeviceOperation::UniformProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    IDevice* device = output.device();
    const auto& output_mesh_tensor = output.mesh_tensor();
    const auto ws = uniform_work_split(output);
    const auto& all_cores = ws.all_cores;

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);
    const uint32_t intermed_tile_size = tile_size(tt::DataFormat::Float32);

    constexpr uint32_t in_out_num_tiles = 1;
    constexpr uint32_t intermed_num_tiles = 2;

    // The op resolves a TTNN compute config but does not pass fp32_dest_acc_en through: it forces
    // the knob on regardless of what the caller asked for. Translate the resolved config, then
    // re-apply that override, or the translation would hand the caller's value back.
    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);
    // if enable_32_bit_dest set to false a precision error may occur which makes
    // generated number out of range [from, to)
    std::get<ComputeGen1Config>(compute_hw).enable_32_bit_dest = true;

    // Intermediate DFB (Float32): compute packs into it, the writer drains it.
    DataflowBufferSpec intermed_dfb{
        .unique_id = INTERMED,
        .entry_size = intermed_tile_size,
        .num_entries = intermed_num_tiles,
        .data_format_metadata = tt::DataFormat::Float32,
    };

    // Output DFB: the writer is its only toucher (it stages the bfloat16 conversion here and NOC-
    // writes straight out of it), so the writer binds both endpoints — a self-loop.
    DataflowBufferSpec dst_dfb{
        .unique_id = DST,
        .entry_size = dtype_tile_size,
        .num_entries = in_out_num_tiles,
        .data_format_metadata = out_data_format,
    };

    // Writer kernel
    KernelSpec::CompilerOptions::Defines writer_defines;
    switch (output_dtype) {
        case DataType::BFLOAT16: writer_defines.emplace("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        case DataType::FLOAT32: writer_defines.emplace("OUTPUT_DTYPE_FLOAT32", "1"); break;
        default: break;
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL_PATH,
        .compiler_options = {.defines = std::move(writer_defines)},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INTERMED,
                    .accessor_name = "intermed",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = DST,
                    .accessor_name = "dst",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
                DFBBinding{
                    .dfb_spec_name = DST,
                    .accessor_name = "dst",
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
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Compute kernel
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = COMPUTE_KERNEL_PATH,
        // Legacy ComputeConfig defaults opt_level to O3; the type-agnostic CompilerOptions
        // defaults to O2, so the level has to be stated to keep the compile identical.
        .compiler_options = {.opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = INTERMED,
                    .accessor_name = "intermed",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"seed", "f2u_from", "f2u_to", "start_id", "num_tiles"}},
        .hw_config = std::move(compute_hw),
    };

    ProgramSpec spec{
        .name = "uniform",
        .kernels = {std::move(writer), std::move(compute)},
        .dataflow_buffers = {std::move(intermed_dfb), std::move(dst_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "main",
                    .kernels = {WRITER, COMPUTE},
                    .target_nodes = all_cores,
                },
            },
    };

    // seed/from/to are DYNAMIC (excluded from the program hash): set here for the cache-miss
    // build, re-applied on every cache hit by override_runtime_arguments().
    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    add_uniform_run_args(writer_run_args, compute_run_args, operation_attributes, uniform_core_layout(ws));

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {{OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ProgramRunArgs UniformDeviceOperation::UniformProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Re-supply the cached program's per-dispatch state: the compute kernel's seed/from/to
    // (hash-excluded) and the output tensor binding. On this concept the framework refreshes
    // nothing on its own, so the binding has to come from here or it stays frozen at the
    // cache-miss address. tile_offset/units_per_core come from the same shared work-split
    // helpers create_program_artifacts uses, so the values cannot drift.
    const auto ws = uniform_work_split(output);

    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    add_uniform_run_args(writer_run_args, compute_run_args, operation_attributes, uniform_core_layout(ws));

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {{OUTPUT, output.mesh_tensor()}};

    return run_args;
}

}  // namespace ttnn::operations::uniform
