// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <bit>
#include <ctime>
#include <limits>
#include <random>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/tensor/types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "rand_device_operation.hpp"

namespace ttnn::operations::rand {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace {

std::mt19937 rng(std::time(nullptr));
std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

auto get_random_seed() -> uint32_t { return distribution(rng); }

constexpr const char* WRITER_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/writer_rand_spec.cpp";
constexpr const char* COMPUTE_KERNEL_PATH = "ttnn/cpp/ttnn/operations/rand/device/kernels/compute_rand_spec.cpp";

// Spec resource names (prefixed to stay distinct under unity builds).
const KernelSpecName RAND_COMPUTE{"rand_compute"};
const KernelSpecName RAND_WRITER{"rand_writer"};
const DFBSpecName RAND_INTERMED{"rand_intermed"};
const DFBSpecName RAND_DST{"rand_dst"};
const TensorParamName RAND_OUTPUT{"rand_output"};

// Work split + per-device seed offset, shared by create_program_artifacts (cache miss) and
// override_runtime_arguments (cache hit) so both derive the identical core list and seed offset.
struct RandWorkSplit {
    uint32_t num_cores = 0;
    CoreRangeSet all_cores;
    CoreRangeSet core_group_1;
    CoreRangeSet core_group_2;
    uint32_t units_per_core_group_1 = 0;
    uint32_t units_per_core_group_2 = 0;
    std::vector<CoreCoord> cores;
    uint32_t device_seed_offset = 0;
};

RandWorkSplit compute_rand_work_split(
    const RandDeviceOperation::operation_attributes_t& attrs,
    RandDeviceOperation::tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    auto grid = output.device()->compute_with_storage_grid_size();
    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    auto cores = grid_to_cores(num_cores, grid.x, grid.y);

    const ttnn::MeshCoordinate mesh_coordinate =
        mesh_dispatch_coordinate.value_or(ttnn::MeshCoordinate::zero_coordinate(attrs.device->shape().dims()));
    uint32_t device_seed_offset = 0;
    const auto& shard_mask = attrs.mesh_dim_is_sharded;
    if (!shard_mask.empty()) {
        const auto& mesh_shape = attrs.device->shape();
        size_t shard_linear_idx = 0;
        size_t shard_stride = 1;
        for (int i = static_cast<int>(shard_mask.size()) - 1; i >= 0; --i) {
            if (shard_mask[i]) {
                shard_linear_idx += mesh_coordinate[i] * shard_stride;
                shard_stride *= mesh_shape[i];
            }
        }
        device_seed_offset = static_cast<uint32_t>(shard_linear_idx) * static_cast<uint32_t>(cores.size());
    }
    return {
        num_cores,
        all_cores,
        core_group_1,
        core_group_2,
        units_per_core_group_1,
        units_per_core_group_2,
        std::move(cores),
        device_seed_offset};
}

// Per-core seed; shared so the miss-build and the hit-patch produce identical values.
uint32_t rand_seed_for_core(
    const RandDeviceOperation::operation_attributes_t& attrs, int i, uint32_t device_seed_offset) {
    return attrs.seed != 0 ? attrs.seed + i + device_seed_offset : get_random_seed();
}

// Per-core work assignment (core, units_per_core, tile_offset). Single-sourced so the cache-miss
// build and the cache-hit override can never drift on core-group selection or tile_offset.
struct RandCoreWork {
    CoreCoord core;
    uint32_t units_per_core;
    uint32_t tile_offset;
};
std::vector<RandCoreWork> rand_core_layout(const RandWorkSplit& ws) {
    std::vector<RandCoreWork> layout;
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

}  // namespace

ttnn::device_operation::ProgramArtifacts RandDeviceOperation::RandProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output) {
    // create_program_artifacts is called once and the resulting spec is stamped across the mesh, so the
    // (per-device) seed offset is baked at the zero coordinate here; override_runtime_arguments re-applies
    // the correct per-coordinate seeds on every cache hit.
    const auto ws = compute_rand_work_split(operation_attributes, output, std::nullopt);

    const DataType output_dtype = output.dtype();
    const auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);
    const uint32_t intermed_tile_size = tile_size(tt::DataFormat::Float32);
    const bool is_bfloat16 = output_dtype == DataType::BFLOAT16;
    switch (output_dtype) {
        case DataType::BFLOAT16:
        case DataType::FLOAT32: break;
        default:
            // The writer kernel only implements float32 and bfloat16 output paths.
            TT_THROW("RandDeviceOperation: unsupported output dtype for writer kernel");
    }

    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

    // ---- ProgramSpec ----
    ProgramSpec spec;
    spec.name = "rand";

    spec.tensor_parameters = {TensorParameter{.unique_id = RAND_OUTPUT, .spec = output.tensor_spec()}};

    // Intermediate DFB: compute PRODUCES Float32 tiles, writer CONSUMES them (double-buffered).
    spec.dataflow_buffers.push_back(DataflowBufferSpec{
        .unique_id = RAND_INTERMED,
        .entry_size = intermed_tile_size,
        .num_entries = 2,
        .data_format_metadata = tt::DataFormat::Float32,
    });
    // bfloat16 output uses a writer-local staging DFB (self-loop) for the fp32->bf16 conversion.
    if (is_bfloat16) {
        spec.dataflow_buffers.push_back(DataflowBufferSpec{
            .unique_id = RAND_DST,
            .entry_size = dtype_tile_size,
            .num_entries = 1,
            .data_format_metadata = out_data_format,
        });
    }

    // Compute kernel: producer of the intermediate DFB.
    KernelSpec compute{
        .unique_id = RAND_COMPUTE,
        .source = COMPUTE_KERNEL_PATH,
        .dfb_bindings = {ProducerOf(RAND_INTERMED, "intermed")},
        .runtime_arg_schema = {.runtime_arg_names = {"seed", "from", "to", "tile_offset", "units_per_core"}},
        .hw_config = ttnn::to_compute_hardware_config(
            output.device()->arch(),
            ttnn::ComputeKernelConfig{
                .math_fidelity = MathFidelity::HiFi4,
                // if fp32_dest_acc_en is false a precision error may make numbers out of range [from, to)
                .math_approx_mode = true,
                .fp32_dest_acc_en = true,
                .dst_full_sync_en = false,
            }),
    };
    // tile_offset/units_per_core are static per core; only seed/from/to change per dispatch.
    compute.advanced_options.enqueue_invariant_runtime_args = {"tile_offset", "units_per_core"};

    // Writer kernel: consumer of the intermediate DFB; writes to the interleaved output.
    KernelSpec writer{
        .unique_id = RAND_WRITER,
        .source = WRITER_KERNEL_PATH,
        .dfb_bindings = {ConsumerOf(RAND_INTERMED, "intermed")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = RAND_OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"tile_offset", "units_per_core"}},
        .hw_config = ttnn::create_writer_datamovement_config(output.device()->arch()),
    };
    writer.advanced_options.enqueue_invariant_runtime_args = {"tile_offset", "units_per_core"};
    if (is_bfloat16) {
        writer.compiler_options.defines.emplace("OUTPUT_DTYPE_BFLOAT16", "1");
        // Writer-local self-loop staging DFB (one PRODUCER + one CONSUMER on the same kernel).
        writer.dfb_bindings.push_back(ProducerOf(RAND_DST, "dst"));
        writer.dfb_bindings.push_back(ConsumerOf(RAND_DST, "dst"));
    }

    spec.kernels.push_back(compute);
    spec.kernels.push_back(writer);

    Group<KernelSpecName> wu_kernels = {RAND_COMPUTE, RAND_WRITER};
    spec.work_units = {WorkUnitSpec{.name = "main", .kernels = wu_kernels, .target_nodes = ws.all_cores}};

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    KernelRunArgs compute_run{.kernel = RAND_COMPUTE};
    KernelRunArgs writer_run{.kernel = RAND_WRITER};

    const auto layout = rand_core_layout(ws);
    for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
        const auto& [core, units_per_core, tile_offset] = layout[i];
        const uint32_t seed = rand_seed_for_core(operation_attributes, i, ws.device_seed_offset);

        AddRuntimeArgsForNode(
            compute_run.runtime_arg_values,
            core,
            {{"seed", seed},
             {"from", from_bits},
             {"to", to_bits},
             {"tile_offset", tile_offset},
             {"units_per_core", units_per_core}});
        AddRuntimeArgsForNode(
            writer_run.runtime_arg_values, core, {{"tile_offset", tile_offset}, {"units_per_core", units_per_core}});
    }

    run_args.kernel_run_args.push_back(compute_run);
    run_args.kernel_run_args.push_back(writer_run);
    run_args.tensor_args.emplace(RAND_OUTPUT, TensorArgument{output.mesh_tensor()});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

tt::tt_metal::experimental::ProgramRunArgs RandDeviceOperation::RandProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    // seed/from/to are DYNAMIC (omitted from the cache key) and re-applied every dispatch; the static
    // tile_offset/units_per_core args are enqueue-invariant and retained from the cache-miss
    // SetProgramRunArgs. The output tensor arg is re-supplied so its (freshly allocated) address is
    // re-patched (UpdateProgramRunArgs applies both).
    const auto ws = compute_rand_work_split(operation_attributes, output, mesh_dispatch_coordinate);

    const float eps = 1e-6f;
    const uint32_t from_bits = std::bit_cast<uint32_t>(operation_attributes.from);
    const uint32_t to_bits = std::bit_cast<uint32_t>(operation_attributes.to - eps);

    ProgramRunArgs run_args;
    KernelRunArgs compute_run{.kernel = RAND_COMPUTE};
    const auto layout = rand_core_layout(ws);
    for (int i = 0; i < static_cast<int>(layout.size()); ++i) {
        const auto& core = layout[i].core;
        const uint32_t seed = rand_seed_for_core(operation_attributes, i, ws.device_seed_offset);
        AddRuntimeArgsForNode(
            compute_run.runtime_arg_values, core, {{"seed", seed}, {"from", from_bits}, {"to", to_bits}});
    }
    run_args.kernel_run_args.push_back(compute_run);
    run_args.tensor_args.emplace(RAND_OUTPUT, TensorArgument{output.mesh_tensor()});
    return run_args;
}

}  // namespace ttnn::operations::rand
