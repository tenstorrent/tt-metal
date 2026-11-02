// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <filesystem>
#include <string_view>

#include "dropout_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <internal/mesh_device.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::experimental::prim {
namespace {
constexpr auto kWriterKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/writer_dropout_interleaved_start_id.cpp";
constexpr auto kReaderKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/dataflow/reader_dropout_interleaved_start_id.cpp";
constexpr auto kComputeKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/dropout/device/kernels/compute/dropout_kernel.cpp";

// Spec names. The kernels reach the DFBs as dfb::in0 / dfb::out and the tensors as
// tensor::input / tensor::output.
const tt::tt_metal::experimental::KernelSpecName kReader{"reader"};
const tt::tt_metal::experimental::KernelSpecName kWriter{"writer"};
const tt::tt_metal::experimental::KernelSpecName kComputeGroup1{"compute_g1"};
const tt::tt_metal::experimental::KernelSpecName kComputeGroup2{"compute_g2"};

const tt::tt_metal::experimental::DFBSpecName kSrc0Dfb{"in0"};
const tt::tt_metal::experimental::DFBSpecName kOutputDfb{"out"};

const tt::tt_metal::experimental::TensorParamName kInputTensor{"input"};
const tt::tt_metal::experimental::TensorParamName kOutputTensor{"output"};

constexpr uint32_t kNumInputTiles = 2;
constexpr uint32_t kNumOutputTiles = 2;

// Offsets the seed by the device ID so each device of a mesh draws a different mask.
// Single-sourced: used by DropoutMeshWorkloadFactory::create_mesh_workload_artifacts (cache miss) and by
// override_runtime_arguments (cache hit), which must reproduce the same seed bit-for-bit.
uint32_t per_device_seed(
    const DropoutParams& args,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate,
    const ttnn::Tensor& input_tensor) {
    auto* device = input_tensor.device();
    return args.seed + (mesh_dispatch_coordinate.has_value()
                            ? tt::tt_metal::internal::get_device(*device, *mesh_dispatch_coordinate)->id()
                            : device->id());
}

}  // namespace

using namespace tt::constants;

/**
 *   Helper struct to hold all kernel specs we create,
 *        used during runtime argument setup.
 */
struct DropoutKernels {
    tt::tt_metal::experimental::KernelSpec reader;
    tt::tt_metal::experimental::KernelSpec writer;
    tt::tt_metal::experimental::KernelSpec compute_group_1;
    std::optional<tt::tt_metal::experimental::KernelSpec> compute_group_2;
};

/**
 *   Helper struct to hold the runtime arguments of every kernel in DropoutKernels.
 */
struct DropoutKernelRunArgs {
    tt::tt_metal::experimental::KernelRunArgs reader{.kernel = kReader};
    tt::tt_metal::experimental::KernelRunArgs writer{.kernel = kWriter};
    tt::tt_metal::experimental::KernelRunArgs compute_group_1{.kernel = kComputeGroup1};
    std::optional<tt::tt_metal::experimental::KernelRunArgs> compute_group_2;
};

/**
 *   Create and configure a dataflow buffer spec.
 */
inline tt::tt_metal::experimental::DataflowBufferSpec create_dataflow_buffer(
    const tt::tt_metal::experimental::DFBSpecName& dfb_name,
    tt::DataFormat data_format,
    uint32_t single_tile_size,
    uint32_t num_tiles) {
    using namespace tt::tt_metal::experimental;

    return DataflowBufferSpec{
        .unique_id = dfb_name,
        .entry_size = single_tile_size,
        .num_entries = num_tiles,
        .data_format_metadata = data_format,
    };
}

/**
 *   Create a reader kernel spec.
 */
inline tt::tt_metal::experimental::KernelSpec create_reader_kernel(std::string_view kernel_path) {
    using namespace tt::tt_metal::experimental;

    return KernelSpec{
        .unique_id = kReader,
        .source = std::filesystem::path(kernel_path),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = kSrc0Dfb,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = kInputTensor,
                    .accessor_name = "input",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
}

/**
 *   Create a writer kernel spec.
 */
inline tt::tt_metal::experimental::KernelSpec create_writer_kernel(std::string_view kernel_path) {
    using namespace tt::tt_metal::experimental;

    return KernelSpec{
        .unique_id = kWriter,
        .source = std::filesystem::path(kernel_path),
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = kOutputDfb,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
            },
        .tensor_bindings =
            {
                TensorBinding{
                    .tensor_parameter_name = kOutputTensor,
                    .accessor_name = "output",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
}

/**
 * Create a compute kernel spec (for dropout) with the given compile-time arguments.
 */
inline tt::tt_metal::experimental::KernelSpec create_compute_kernel(
    const tt::tt_metal::experimental::KernelSpecName& unique_id,
    tt::tt_metal::experimental::KernelSpec::CompileTimeArgs&& compile_time_args,
    std::string_view kernel_path,
    bool math_approx_mode) {
    using namespace tt::tt_metal::experimental;

    return KernelSpec{
        .unique_id = unique_id,
        .source = std::filesystem::path(kernel_path),
        .compiler_options = {.opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = kSrc0Dfb,
                    .accessor_name = "in0",
                    .endpoint_type = DFBEndpointType::CONSUMER,
                },
                DFBBinding{
                    .dfb_spec_name = kOutputDfb,
                    .accessor_name = "out",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .compile_time_args = std::move(compile_time_args),
        .runtime_arg_schema = {.runtime_arg_names = {"seed"}},
        .hw_config =
            ComputeHardwareConfig{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode =
                    math_approx_mode ? tt::tt_metal::Precision::Approximate : tt::tt_metal::Precision::Precise,
                .enable_32_bit_dest = false,  // fp32_dest_acc_en = false
                .double_buffer_dest = true,   // dst_full_sync_en = false
            },
    };
}

// Work split used by create_program_artifacts (cache miss) and override_runtime_arguments (cache hit).
struct DropoutCoreSplit {
    uint32_t num_cores = 0;
    uint32_t num_cores_y = 0;
    tt::tt_metal::CoreRangeSet all_cores;
    tt::tt_metal::CoreRangeSet core_group_1;
    tt::tt_metal::CoreRangeSet core_group_2;
    uint32_t num_tiles_per_core_group_1 = 0;
    uint32_t num_tiles_per_core_group_2 = 0;
};

DropoutCoreSplit dropout_core_split(const Tensor& input) {
    auto grid = input.device()->compute_with_storage_grid_size();
    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_tiles);
    return {
        num_cores,
        grid.y,
        all_cores,
        core_group_1,
        core_group_2,
        num_tiles_per_core_group_1,
        num_tiles_per_core_group_2};
}

// Per-core slice of the work split.
struct DropoutCoreWork {
    tt::tt_metal::CoreCoord core;
    uint32_t num_tiles = 0;
    uint32_t tile_offset = 0;
    bool in_group_1 = false;  // otherwise in core_group_2
};

// Walks the cores in the exact order create_program_artifacts emits runtime args for them. Shared with
// override_runtime_arguments so the per-core layout the cache-hit patch writes cannot drift from the
// one the cache-miss build baked.
template <typename Fn>
void for_each_dropout_core(const DropoutCoreSplit& split, const Fn& fn) {
    for (uint32_t i = 0, num_tiles_written = 0; i < split.num_cores; i++) {
        const tt::tt_metal::CoreCoord core = {i / split.num_cores_y, i % split.num_cores_y};
        const bool in_group_1 = split.core_group_1.contains(core);
        TT_FATAL(
            in_group_1 || split.core_group_2.contains(core),
            "Core ({}, {}) is not in the specified core ranges",
            core.x,
            core.y);
        const uint32_t num_tiles = in_group_1 ? split.num_tiles_per_core_group_1 : split.num_tiles_per_core_group_2;

        fn(DropoutCoreWork{core, num_tiles, num_tiles_written, in_group_1});

        num_tiles_written += num_tiles;
    }
}

/**
 * Set up the runtime arguments for the relevant kernels (reader, writer, compute G1, compute G2)
 *        for each core in the grid.
 */
inline void assign_per_core_runtime_args(DropoutKernelRunArgs& run_args, const DropoutCoreSplit& split, uint32_t seed) {
    using namespace tt::tt_metal::experimental;

    for_each_dropout_core(split, [&](const DropoutCoreWork& work) {
        // Compute kernel: (seed)
        if (work.in_group_1) {
            AddRuntimeArgsForNode(run_args.compute_group_1.runtime_arg_values, work.core, {{"seed", seed}});
        } else {
            TT_FATAL(run_args.compute_group_2.has_value(), "Core group 2 descriptor should be present");
            AddRuntimeArgsForNode(run_args.compute_group_2->runtime_arg_values, work.core, {{"seed", seed}});
        }

        // Reader kernel: (number_of_tiles, offset_in_tiles).  src/dst reach the kernels as the input /
        // output tensor bindings, so this cache-miss build binds their current buffers; on a cache hit
        // override_runtime_arguments re-supplies them (correct for the input==output in-place case).
        AddRuntimeArgsForNode(
            run_args.reader.runtime_arg_values,
            work.core,
            {
                {"num_tiles", work.num_tiles},
                {"start_id", work.tile_offset},
            });

        // Writer kernel: (number_of_tiles, offset_in_tiles)
        AddRuntimeArgsForNode(
            run_args.writer.runtime_arg_values,
            work.core,
            {
                {"num_tiles", work.num_tiles},
                {"start_id", work.tile_offset},
            });
    });
}

ttnn::device_operation::ProgramArtifacts DropoutProgramFactory::create_program_artifacts(
    const DropoutParams& args, const DropoutInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    // -------------------------------------------------------------------------
    // 1) Setup device, data formats, tile sizes, and compute split
    // -------------------------------------------------------------------------
    const auto& input = tensor_args.input;

    tt::DataFormat data_fmt_in = datatype_to_dataformat_converter(input.dtype());
    tt::DataFormat data_fmt_out = datatype_to_dataformat_converter(output.dtype());

    uint32_t single_tile_size_in = tt::tile_size(data_fmt_in);
    uint32_t single_tile_size_out = tt::tile_size(data_fmt_out);

    // Kept whole so it can be handed to for_each_dropout_core, the walk shared with the cache-hit patch.
    const auto split = dropout_core_split(input);
    const auto& core_group_1 = split.core_group_1;
    const auto& core_group_2 = split.core_group_2;
    const uint32_t num_tiles_per_core_group_1 = split.num_tiles_per_core_group_1;
    const uint32_t num_tiles_per_core_group_2 = split.num_tiles_per_core_group_2;

    // -------------------------------------------------------------------------
    // 2) Create and configure dataflow buffers
    // -------------------------------------------------------------------------
    Group<DataflowBufferSpec> dataflow_buffers = {
        create_dataflow_buffer(kSrc0Dfb, data_fmt_in, single_tile_size_in, kNumInputTiles),
        create_dataflow_buffer(kOutputDfb, data_fmt_out, single_tile_size_out, kNumOutputTiles),
    };

    // -------------------------------------------------------------------------
    // 3) Create reader/writer kernels
    // -------------------------------------------------------------------------
    DropoutKernels kernels{
        .reader = create_reader_kernel(kReaderKernelPath),
        .writer = create_writer_kernel(kWriterKernelPath),
    };

    // -------------------------------------------------------------------------
    // 4) Create compute kernels for dropout
    // -------------------------------------------------------------------------
    uint32_t uscale = std::bit_cast<uint32_t>(args.scale);

    // Convert probability (args.prob) to integer representation
    uint32_t prob_int = static_cast<uint32_t>(static_cast<double>(INT_MAX) * args.prob);

    // Group 1 compile-time arguments
    KernelSpec::CompileTimeArgs compute_group_1_args = {
        {"per_core_block_cnt", num_tiles_per_core_group_1},
        {"per_core_block_dim", 1},
        {"int_probability", prob_int},
        {"int_scale_factor", uscale},
    };

    bool math_approx_mode = false;

    kernels.compute_group_1 =
        create_compute_kernel(kComputeGroup1, std::move(compute_group_1_args), kComputeKernelPath, math_approx_mode);

    // Group 2 (if present) compile-time arguments
    if (!core_group_2.ranges().empty()) {
        KernelSpec::CompileTimeArgs compute_group_2_args = {
            {"per_core_block_cnt", num_tiles_per_core_group_2},
            {"per_core_block_dim", 1},
            {"int_probability", prob_int},
            {"int_scale_factor", uscale},
        };

        kernels.compute_group_2 = create_compute_kernel(
            kComputeGroup2, std::move(compute_group_2_args), kComputeKernelPath, math_approx_mode);
    }

    // -------------------------------------------------------------------------
    // 5) Assign runtime args for each core
    // -------------------------------------------------------------------------
    DropoutKernelRunArgs kernel_run_args;
    if (kernels.compute_group_2.has_value()) {
        kernel_run_args.compute_group_2 = KernelRunArgs{.kernel = kComputeGroup2};
    }
    assign_per_core_runtime_args(kernel_run_args, split, args.seed);

    // -------------------------------------------------------------------------
    // 6) Return the fully configured spec and its run args
    // -------------------------------------------------------------------------
    // Each core runs the reader, the writer, and the compute kernel of its core group.
    Group<WorkUnitSpec> work_units = {
        WorkUnitSpec{
            .name = "core_group_1",
            .kernels = {kReader, kWriter, kComputeGroup1},
            .target_nodes = core_group_1,
        },
    };
    if (kernels.compute_group_2.has_value()) {
        work_units.push_back(WorkUnitSpec{
            .name = "core_group_2",
            .kernels = {kReader, kWriter, kComputeGroup2},
            .target_nodes = core_group_2,
        });
    }

    Group<KernelSpec> kernel_specs = {
        std::move(kernels.reader),
        std::move(kernels.writer),
        std::move(kernels.compute_group_1),
    };
    if (kernels.compute_group_2.has_value()) {
        kernel_specs.push_back(std::move(*kernels.compute_group_2));
    }

    ProgramSpec spec{
        .name = "dropout",
        .kernels = std::move(kernel_specs),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters =
            {
                TensorParameter{.unique_id = kInputTensor, .spec = input.tensor_spec()},
                TensorParameter{.unique_id = kOutputTensor, .spec = output.tensor_spec()},
            },
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        std::move(kernel_run_args.reader),
        std::move(kernel_run_args.writer),
        std::move(kernel_run_args.compute_group_1),
    };
    if (kernel_run_args.compute_group_2.has_value()) {
        run_args.kernel_run_args.push_back(std::move(*kernel_run_args.compute_group_2));
    }
    run_args.tensor_args.emplace(kInputTensor, input.mesh_tensor());
    run_args.tensor_args.emplace(kOutputTensor, output.mesh_tensor());

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

ttnn::device_operation::MeshWorkloadArtifacts DropoutMeshWorkloadFactory::create_mesh_workload_artifacts(
    const DropoutParams& args,
    const DropoutInputs& tensor_args,
    Tensor& output,
    const ttnn::MeshCoordinateRangeSet& tensor_coords) {
    TT_ASSERT(args.use_per_device_seed, "DropoutMeshWorkloadFactory should only be used if per-device seed is used.");
    // One program per coordinate: each device's program has its own seed.
    ttnn::device_operation::MeshWorkloadArtifacts artifacts;
    for (const auto& coord : tensor_coords.coords()) {
        DropoutParams effective_args = args;
        effective_args.seed = per_device_seed(args, coord, tensor_args.input);
        auto per_coord = DropoutProgramFactory::create_program_artifacts(effective_args, tensor_args, output);
        artifacts.programs.push_back({
            .range = ttnn::MeshCoordinateRange(coord),
            .spec = std::move(per_coord.spec),
            .run_params = std::move(per_coord.run_params),
        });
    }
    return artifacts;
}

tt::tt_metal::experimental::ProgramRunArgs DropoutProgramFactory::override_runtime_arguments(
    const DropoutParams& operation_attributes,
    const DropoutInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    using namespace tt::tt_metal::experimental;

    const auto& input = tensor_args.input;

    // `seed` is excluded from the program hash, so the cached program still carries the first miss's
    // seed and it must be re-derived here exactly as the selected factory derives it:
    // DropoutMeshWorkloadFactory offsets it by the dispatch coordinate's device id.
    const uint32_t seed = operation_attributes.use_per_device_seed
                              ? per_device_seed(operation_attributes, mesh_dispatch_coordinate, input)
                              : operation_attributes.seed;

    // override_runtime_arguments supersedes the framework's tensor-binding refresh, so both tensor
    // bindings are ours to re-supply. Each comes from its own tensor, which is what makes the in-place
    // case (input == output, so both bindings name the same tensor) correct.
    ProgramRunArgs run_args;
    run_args.tensor_args.emplace(kInputTensor, input.mesh_tensor());
    run_args.tensor_args.emplace(kOutputTensor, tensor_return_value.mesh_tensor());

    // Everything else the build emitted (tile counts/offsets) derives from the input spec and the
    // compute grid, both fixed on a cache hit; rewritten anyway since the shared walk already has them.
    // Both DFBs are program-local (not borrowed from a tensor), so there is no DFB address to re-point.
    const DropoutCoreSplit split = dropout_core_split(input);
    const bool has_compute_group_2 = !split.core_group_2.ranges().empty();
    KernelRunArgs reader_run_args{.kernel = kReader};
    KernelRunArgs writer_run_args{.kernel = kWriter};
    KernelRunArgs compute_group_1_run_args{.kernel = kComputeGroup1};
    KernelRunArgs compute_group_2_run_args{.kernel = kComputeGroup2};

    for_each_dropout_core(split, [&](const DropoutCoreWork& work) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            work.core,
            {
                {"num_tiles", work.num_tiles},
                {"start_id", work.tile_offset},
            });

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            work.core,
            {
                {"num_tiles", work.num_tiles},
                {"start_id", work.tile_offset},
            });

        TT_FATAL(work.in_group_1 || has_compute_group_2, "Core group 2 kernel should be present");
        auto& compute_run_args = work.in_group_1 ? compute_group_1_run_args : compute_group_2_run_args;
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, work.core, {{"seed", seed}});
    });

    run_args.kernel_run_args = {
        std::move(reader_run_args),
        std::move(writer_run_args),
        std::move(compute_group_1_run_args),
    };
    if (has_compute_group_2) {
        run_args.kernel_run_args.push_back(std::move(compute_group_2_run_args));
    }
    return run_args;
}

tt::tt_metal::experimental::ProgramRunArgs DropoutMeshWorkloadFactory::override_runtime_arguments(
    const DropoutParams& operation_attributes,
    const DropoutInputs& tensor_args,
    Tensor& tensor_return_value,
    const ttnn::MeshCoordinateRange& range) {
    return DropoutProgramFactory::override_runtime_arguments(
        operation_attributes, tensor_args, tensor_return_value, range.start_coord());
}

}  // namespace ttnn::experimental::prim
