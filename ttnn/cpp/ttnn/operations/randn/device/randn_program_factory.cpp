// SPDX-FileCopyrightText: 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstring>
#include <random>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/tensor/types.hpp"
#include "randn_device_operation.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::randn {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

std::uniform_int_distribution distribution(1, std::numeric_limits<int32_t>::max());

namespace {
// Persistent generator for the unseeded path, so consecutive calls advance instead of repeating.
std::mt19937 default_rng(std::random_device{}());

// Spec names shared by create_program_artifacts and override_runtime_arguments.
const KernelSpecName WRITER{"writer"};
const KernelSpecName COMPUTE{"compute"};
const DFBSpecName DST{"dst"};
const TensorParamName OUTPUT{"output"};

// Core layout shared by create_program_artifacts and override_runtime_arguments. It mirrors
// split_work_to_cores (num_cores = min(tiles, grid)) + column-major grid_to_cores, so the cache-hit
// path can walk the cores without re-running the full work split.
uint32_t randn_num_cores(uint32_t units_to_divide, const CoreCoord& grid) {
    return std::min<uint32_t>(units_to_divide, grid.x * grid.y);
}
CoreCoord randn_core(uint32_t i, const CoreCoord& grid) { return {i / grid.y, i % grid.y}; }

}  // namespace

auto get_random_seed(std::mt19937& rng) -> uint32_t { return distribution(rng); }

ttnn::device_operation::ProgramArtifacts RandnDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    MeshDevice* device = output.device();
    auto grid = device->compute_with_storage_grid_size();

    // The Metal 2.0 binding layer works with the Metalium tensor type; extract once.
    const auto& output_mesh_tensor = output.mesh_tensor();

    uint32_t units_to_divide = output.physical_volume() / constants::TILE_HW;
    auto [num_cores, all_cores, core_group_1, core_group_2, units_per_core_group_1, units_per_core_group_2] =
        split_work_to_cores(grid, units_to_divide);
    TT_FATAL(
        num_cores == randn_num_cores(units_to_divide, grid),
        "randn core count {} diverged from split_work_to_cores ({}); override_runtime_arguments would patch the wrong "
        "cores",
        randn_num_cores(units_to_divide, grid),
        num_cores);

    DataType output_dtype = output.dtype();
    auto out_data_format = datatype_to_dataformat_converter(output_dtype);
    const uint32_t dtype_tile_size = tile_size(out_data_format);

    constexpr uint32_t in_out_num_tiles = 2;

    DataflowBufferSpec dst_dfb{
        .unique_id = DST,
        .entry_size = dtype_tile_size,
        .num_entries = in_out_num_tiles,
        .data_format_metadata = out_data_format,
    };

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/randn/device/kernels/";
    const std::string writer_file_path = kernels_dir_path + "writer_standard_normal.cpp";
    const std::string compute_file_path = kernels_dir_path + "compute_standard_normal.cpp";

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_file_path,
        .dfb_bindings =
            {
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
                    .accessor_name = "output",
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"start_id", "num_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    KernelSpec::CompilerOptions::Defines compute_defines;
    switch (output_dtype) {
        case DataType::BFLOAT16: compute_defines.emplace("OUTPUT_DTYPE_BFLOAT16", "1"); break;
        default: break;
    }

    auto compute_hw_config = ttnn::to_compute_hardware_config(operation_attributes.compute_kernel_config);
    // if fp32_dest_acc_en set to false a precision error may occur which makes generated number out of range
    // [from, to)
    compute_hw_config.enable_32_bit_dest = true;

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = compute_file_path,
        // Legacy compute kernels default to O3; Metal 2.0 defaults every kernel to O2.
        .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DST,
                    .accessor_name = "dst",
                    .endpoint_type = DFBEndpointType::PRODUCER,
                },
            },
        .runtime_arg_schema = {.runtime_arg_names = {"seed", "num_tiles"}},
        .hw_config = compute_hw_config,
    };

    std::mt19937 seeded_rng(operation_attributes.seed.value_or(0));
    std::mt19937& rng = operation_attributes.seed.has_value() ? seeded_rng : default_rng;

    KernelRunArgs writer_run_args{.kernel = WRITER};
    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = randn_core(i, grid);
        uint32_t units_per_core;
        if (core_group_1.contains(core)) {
            units_per_core = units_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            units_per_core = units_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        // The per-core seed is hash-excluded (see compute_program_hash): baked here for the cache-miss
        // build and redrawn on every cache hit by override_runtime_arguments.
        uint32_t seed = get_random_seed(rng);
        AddRuntimeArgsForNode(
            compute_run_args.runtime_arg_values, core, {{"seed", seed}, {"num_tiles", units_per_core}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"start_id", tile_offset}, {"num_tiles", units_per_core}});

        tile_offset += units_per_core;
    }

    ProgramSpec spec{
        .name = "randn",
        .kernels = {std::move(writer), std::move(compute)},
        .dataflow_buffers = {std::move(dst_dfb)},
        .tensor_parameters =
            {
                TensorParameter{.unique_id = OUTPUT, .spec = output_mesh_tensor.tensor_spec()},
            },
        .work_units =
            {
                WorkUnitSpec{
                    .name = "randn",
                    .kernels = {WRITER, COMPUTE},
                    .target_nodes = all_cores,
                },
            },
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(writer_run_args), std::move(compute_run_args)};
    run_args.tensor_args = {{OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_args),
    };
}

ProgramRunArgs RandnDeviceOperation::ProgramFactory::override_runtime_arguments(
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t& tensor_args,
    tensor_return_value_t& output,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Only the per-core seeds (hash-excluded) and the output address change between hits; tile
    // offsets and counts derive from the hashed shape and are left as built. On this concept the
    // framework refreshes nothing on our behalf, so the output tensor binding is re-supplied here.
    const auto grid = output.device()->compute_with_storage_grid_size();
    const uint32_t num_cores = randn_num_cores(output.physical_volume() / constants::TILE_HW, grid);

    std::mt19937 seeded_rng(operation_attributes.seed.value_or(0));
    std::mt19937& rng = operation_attributes.seed.has_value() ? seeded_rng : default_rng;

    KernelRunArgs compute_run_args{.kernel = COMPUTE};
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = randn_core(i, grid);
        AddRuntimeArgsForNode(compute_run_args.runtime_arg_values, core, {{"seed", get_random_seed(rng)}});
    }

    ProgramRunArgs params;
    params.kernel_run_args = {std::move(compute_run_args)};
    params.tensor_args = {{OUTPUT, output.mesh_tensor()}};
    return params;
}

}  // namespace ttnn::operations::randn
