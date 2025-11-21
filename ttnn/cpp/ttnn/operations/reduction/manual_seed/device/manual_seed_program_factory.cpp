// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::reduction::manual_seed::program {
using namespace tt::tt_metal;

ManualSeedSingleSeedToAllCoresProgramFactory::cached_program_t ManualSeedSingleSeedToAllCoresProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};
    // Get device
    const auto device = operation_attributes.device;
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const auto num_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    // Calculate core range
    CoreRangeSet core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);

    // Check for sub_core_grids
    if (operation_attributes.sub_core_grids.has_value()) {
        core_grid = operation_attributes.sub_core_grids.value();
    }

    // Create compute kernel
    std::vector<uint32_t> compute_args = {operation_attributes.seeds.value_or(0)};
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_simple_set_seed.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    return cached_program_t{std::move(program), {}};
}

void ManualSeedSingleSeedToAllCoresProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*output_tensor*/) {
    // NOTE: No runtime arguments to override for this OP
}

ManualSeedSingleSeedSingleCoreProgramFactory::cached_program_t ManualSeedSingleSeedSingleCoreProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    tt::tt_metal::Program program{};
    std::cout << "device->num_devices(): " << operation_attributes.device->num_devices() << std::endl;
    // Get device
    const auto device = operation_attributes.device;
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    std::cout << "compute_with_storage_grid_size.x: " << compute_with_storage_grid_size.x << std::endl;
    std::cout << "compute_with_storage_grid_size.y: " << compute_with_storage_grid_size.y << std::endl;
    const auto num_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;
    std::cout << "num_cores: " << num_cores << std::endl;
    // Calculate core range
    CoreRangeSet core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);
    std::cout << "core_grid size: " << core_grid.str() << std::endl;
    // Check for sub_core_grids
    if (operation_attributes.sub_core_grids.has_value()) {
        core_grid = operation_attributes.sub_core_grids.value();
    }
    const auto& cores = corerange_to_cores(core_grid, num_cores, true);
    const auto& core_chosen = cores.at(operation_attributes.user_ids.value_or(0));
    std::cout << "core_chosen: (" << core_chosen.x << ", " << core_chosen.y << ")" << std::endl;
    // Create compute kernel
    std::vector<uint32_t> compute_args = {operation_attributes.seeds.value_or(0)};
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_simple_set_seed.cpp",
        core_chosen,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    return cached_program_t{std::move(program), {}};
}

void ManualSeedSingleSeedSingleCoreProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*output_tensor*/) {
    // NOTE: No runtime arguments to override for this OP
}

}  // namespace ttnn::operations::reduction::manual_seed::program
