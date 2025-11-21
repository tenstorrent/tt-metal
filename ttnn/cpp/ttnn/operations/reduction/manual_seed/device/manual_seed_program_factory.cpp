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
    std::cout << "ManualSeedSingleSeedToAllCoresProgramFactory::create called" << std::endl;
    // Get device
    // const auto device = operation_attributes.device;
    // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // const auto num_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    // // Calculate core range
    // CoreRangeSet core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size,
    // true);

    // if (operation_attributes.sub_core_grids.has_value()) {
    //     core_grid = operation_attributes.sub_core_grids.value();
    // }
    // // const auto cores = corerange_to_cores(core_grid, num_cores, true);

    // std::vector<uint32_t> compute_args = {operation_attributes.seeds.value()};
    // tt::tt_metal::CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
    //     "manual_seed_single_seed_to_all_cores.cpp",
    //     core_grid,
    //     tt::tt_metal::ComputeConfig{.compile_args = compute_args});
    // std::cout << "ManualSeedSingleSeedToAllCoresProgramFactory::create completed" << std::endl;
    return cached_program_t{std::move(program), {}};
}

void ManualSeedSingleSeedToAllCoresProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*output_tensor*/) {
    // NOTE: No runtime arguments to override for this OP
}
}  // namespace ttnn::operations::reduction::manual_seed::program
