// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"

#include <vector>

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
    std::vector<uint32_t> compute_compile_time_args = {operation_attributes.seeds.value_or(0)};
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_simple_set_seed.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

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
    const auto& cores = corerange_to_cores(core_grid, num_cores, true);
    const auto& core_chosen = cores.at(operation_attributes.user_ids.value_or(0));

    // Create compute kernel
    std::vector<uint32_t> compute_compile_time_args = {operation_attributes.seeds.value_or(0)};
    tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_simple_set_seed.cpp",
        core_chosen,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});
    return cached_program_t{std::move(program), {}};
}

void ManualSeedSingleSeedSingleCoreProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*output_tensor*/) {
    // NOTE: No runtime arguments to override for this OP
}

ManualSeedSingleSeedSetCoresProgramFactory::cached_program_t ManualSeedSingleSeedSetCoresProgramFactory::create(
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
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_grid, num_cores, true);

    // Safety check
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSingleSeedSetCoresProgramFactory");

    // Tensor config info
    const auto& user_ids_tensor = tensor_args.user_ids.value();
    const tt::DataFormat user_ids_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(user_ids_tensor.dtype());
    const uint32_t user_ids_tensor_tile_size =
        user_ids_tensor.tensor_spec().tile().get_tile_size(user_ids_cb_data_format);
    const auto user_ids_tensor_buffer = user_ids_tensor.buffer();
    const auto number_of_ids = static_cast<uint32_t>(user_ids_tensor.logical_volume());

    // Circular buffers
    constexpr uint32_t user_ids_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig user_ids_cb_config =
        tt::tt_metal::CircularBufferConfig(user_ids_tensor_tile_size, {{user_ids_cb_index, user_ids_cb_data_format}})
            .set_page_size(user_ids_cb_index, user_ids_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, user_ids_cb_config);

    // Create core kernels
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> compute_kernel_ids;
    reader_kernel_ids.reserve(cores.size());
    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        const auto& core = cores[core_id];

        // Create reader kernel
        std::vector<uint32_t> reader_compile_time_args = {core_id, user_ids_cb_index};
        TensorAccessorArgs(*user_ids_tensor_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/dataflow/"
            "reader_manual_seed.cpp",
            core,
            tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {user_ids_tensor_buffer->address()});
        reader_kernel_ids.push_back(reader_kernel_id);

        // Create compute kernel
        std::vector<uint32_t> compute_compile_time_args = {
            core_id, user_ids_cb_index, operation_attributes.seeds.value_or(0)};
        tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
            "manual_seed_receive_set_seed.cpp",
            core,
            tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {number_of_ids});
        compute_kernel_ids.push_back(compute_kernel_id);
    }

    return cached_program_t{std::move(program), {reader_kernel_ids, compute_kernel_ids, cores}};
}

void ManualSeedSingleSeedSetCoresProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    // Get user_ids tensor info
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");
    const auto& user_ids_tensor = tensor_args.user_ids.value();
    const auto number_of_ids = static_cast<uint32_t>(user_ids_tensor.logical_volume());

    // Override runtime args for each core
    for (uint32_t i = 0; i < cached_program.shared_variables.cores.size(); ++i) {
        const auto& core = cached_program.shared_variables.cores[i];

        const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_ids[i];
        const auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_ids[i];

        auto& reader_runtime_args = GetRuntimeArgs(cached_program.program, reader_kernel_id, core);
        reader_runtime_args[0] = tensor_args.user_ids.value().buffer()->address();

        auto& compute_runtime_args = GetRuntimeArgs(cached_program.program, compute_kernel_id, core);
        compute_runtime_args[0] = number_of_ids;
    }
}

ManualSeedSetSeedsSetCoresProgramFactory::cached_program_t ManualSeedSetSeedsSetCoresProgramFactory::create(
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
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_grid, num_cores, true);

    // Safety check
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSingleSeedSetCoresProgramFactory");
    TT_FATAL(
        tensor_args.seeds.has_value(), "seeds tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");

    // Tensor config info
    const auto& user_ids_tensor = tensor_args.user_ids.value();
    const tt::DataFormat user_ids_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(user_ids_tensor.dtype());
    const uint32_t user_ids_tensor_tile_size =
        user_ids_tensor.tensor_spec().tile().get_tile_size(user_ids_cb_data_format);
    const auto user_ids_tensor_buffer = user_ids_tensor.buffer();
    const auto number_of_ids = static_cast<uint32_t>(user_ids_tensor.logical_volume());

    const auto& seeds_tensor = tensor_args.seeds.value();
    const tt::DataFormat seeds_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(seeds_tensor.dtype());
    const uint32_t seeds_tensor_tile_size = seeds_tensor.tensor_spec().tile().get_tile_size(seeds_cb_data_format);
    const auto seeds_tensor_buffer = seeds_tensor.buffer();

    // Circular buffers
    constexpr uint32_t user_ids_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig user_ids_cb_config =
        tt::tt_metal::CircularBufferConfig(user_ids_tensor_tile_size, {{user_ids_cb_index, user_ids_cb_data_format}})
            .set_page_size(user_ids_cb_index, user_ids_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, user_ids_cb_config);

    constexpr uint32_t seeds_cb_index = tt::CBIndex::c_1;
    const tt::tt_metal::CircularBufferConfig seeds_cb_config =
        tt::tt_metal::CircularBufferConfig(seeds_tensor_tile_size, {{seeds_cb_index, seeds_cb_data_format}})
            .set_page_size(seeds_cb_index, seeds_tensor_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, seeds_cb_config);

    // Create core kernels
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    std::vector<tt::tt_metal::KernelHandle> compute_kernel_ids;
    reader_kernel_ids.reserve(cores.size());
    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        const auto& core = cores[core_id];

        // Create reader kernel
        std::vector<uint32_t> reader_compile_time_args = {core_id, user_ids_cb_index, seeds_cb_index};
        TensorAccessorArgs(*user_ids_tensor_buffer).append_to(reader_compile_time_args);
        TensorAccessorArgs(*seeds_tensor_buffer).append_to(reader_compile_time_args);
        tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/dataflow/"
            "reader_manual_seed_read_all_data.cpp",
            core,
            tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});
        tt::tt_metal::SetRuntimeArgs(
            program, reader_kernel_id, core, {user_ids_tensor_buffer->address(), seeds_tensor_buffer->address()});
        reader_kernel_ids.push_back(reader_kernel_id);

        // Create compute kernel
        std::vector<uint32_t> compute_compile_time_args = {core_id, user_ids_cb_index, seeds_cb_index};
        tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
            "manual_seed_receive_all_data.cpp",
            core,
            tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, {number_of_ids});
        compute_kernel_ids.push_back(compute_kernel_id);
    }

    return cached_program_t{std::move(program), {reader_kernel_ids, compute_kernel_ids, cores}};
}

void ManualSeedSetSeedsSetCoresProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    // Get user_ids tensor info
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");
    TT_FATAL(
        tensor_args.seeds.has_value(), "seeds tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");

    const auto& user_ids_tensor = tensor_args.user_ids.value();
    const auto& seeds_tensor = tensor_args.seeds.value();
    const auto number_of_ids = static_cast<uint32_t>(user_ids_tensor.logical_volume());

    // Override runtime args for each core
    for (uint32_t i = 0; i < cached_program.shared_variables.cores.size(); ++i) {
        const auto& core = cached_program.shared_variables.cores[i];

        const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_ids[i];
        const auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_ids[i];

        auto& reader_runtime_args = GetRuntimeArgs(cached_program.program, reader_kernel_id, core);
        reader_runtime_args[0] = user_ids_tensor.buffer()->address();
        reader_runtime_args[1] = seeds_tensor.buffer()->address();

        auto& compute_runtime_args = GetRuntimeArgs(cached_program.program, compute_kernel_id, core);
        compute_runtime_args[0] = number_of_ids;
    }
}

}  // namespace ttnn::operations::reduction::manual_seed::program
