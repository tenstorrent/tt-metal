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

namespace {
// Helper function to compute core grid from device and operation attributes
CoreRangeSet compute_core_grid(
    const operation_attributes_t& operation_attributes, const IDevice* device, uint32_t& out_num_cores) {
    // Get device core grid
    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    out_num_cores = compute_with_storage_grid_size.x * compute_with_storage_grid_size.y;

    // Create core grid
    CoreRangeSet core_grid =
        tt::tt_metal::num_cores_to_corerangeset(out_num_cores, compute_with_storage_grid_size, true);

    // Override core grid if sub_core_grids is provided in operation attributes
    if (operation_attributes.sub_core_grids.has_value()) {
        core_grid = operation_attributes.sub_core_grids.value();
    }

    return core_grid;
}

// Helper function to create circular buffer for a tensor
void create_tensor_circular_buffer(
    Program& program, const CoreRangeSet& core_grid, const Tensor& tensor, uint32_t cb_index) {
    // Circular buffer config
    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    const uint32_t tensor_tile_size = tensor.tensor_spec().tile().get_tile_size(cb_data_format);

    // Create circular buffer config
    const tt::tt_metal::CircularBufferConfig cb_config =
        tt::tt_metal::CircularBufferConfig(tensor_tile_size, {{cb_index, cb_data_format}})
            .set_page_size(cb_index, tensor_tile_size);

    // Create circular buffer
    tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_config);
}

// Helper function to override runtime args for multi-core programs
template <typename CachedProgramType>
void override_multi_core_runtime_args(
    CachedProgramType& cached_program,
    const Tensor& user_ids_tensor,
    const std::optional<Tensor>& seeds_tensor = std::nullopt) {
    // Override runtime args for each core
    for (uint32_t i = 0; i < cached_program.shared_variables.cores.size(); ++i) {
        const auto& core = cached_program.shared_variables.cores[i];
        const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_ids[i];

        auto& reader_runtime_args = GetRuntimeArgs(cached_program.program, reader_kernel_id, core);
        reader_runtime_args[0] = user_ids_tensor.buffer()->address();
        if (seeds_tensor.has_value()) {
            reader_runtime_args[1] = seeds_tensor.value().buffer()->address();
        }
    }
}

}  // anonymous namespace

ManualSeedSingleSeedToAllCoresProgramFactory::cached_program_t ManualSeedSingleSeedToAllCoresProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    tt::tt_metal::Program program{};

    // Calculate core grid
    uint32_t num_cores{};
    CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);

    // Create compute kernel
    std::vector<uint32_t> compute_compile_time_args = {operation_attributes.seeds.value_or(0)};
    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/manual_seed_set_seed.cpp";
    tt::tt_metal::CreateKernel(
        program, kernel_path, core_grid, tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    return cached_program_t{std::move(program), {}};
}

void ManualSeedSingleSeedToAllCoresProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    // NOTE: No runtime arguments to override for this OP
}

ManualSeedSingleSeedSingleCoreProgramFactory::cached_program_t ManualSeedSingleSeedSingleCoreProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    tt::tt_metal::Program program{};

    uint32_t num_cores{};
    CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);
    const auto& cores = corerange_to_cores(core_grid, num_cores, true);
    const auto& core_chosen = cores.at(operation_attributes.user_ids.value_or(0));

    // Create compute kernel
    std::vector<uint32_t> compute_compile_time_args = {operation_attributes.seeds.value_or(0)};
    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/manual_seed_set_seed.cpp";
    tt::tt_metal::CreateKernel(
        program, kernel_path, core_chosen, tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});
    return cached_program_t{std::move(program), {}};
}

void ManualSeedSingleSeedSingleCoreProgramFactory::override_runtime_arguments(
    cached_program_t& /*cached_program*/,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& /*tensor_args*/,
    Tensor& /*output_tensor*/) {
    // NOTE: No runtime arguments to override for this OP
}

ManualSeedSingleSeedSetCoresProgramFactory::cached_program_t ManualSeedSingleSeedSetCoresProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& /*output_tensor*/) {
    tt::tt_metal::Program program{};

    // Safety check
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSingleSeedSetCoresProgramFactory");

    uint32_t num_cores{};
    const CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_grid, num_cores, true);

    // Tensor config info
    const auto& user_ids_tensor = tensor_args.user_ids.value();
    auto* const user_ids_tensor_buffer = user_ids_tensor.buffer();
    const auto number_of_ids = static_cast<uint32_t>(user_ids_tensor.logical_volume());

    // Create circular buffer for user_ids
    constexpr uint32_t user_ids_cb_index = tt::CBIndex::c_0;
    create_tensor_circular_buffer(program, core_grid, user_ids_tensor, user_ids_cb_index);

    constexpr uint32_t kernel_communication_cb_index = tt::CBIndex::c_1;
    create_tensor_circular_buffer(program, core_grid, user_ids_tensor, kernel_communication_cb_index);

    // Create core kernels
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    reader_kernel_ids.reserve(cores.size());
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/dataflow/"
        "reader_manual_seed_read_user_id.cpp";
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_single_seed_receive_user_id.cpp";

    // Create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {user_ids_cb_index, kernel_communication_cb_index, number_of_ids};
    TensorAccessorArgs(*user_ids_tensor_buffer).append_to(reader_compile_time_args);
    const tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, core_grid, tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    // Create compute kernel
    std::vector<uint32_t> compute_compile_time_args = {
        kernel_communication_cb_index, operation_attributes.seeds.value_or(0)};
    tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        core_grid,
        tt::tt_metal::ComputeConfig{.compile_args = compute_compile_time_args});

    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        // Get core
        const auto& core = cores[core_id];

        // Set runtime args for reader kernel
        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, {user_ids_tensor_buffer->address(), core_id});
        reader_kernel_ids.push_back(reader_kernel_id);
    }

    return cached_program_t{std::move(program), {reader_kernel_ids, cores}};
}

void ManualSeedSingleSeedSetCoresProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& /*output_tensor*/) {
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSingleSeedSetCoresProgramFactory");

    override_multi_core_runtime_args(cached_program, tensor_args.user_ids.value());
}

ManualSeedSetSeedsSetCoresProgramFactory::cached_program_t ManualSeedSetSeedsSetCoresProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& /*output_tensor*/) {
    tt::tt_metal::Program program{};

    // Safety checks
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");
    TT_FATAL(
        tensor_args.seeds.has_value(), "seeds tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");

    uint32_t num_cores{};
    const CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);
    const std::vector<CoreCoord>& cores = corerange_to_cores(core_grid, num_cores, true);

    // Tensor config info
    const auto& user_ids_tensor = tensor_args.user_ids.value();
    auto* const user_ids_tensor_buffer = user_ids_tensor.buffer();
    const auto number_of_ids = static_cast<uint32_t>(user_ids_tensor.logical_volume());

    const auto& seeds_tensor = tensor_args.seeds.value();
    auto* const seeds_tensor_buffer = seeds_tensor.buffer();

    // Create circular buffers
    constexpr uint32_t user_ids_cb_index = tt::CBIndex::c_0;
    create_tensor_circular_buffer(program, core_grid, user_ids_tensor, user_ids_cb_index);

    constexpr uint32_t seeds_cb_index = tt::CBIndex::c_1;
    create_tensor_circular_buffer(program, core_grid, seeds_tensor, seeds_cb_index);

    constexpr uint32_t kernel_communication_cb_index = tt::CBIndex::c_2;
    create_tensor_circular_buffer(program, core_grid, seeds_tensor, kernel_communication_cb_index);

    // Create core kernels
    std::vector<tt::tt_metal::KernelHandle> reader_kernel_ids;
    reader_kernel_ids.reserve(cores.size());
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/dataflow/"
        "reader_manual_seed_read_all_data.cpp";
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_receive_all_data.cpp";

    // Create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {
        user_ids_cb_index, seeds_cb_index, kernel_communication_cb_index, number_of_ids};
    TensorAccessorArgs(*user_ids_tensor_buffer).append_to(reader_compile_time_args);
    TensorAccessorArgs(*seeds_tensor_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program, reader_kernel_path, core_grid, tt::tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

    // Create compute kernel
    tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        core_grid,
        tt::tt_metal::ComputeConfig{.compile_args = {kernel_communication_cb_index}});

    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        // Get core
        const auto& core = cores[core_id];

        // Set runtime args for reader kernel
        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {user_ids_tensor_buffer->address(), seeds_tensor_buffer->address(), core_id});
        reader_kernel_ids.push_back(reader_kernel_id);
    }

    return cached_program_t{std::move(program), {reader_kernel_ids, cores}};
}

void ManualSeedSetSeedsSetCoresProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& /*output_tensor*/) {
    TT_FATAL(
        tensor_args.user_ids.has_value(),
        "user_ids tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");
    TT_FATAL(
        tensor_args.seeds.has_value(), "seeds tensor must be provided for ManualSeedSetSeedsSetCoresProgramFactory");

    override_multi_core_runtime_args(cached_program, tensor_args.user_ids.value(), tensor_args.seeds);
}

}  // namespace ttnn::operations::reduction::manual_seed::program
