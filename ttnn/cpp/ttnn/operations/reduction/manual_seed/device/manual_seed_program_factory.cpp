// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "manual_seed_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "tt-metalium/core_coord.hpp"
#include "tt-metalium/kernel_types.hpp"

#include <vector>

namespace ttnn::prim {
using namespace tt::tt_metal;

namespace {
// Helper function to compute core grid from device and operation attributes
CoreRangeSet compute_core_grid(
    const ManualSeedParams& operation_attributes, const IDevice* device, uint32_t& out_num_cores) {
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

// Helper function to push a circular buffer descriptor for a tensor
void push_tensor_circular_buffer(
    ProgramDescriptor& desc, const CoreRangeSet& core_grid, const Tensor& tensor, uint32_t cb_index) {
    // Circular buffer config
    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tensor.dtype());
    const uint32_t tensor_tile_size = tensor.tensor_spec().tile().get_tile_size(cb_data_format);

    desc.cbs.push_back(CBDescriptor{
        .total_size = tensor_tile_size,
        .core_ranges = core_grid,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = cb_data_format,
            .page_size = tensor_tile_size,
        }}},
    });
}

}  // anonymous namespace

tt::tt_metal::ProgramDescriptor ManualSeedSingleSeedToAllCoresProgramFactory::create_descriptor(
    const ManualSeedParams& operation_attributes, const ManualSeedInputs& /*tensor_args*/, Tensor& /*output_tensor*/) {
    ProgramDescriptor desc;

    // Calculate core grid
    uint32_t num_cores{};
    CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);

    // Create compute kernel
    std::vector<uint32_t> compute_compile_time_args = {operation_attributes.seeds.value_or(0)};
    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/manual_seed_set_seed.cpp";

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kernel_path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{};
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

tt::tt_metal::ProgramDescriptor ManualSeedSingleSeedSingleCoreProgramFactory::create_descriptor(
    const ManualSeedParams& operation_attributes, const ManualSeedInputs& /*tensor_args*/, Tensor& /*output_tensor*/) {
    ProgramDescriptor desc;

    uint32_t num_cores{};
    CoreRangeSet core_grid = compute_core_grid(operation_attributes, operation_attributes.device, num_cores);
    const auto& cores = corerange_to_cores(core_grid, num_cores, true);
    const auto& core_chosen = cores.at(operation_attributes.user_ids.value_or(0));
    CoreRangeSet chosen_core_ranges{CoreRange(core_chosen, core_chosen)};

    // Create compute kernel
    std::vector<uint32_t> compute_compile_time_args = {operation_attributes.seeds.value_or(0)};
    const std::string kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/manual_seed_set_seed.cpp";

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = kernel_path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = chosen_core_ranges;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{};
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

tt::tt_metal::ProgramDescriptor ManualSeedSingleSeedSetCoresProgramFactory::create_descriptor(
    const ManualSeedParams& operation_attributes, const ManualSeedInputs& tensor_args, Tensor& /*output_tensor*/) {
    ProgramDescriptor desc;

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
    push_tensor_circular_buffer(desc, core_grid, user_ids_tensor, user_ids_cb_index);

    constexpr uint32_t kernel_communication_cb_index = tt::CBIndex::c_1;
    push_tensor_circular_buffer(desc, core_grid, user_ids_tensor, kernel_communication_cb_index);

    // Create core kernels
    const std::string reader_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/dataflow/"
        "reader_manual_seed_read_user_id.cpp";
    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/reduction/manual_seed/device/kernels/compute/"
        "manual_seed_single_seed_receive_user_id.cpp";

    // Create reader kernel
    std::vector<uint32_t> reader_compile_time_args = {user_ids_cb_index, kernel_communication_cb_index, number_of_ids};
    TensorAccessorArgs(*user_ids_tensor_buffer).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        // Get core
        const auto& core = cores[core_id];

        // Set runtime args for reader kernel
        reader_desc.emplace_runtime_args(core, {user_ids_tensor_buffer, core_id});
    }
    desc.kernels.push_back(std::move(reader_desc));

    // Create compute kernel
    std::vector<uint32_t> compute_compile_time_args = {
        kernel_communication_cb_index, operation_attributes.seeds.value_or(0)};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel_path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = compute_compile_time_args;
    compute_desc.config = ComputeConfigDescriptor{};
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

tt::tt_metal::ProgramDescriptor ManualSeedSetSeedsSetCoresProgramFactory::create_descriptor(
    const ManualSeedParams& operation_attributes, const ManualSeedInputs& tensor_args, Tensor& /*output_tensor*/) {
    ProgramDescriptor desc;

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
    push_tensor_circular_buffer(desc, core_grid, user_ids_tensor, user_ids_cb_index);

    constexpr uint32_t seeds_cb_index = tt::CBIndex::c_1;
    push_tensor_circular_buffer(desc, core_grid, seeds_tensor, seeds_cb_index);

    constexpr uint32_t kernel_communication_cb_index = tt::CBIndex::c_2;
    push_tensor_circular_buffer(desc, core_grid, seeds_tensor, kernel_communication_cb_index);

    // Create core kernels
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

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = reader_kernel_path;
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = core_grid;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    for (uint32_t core_id = 0; core_id < cores.size(); ++core_id) {
        // Get core
        const auto& core = cores[core_id];

        // Set runtime args for reader kernel
        reader_desc.emplace_runtime_args(core, {user_ids_tensor_buffer, seeds_tensor_buffer, core_id});
    }
    desc.kernels.push_back(std::move(reader_desc));

    // Create compute kernel
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel_path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = core_grid;
    compute_desc.compile_time_args = {kernel_communication_cb_index};
    compute_desc.config = ComputeConfigDescriptor{};
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
