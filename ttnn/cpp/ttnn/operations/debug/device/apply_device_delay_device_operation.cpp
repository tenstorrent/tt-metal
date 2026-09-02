// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "apply_device_delay_device_operation.hpp"

#include <cstddef>

#include <tt_stl/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::debug {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

void ApplyDeviceDelayDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /*tensor_args*/) {
    const auto& mesh_device = *operation_attributes.mesh_device;
    TT_FATAL(operation_attributes.mesh_device != nullptr, "mesh_device is nullptr");
    const auto& view = mesh_device.get_view();

    const auto& delays = operation_attributes.delays;

    TT_FATAL(view.is_mesh_2d(), "apply_device_delay currently supports only 2D mesh");
    TT_FATAL(
        delays.size() == view.num_rows(), "delays rows ({}) must match mesh rows ({})", delays.size(), view.num_rows());
    for (size_t r = 0; r < delays.size(); ++r) {
        TT_FATAL(
            delays[r].size() == view.num_cols(),
            "delays cols for row {} ({}) must match mesh cols ({})",
            r,
            delays[r].size(),
            view.num_cols());
    }
}

void ApplyDeviceDelayDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    // No additional validation needed on cache hit
}

ApplyDeviceDelayDeviceOperation::spec_return_value_t ApplyDeviceDelayDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return std::vector<tt::tt_metal::TensorSpec>{};
}

ApplyDeviceDelayDeviceOperation::tensor_return_value_t ApplyDeviceDelayDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return std::vector<ttnn::Tensor>{};
}

tt::tt_metal::ProgramDescriptor ApplyDeviceDelayDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "ApplyDeviceDelayDeviceOperation::create_descriptor requires a mesh dispatch coordinate");
    const ttnn::MeshCoordinate& mesh_coordinate = mesh_dispatch_coordinate.value();
    log_info(tt::LogAlways, "Creating delay program at mesh coordinate: {}", mesh_coordinate);

    tt::tt_metal::ProgramDescriptor desc;

    const auto subdevice_cores = corerange_to_cores(operation_attributes.worker_core_range_set);
    const CoreCoord target_core = subdevice_cores.at(0);
    const CoreRangeSet kernel_core_range_set{CoreRange(target_core, target_core)};

    tt::tt_metal::KernelDescriptor kernel_desc;
    kernel_desc.kernel_source = "ttnn/cpp/ttnn/operations/debug/device/kernels/dataflow/device_delay_spin.cpp";
    kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel_desc.core_ranges = kernel_core_range_set;
    kernel_desc.compile_time_args = {operation_attributes.delays[mesh_coordinate[0]][mesh_coordinate[1]]};
    // Default DataMovementConfig (RISCV_0 / NOC_0) preserves the original CreateKernel(... DataMovementConfig{...})
    // call.
    kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{};

    desc.kernels.push_back(std::move(kernel_desc));

    log_info(tt::LogAlways, "Created delay program at mesh coordinate: {}", mesh_coordinate);
    return desc;
}

}  // namespace ttnn::operations::debug

namespace ttnn::prim {

ttnn::operations::debug::ApplyDeviceDelayDeviceOperation::tensor_return_value_t apply_device_delay(
    ttnn::MeshDevice& mesh_device,
    const std::vector<std::vector<uint32_t>>& delays,
    const CoreRangeSet& subdevice_core_range_set) {
    using OperationType = ttnn::operations::debug::ApplyDeviceDelayDeviceOperation;

    log_info(tt::LogAlways, "Initializing delay op structs");
    auto operation_attributes = OperationType::operation_attributes_t{
        .delays = delays, .worker_core_range_set = subdevice_core_range_set, .mesh_device = &mesh_device};
    auto tensor_args = OperationType::tensor_args_t{};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
