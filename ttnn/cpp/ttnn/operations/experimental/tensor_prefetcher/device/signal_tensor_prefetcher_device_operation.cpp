// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "signal_tensor_prefetcher_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/mesh_device.hpp>

namespace ttnn::operations::experimental::tensor_prefetcher {

namespace {
constexpr const char* kKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/tensor_prefetcher/device/kernels/signal_tensor_prefetcher.cpp";
}  // namespace

void SignalTensorPrefetcherDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(attrs.mesh_device != nullptr, "signal_tensor_prefetcher requires a mesh device");
    const tt::tt_metal::CoreCoord grid = attrs.mesh_device->compute_with_storage_grid_size();
    TT_FATAL(
        attrs.core.x < grid.x && attrs.core.y < grid.y,
        "signal_tensor_prefetcher core ({}, {}) is outside the {}x{} worker grid",
        attrs.core.x,
        attrs.core.y,
        grid.x,
        grid.y);
}

SignalTensorPrefetcherDeviceOperation::spec_return_value_t SignalTensorPrefetcherDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return {};
}

SignalTensorPrefetcherDeviceOperation::tensor_return_value_t
SignalTensorPrefetcherDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return {};
}

ttsl::hash::hash_t SignalTensorPrefetcherDeviceOperation::compute_program_hash(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/) {
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<SignalTensorPrefetcherDeviceOperation>, attrs.core);
}

tt::tt_metal::ProgramDescriptor SignalTensorPrefetcherDeviceOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*output*/) {
    tt::tt_metal::KernelDescriptor kernel_desc;
    kernel_desc.kernel_source = kKernelPath;
    kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel_desc.core_ranges = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(attrs.core, attrs.core));
    kernel_desc.runtime_args.emplace_back(
        attrs.core, tt::tt_metal::KernelDescriptor::CoreRuntimeArgs{attrs.signal_addr});
    kernel_desc.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0};

    tt::tt_metal::ProgramDescriptor desc;
    desc.kernels.push_back(std::move(kernel_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::tensor_prefetcher

namespace ttnn::prim {
void signal_tensor_prefetcher(
    ttnn::MeshDevice* mesh_device, uint32_t signal_addr, const tt::tt_metal::CoreCoord& core) {
    using OperationType = ttnn::operations::experimental::tensor_prefetcher::SignalTensorPrefetcherDeviceOperation;
    const OperationType::operation_attributes_t attrs{
        .signal_addr = signal_addr, .core = core, .mesh_device = mesh_device};
    ttnn::device_operation::launch<OperationType>(attrs, OperationType::tensor_args_t{});
}
}  // namespace ttnn::prim
