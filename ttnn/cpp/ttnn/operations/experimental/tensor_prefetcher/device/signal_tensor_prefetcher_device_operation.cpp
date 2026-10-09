// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "signal_tensor_prefetcher_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>

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

void SignalTensorPrefetcherDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

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

SignalTensorPrefetcherDeviceOperation::ProgramFactory::cached_program_t
SignalTensorPrefetcherDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*output*/) {
    tt::tt_metal::Program program{};
    const tt::tt_metal::KernelHandle kernel_id = tt::tt_metal::CreateKernel(
        program,
        kKernelPath,
        attrs.core,
        tt::tt_metal::DataMovementConfig{
            .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0});
    tt::tt_metal::SetRuntimeArgs(program, kernel_id, attrs.core, {attrs.signal_addr});
    return {std::move(program), shared_variables_t{.kernel_id = kernel_id, .core = attrs.core}};
}

void SignalTensorPrefetcherDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& attrs,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*output*/) {
    const auto& shared = cached_program.shared_variables;
    tt::tt_metal::GetRuntimeArgs(cached_program.program, shared.kernel_id, shared.core)[0] = attrs.signal_addr;
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
