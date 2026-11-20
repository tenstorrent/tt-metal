// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "signal_tensor_prefetcher_device_operation.hpp"

#include <filesystem>
#include <utility>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/workload/specification/program_spec.hpp>

namespace ttnn::operations::experimental::tensor_prefetcher {

namespace {

constexpr const char* kKernelPath =
    "ttnn/cpp/ttnn/operations/experimental/tensor_prefetcher/device/kernels/signal_tensor_prefetcher.cpp";
const tt::tt_metal::KernelSpecName kSignalKernel{"signal"};

tt::tt_metal::ProgramRunArgs signal_run_args(
    const SignalTensorPrefetcherDeviceOperation::operation_attributes_t& attrs) {
    tt::tt_metal::KernelRunArgs kernel_args{.kernel = kSignalKernel};
    tt::tt_metal::AddRuntimeArgsForNode(
        kernel_args.runtime_arg_values, attrs.core, {{"signal_addr", attrs.signal_addr}});
    tt::tt_metal::ProgramRunArgs run_args;
    run_args.kernel_run_args.push_back(std::move(kernel_args));
    return run_args;
}

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

ttnn::device_operation::ProgramArtifacts
SignalTensorPrefetcherDeviceOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/, tensor_return_value_t& /*output*/) {
    using tt::tt_metal::DataMovementHardwareConfig;
    tt::tt_metal::KernelSpec kernel{.unique_id = kSignalKernel, .source = std::filesystem::path{kKernelPath}};
    kernel.runtime_arg_schema.runtime_arg_names = {"signal_addr"};
    kernel.hw_config = DataMovementHardwareConfig{
        .config_1xx =
            DataMovementHardwareConfig::DataMovement1XXConfig{
                .processor = tt::tt_metal::DataMovementProcessor::RISCV_0, .noc = tt::tt_metal::NOC::NOC_0},
        .config_2xx = DataMovementHardwareConfig::DataMovement2XXConfig{}};
    tt::tt_metal::ProgramSpec spec{
        .name = "signal_tensor_prefetcher",
        .kernels = {std::move(kernel)},
        .work_units = {
            {.name = "signal",
             .kernels = {kSignalKernel},
             .target_nodes = tt::tt_metal::CoreRangeSet(tt::tt_metal::CoreRange(attrs.core, attrs.core))}}};
    return {.spec = std::move(spec), .run_params = signal_run_args(attrs)};
}

tt::tt_metal::ProgramRunArgs SignalTensorPrefetcherDeviceOperation::ProgramFactory::override_runtime_arguments(
    const operation_attributes_t& attrs,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*output*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    return signal_run_args(attrs);
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
