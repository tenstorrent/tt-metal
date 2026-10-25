// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "subdevice_sync_device_operation.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::subdevice_sync {

using namespace tt::tt_metal;

void SubdeviceSyncDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attrs, const tensor_args_t& /*tensor_args*/) {
    TT_FATAL(attrs.mesh_device != nullptr, "subdevice_sync: mesh_device is nullptr");
    TT_FATAL(attrs.value > 0, "subdevice_sync: value must be > 0");
    const SubDeviceId sd{attrs.sub_device_id};
    const auto sd_cores = attrs.mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, sd);
    TT_FATAL(
        sd_cores.contains(attrs.core),
        "subdevice_sync: core {} is not a worker core of sub-device {} ({})",
        attrs.core,
        attrs.sub_device_id,
        sd_cores);
    if (attrs.mode == SubdeviceSyncMode::Wait) {
        TT_FATAL(attrs.target_core == attrs.core, "subdevice_wait: the semaphore word must be on the wait core");
    }
}

void SubdeviceSyncDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {}

SubdeviceSyncDeviceOperation::spec_return_value_t SubdeviceSyncDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return {};
}

SubdeviceSyncDeviceOperation::tensor_return_value_t SubdeviceSyncDeviceOperation::create_output_tensors(
    const operation_attributes_t& /*attrs*/, const tensor_args_t& /*tensor_args*/) {
    return {};
}

namespace {
std::vector<uint32_t> runtime_args(const SubdeviceSyncDeviceOperation::operation_attributes_t& attrs) {
    if (attrs.mode == SubdeviceSyncMode::Signal) {
        // Virtual (translated) NOC coords are identical on every chip of the mesh.
        const CoreCoord target = attrs.mesh_device->worker_core_from_logical_core(attrs.target_core);
        return {attrs.semaphore_address, static_cast<uint32_t>(target.x), static_cast<uint32_t>(target.y), attrs.value};
    }
    return {attrs.semaphore_address, attrs.value, attrs.timeout_cycles};
}
}  // namespace

ProgramDescriptor SubdeviceSyncDeviceOperation::SubdeviceSyncProgramFactory::create_descriptor(
    const operation_attributes_t& attrs,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/) {
    ProgramDescriptor desc;
    KernelDescriptor k;
    k.source_type = KernelDescriptor::SourceType::FILE_PATH;
    k.core_ranges = CoreRangeSet{CoreRange(attrs.core, attrs.core)};
    // BRISC on NOC0: the go signal lands on BRISC, so this is the lowest-latency start; nothing else runs here.
    k.config = DataMovementConfigDescriptor{
        .processor = DataMovementProcessor::RISCV_0, .noc = NOC::RISCV_0_default, .noc_mode = NOC_MODE::DM_DEDICATED_NOC};
    k.kernel_source = attrs.mode == SubdeviceSyncMode::Signal
                          ? "ttnn/cpp/ttnn/operations/experimental/subdevice_sync/device/kernels/subdevice_signal.cpp"
                          : "ttnn/cpp/ttnn/operations/experimental/subdevice_sync/device/kernels/subdevice_wait.cpp";
    k.runtime_args.emplace_back(attrs.core, runtime_args(attrs));
    desc.kernels.push_back(std::move(k));
    return desc;
}

void SubdeviceSyncDeviceOperation::SubdeviceSyncProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const operation_attributes_t& attrs,
    const tensor_args_t& /*tensor_args*/,
    tensor_return_value_t& /*tensor_return_value*/,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    constexpr uint32_t kernel_idx = 0;
    auto& args = GetRuntimeArgs(program, kernel_idx, attrs.core);
    const auto v = runtime_args(attrs);
    TT_FATAL(args.size() == v.size(), "subdevice_sync: runtime arg count mismatch ({} vs {})", args.size(), v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        args[i] = v[i];
    }
}

}  // namespace ttnn::operations::experimental::subdevice_sync

namespace ttnn::prim {

void subdevice_sync(
    ttnn::MeshDevice& mesh_device,
    ttnn::operations::experimental::subdevice_sync::SubdeviceSyncMode mode,
    uint32_t semaphore_address,
    const CoreCoord& core,
    const CoreCoord& target_core,
    uint32_t value,
    uint32_t timeout_cycles,
    tt::tt_metal::SubDeviceId sub_device_id) {
    using OperationType = ttnn::operations::experimental::subdevice_sync::SubdeviceSyncDeviceOperation;
    auto attrs = OperationType::operation_attributes_t{
        .mode = mode,
        .semaphore_address = semaphore_address,
        .core = core,
        .target_core = target_core,
        .value = value,
        .timeout_cycles = timeout_cycles,
        .sub_device_id = static_cast<uint8_t>(*sub_device_id),
        .mesh_device = &mesh_device};
    ttnn::device_operation::launch<OperationType>(attrs, OperationType::tensor_args_t{});
}

}  // namespace ttnn::prim
