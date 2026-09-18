// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/cluster.hpp>
#include <tt-metalium/hal.hpp>

namespace ttnn {

namespace device {

namespace {

tt::tt_metal::DispatchCoreType cluster_aware_dispatch_core_type() {
    const auto cluster_type = tt::tt_metal::GetClusterType();
    if (cluster_type == tt::tt_metal::ClusterType::N300 || cluster_type == tt::tt_metal::ClusterType::T3K ||
        cluster_type == tt::tt_metal::ClusterType::N300_2x2) {
        return tt::tt_metal::DispatchCoreType::ETH;
    }
    return tt::tt_metal::DispatchCoreType::WORKER;
}

tt::tt_metal::DispatchCoreAxis default_dispatch_core_axis(tt::tt_fabric::FabricTensixConfig fabric_tensix_config) {
    if (tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE &&
        fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::MUX) {
        return tt::tt_metal::DispatchCoreAxis::COL;
    }
    return tt::tt_metal::DispatchCoreAxis::ROW;
}

void validate_dispatch_core_config(
    std::optional<tt::tt_metal::DispatchCoreType> type,
    std::optional<tt::tt_metal::DispatchCoreAxis> axis,
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config) {
    if (type.has_value() && axis.has_value() && type.value() == tt::tt_metal::DispatchCoreType::ETH &&
        axis.value() == tt::tt_metal::DispatchCoreAxis::COL) {
        TT_THROW("COL axis is not supported for ETH dispatch core type");
    }

    if (axis.has_value() && axis.value() == tt::tt_metal::DispatchCoreAxis::ROW &&
        tt::tt_metal::hal::get_arch() == tt::ARCH::BLACKHOLE &&
        fabric_tensix_config != tt::tt_fabric::FabricTensixConfig::MUX) {
        TT_THROW("ROW dispatch core axis is not supported for blackhole arch unless fabric tensix MUX is enabled");
    }
}

}  // namespace

tt::tt_metal::DispatchCoreConfig create_cluster_aware_dispatch_config(
    std::optional<tt::tt_metal::DispatchCoreType> type,
    std::optional<tt::tt_metal::DispatchCoreAxis> axis,
    tt::tt_fabric::FabricTensixConfig fabric_tensix_config) {
    validate_dispatch_core_config(type, axis, fabric_tensix_config);

    const auto resolved_axis = axis.value_or(default_dispatch_core_axis(fabric_tensix_config));
    // When the type is not specified, the default depends on the axis: COL is only supported on WORKER
    // dispatch cores, so it must not fall back to the cluster-aware (potentially ETH) default type.
    const auto resolved_type = type.value_or(
        resolved_axis == tt::tt_metal::DispatchCoreAxis::COL ? tt::tt_metal::DispatchCoreType::WORKER
                                                             : cluster_aware_dispatch_core_type());
    return tt::tt_metal::DispatchCoreConfig(resolved_type, resolved_axis);
}

std::shared_ptr<MeshDevice> open_mesh_device(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const std::optional<tt::tt_metal::DispatchCoreConfig>& dispatch_core_config,
    size_t worker_l1_size) {
    return MeshDevice::create_unit_mesh(
        device_id,
        l1_small_size,
        trace_region_size,
        num_command_queues,
        dispatch_core_config.value_or(create_cluster_aware_dispatch_config()),
        {},
        worker_l1_size);
}

void enable_program_cache(IDevice& device) { device.enable_program_cache(); }

void disable_and_clear_program_cache(IDevice& device) { device.disable_and_clear_program_cache(); }


bool is_wormhole_or_blackhole(tt::ARCH arch) { return arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE; }

void deallocate_buffers(IDevice* device) { device->allocator()->deallocate_buffers(); }

// Device management for auto-formatting
// Note: This functionality is planned for deprecation in the future.
namespace {
MeshDevice* default_device = nullptr;

// The default device is a raw pointer with no ownership. If it still points at a device (or one of
// its submeshes) when that device is closed, every later GetDefaultDevice() hands out a dangling
// pointer: the Python binding resolves the dynamic type through the freed object's vtable and
// segfaults once the memory is reused. Drop the default before the device goes away.
void forget_default_device_if_closing(const MeshDevice& device) {
    if (default_device == nullptr) {
        return;
    }
    if (default_device == &device) {
        default_device = nullptr;
        return;
    }
    for (const auto& submesh : device.get_submeshes()) {
        if (default_device == submesh.get()) {
            default_device = nullptr;
            return;
        }
    }
}
}  // namespace

void close_device(MeshDevice& device) {
    forget_default_device_if_closing(device);
    device.close();
}

void SetDefaultDevice(MeshDevice* dev) { default_device = dev; }

MeshDevice* GetDefaultDevice() { return default_device; }

}  // namespace device

using namespace device;

}  // namespace ttnn
