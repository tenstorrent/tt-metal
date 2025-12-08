// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device.hpp"

namespace ttnn {

namespace device {

std::shared_ptr<MeshDevice> open_mesh_device(
    int device_id,
    size_t l1_small_size,
    size_t trace_region_size,
    size_t num_command_queues,
    const tt::tt_metal::DispatchCoreConfig& dispatch_core_config,
    size_t worker_l1_size) {
    return MeshDevice::create_unit_mesh(
        device_id, l1_small_size, trace_region_size, num_command_queues, dispatch_core_config, {}, worker_l1_size);
}

void enable_program_cache(IDevice& device) { device.enable_program_cache(); }

void disable_and_clear_program_cache(IDevice& device) { device.disable_and_clear_program_cache(); }

void close_device(MeshDevice& device) { device.close(); }

bool is_wormhole_or_blackhole(tt::ARCH arch) { return arch == tt::ARCH::WORMHOLE_B0 or arch == tt::ARCH::BLACKHOLE; }

void deallocate_buffers(IDevice* device) { device->allocator()->deallocate_buffers(); }

// Device management for auto-formatting
// Note: This functionality is planned for deprecation in the future.
namespace {
MeshDevice* default_device = nullptr;
}  // namespace

void SetDefaultDevice(MeshDevice* dev) { default_device = dev; }

MeshDevice* GetDefaultDevice() { return default_device; }

}  // namespace device

using namespace device;

}  // namespace ttnn
