// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/execution_context.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <vector>

namespace ttnn::execution_context {

struct CurrentSubDeviceGuardImpl {};

namespace {

using Entry = std::pair<tt::tt_metal::distributed::MeshDevice*, tt::tt_metal::SubDeviceId>;
thread_local std::vector<Entry> g_current_sub_device_stack;

tt::tt_metal::SubDeviceId get_current_sub_device_id_impl(tt::tt_metal::distributed::MeshDevice* device) {
    if (device == nullptr) {
        return tt::tt_metal::SubDeviceId{0};
    }
    // Top of stack (last push) is the current; search from end for this device.
    for (auto it = g_current_sub_device_stack.rbegin(); it != g_current_sub_device_stack.rend(); ++it) {
        if (it->first == device) {
            return it->second;
        }
    }
    return device->get_sub_device_ids().at(0);
}

void current_sub_device_guard_deleter(CurrentSubDeviceGuardImpl* p) noexcept {
    if (p) {
        if (!g_current_sub_device_stack.empty()) {
            g_current_sub_device_stack.pop_back();
        }
        delete p;
    }
}

}  // namespace

CurrentSubDeviceGuard::CurrentSubDeviceGuard() : ptr_(nullptr, current_sub_device_guard_deleter) {}

CurrentSubDeviceGuard::CurrentSubDeviceGuard(CurrentSubDeviceGuardImpl* p) :
    ptr_(p, current_sub_device_guard_deleter) {}

tt::tt_metal::SubDeviceId get_current_sub_device_id(tt::tt_metal::IDevice* device) {
    if (device == nullptr) {
        return tt::tt_metal::SubDeviceId{0};
    }
    auto mesh = device->get_mesh_device();
    if (mesh) {
        return get_current_sub_device_id_impl(mesh.get());
    }
    return device->get_sub_device_ids().at(0);
}

tt::tt_metal::SubDeviceId get_current_sub_device_id(tt::tt_metal::distributed::MeshDevice* device) {
    if (device == nullptr) {
        return tt::tt_metal::SubDeviceId{0};
    }
    return get_current_sub_device_id_impl(device);
}

CurrentSubDeviceGuard set_current_sub_device(
    tt::tt_metal::distributed::MeshDevice* device, tt::tt_metal::SubDeviceId sub_device_id) {
    if (device == nullptr) {
        return CurrentSubDeviceGuard();
    }
    g_current_sub_device_stack.push_back({device, sub_device_id});
    return CurrentSubDeviceGuard(new CurrentSubDeviceGuardImpl());
}

}  // namespace ttnn::execution_context
