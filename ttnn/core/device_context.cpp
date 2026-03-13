// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/device_context.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-logger/tt-logger.hpp>
#include <tt_stl/assert.hpp>

#include <algorithm>
#include <cstdlib>
#include <ranges>
#include <string>
#include <variant>
#include <vector>

namespace ttnn {

struct CurrentSubDeviceGuardImpl {
    tt::tt_metal::distributed::MeshDevice* device;
    tt::tt_metal::SubDeviceId sub_device_id;
    CurrentSubDeviceGuardImpl(tt::tt_metal::distributed::MeshDevice* d, tt::tt_metal::SubDeviceId id) :
        device(d), sub_device_id(id) {}
};

namespace {

using Entry = std::pair<tt::tt_metal::distributed::MeshDevice*, tt::tt_metal::SubDeviceId>;
thread_local std::vector<Entry> g_current_sub_device_stack;

tt::tt_metal::SubDeviceId get_current_sub_device_id_impl(tt::tt_metal::distributed::MeshDevice* device) {
    TT_FATAL(device != nullptr, "DeviceContext: device must not be null");
    for (auto& it : std::ranges::reverse_view(g_current_sub_device_stack)) {
        if (it.first == device) {
            return it.second;
        }
    }
    return device->get_sub_device_ids().at(0);
}

void current_sub_device_guard_deleter(CurrentSubDeviceGuardImpl* p) {
    if (p) {
        const bool stack_ok = !g_current_sub_device_stack.empty() &&
                              g_current_sub_device_stack.back().first == p->device &&
                              g_current_sub_device_stack.back().second == p->sub_device_id;
        if (!stack_ok) {
            log_critical(tt::LogAlways, "CurrentSubDeviceGuard destroyed out of order or stack corrupted");
            std::abort();
        }
        g_current_sub_device_stack.pop_back();
        delete p;
    }
}

tt::tt_metal::IDevice* get_reference_device_from_mesh(const tt::tt_metal::distributed::MeshDevice* mesh_device) {
    const auto devices = mesh_device->get_devices();
    TT_FATAL(!devices.empty(), "DeviceContext: MeshDevice has no devices");
    return devices[0];
}

tt::tt_metal::CoreCoord get_compute_with_storage_grid_size_mesh(const tt::tt_metal::distributed::MeshDevice* device) {
    TT_FATAL(device != nullptr, "DeviceContext: device must not be null");
    const auto sub_device_id =
        get_current_sub_device_id_impl(const_cast<tt::tt_metal::distributed::MeshDevice*>(device));
    return device->worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id)
        .bounding_box()
        .grid_size();
}

tt::tt_metal::CoreCoord get_logical_grid_size_mesh(const tt::tt_metal::distributed::MeshDevice* device) {
    TT_FATAL(device != nullptr, "DeviceContext: device must not be null");
    return get_reference_device_from_mesh(device)->logical_grid_size();
}

tt::tt_metal::CoreCoord get_dram_grid_size_mesh(const tt::tt_metal::distributed::MeshDevice* device) {
    TT_FATAL(device != nullptr, "DeviceContext: device must not be null");
    return get_reference_device_from_mesh(device)->dram_grid_size();
}

}  // namespace

// --- CurrentSubDeviceGuard ---

CurrentSubDeviceGuard::CurrentSubDeviceGuard() : ptr_(nullptr, current_sub_device_guard_deleter) {}

CurrentSubDeviceGuard::CurrentSubDeviceGuard(CurrentSubDeviceGuardImpl* p) :
    ptr_(p, current_sub_device_guard_deleter) {}

// --- DeviceContext ---

DeviceContext::DeviceContext(tt::tt_metal::IDevice* device) :
    device_([device]() -> DeviceContext::DeviceVariant {
        TT_FATAL(device != nullptr, "DeviceContext: device must not be null");
        if (auto* m = dynamic_cast<tt::tt_metal::distributed::MeshDevice*>(device)) {
            return m;
        }
        return device;
    }()) {}

DeviceContext::DeviceContext(tt::tt_metal::distributed::MeshDevice* device) : device_(device) {
    TT_FATAL(device != nullptr, "DeviceContext: device must not be null");
}

tt::tt_metal::CoreCoord DeviceContext::get_compute_with_storage_grid_size() const {
    if (const auto* m = std::get_if<tt::tt_metal::distributed::MeshDevice*>(&device_)) {
        return get_compute_with_storage_grid_size_mesh(*m);
    }
    return std::get<tt::tt_metal::IDevice*>(device_)->compute_with_storage_grid_size();
}

tt::tt_metal::CoreCoord DeviceContext::get_grid_size() const {
    if (const auto* m = std::get_if<tt::tt_metal::distributed::MeshDevice*>(&device_)) {
        return get_compute_with_storage_grid_size_mesh(*m);
    }
    return std::get<tt::tt_metal::IDevice*>(device_)->grid_size();
}

tt::tt_metal::CoreCoord DeviceContext::get_logical_grid_size() const {
    if (const auto* m = std::get_if<tt::tt_metal::distributed::MeshDevice*>(&device_)) {
        return get_logical_grid_size_mesh(*m);
    }
    return std::get<tt::tt_metal::IDevice*>(device_)->logical_grid_size();
}

tt::tt_metal::CoreCoord DeviceContext::get_dram_grid_size() const {
    if (const auto* m = std::get_if<tt::tt_metal::distributed::MeshDevice*>(&device_)) {
        return get_dram_grid_size_mesh(*m);
    }
    return std::get<tt::tt_metal::IDevice*>(device_)->dram_grid_size();
}

tt::tt_metal::SubDeviceId DeviceContext::get_current_sub_device_id() const {
    if (const auto* m = std::get_if<tt::tt_metal::distributed::MeshDevice*>(&device_)) {
        return get_current_sub_device_id_impl(*m);
    }
    return tt::tt_metal::SubDeviceId{0};
}

tt::tt_metal::SubDeviceId DeviceContext::get_effective_sub_device_id(
    const std::optional<tt::tt_metal::SubDeviceId>& explicit_id) const {
    return explicit_id.value_or(get_current_sub_device_id());
}

bool DeviceContext::is_mesh_device() const noexcept {
    return std::holds_alternative<tt::tt_metal::distributed::MeshDevice*>(device_);
}

CurrentSubDeviceGuard DeviceContext::set_current_sub_device(tt::tt_metal::SubDeviceId sub_device_id) {
    if (auto* m = std::get_if<tt::tt_metal::distributed::MeshDevice*>(&device_)) {
        const auto& valid_ids = (*m)->get_sub_device_ids();
        const bool valid = std::find(valid_ids.begin(), valid_ids.end(), sub_device_id) != valid_ids.end();
        if (!valid) {
            std::string valid_str;
            for (size_t i = 0; i < valid_ids.size(); ++i) {
                if (i > 0) {
                    valid_str += ", ";
                }
                valid_str += std::to_string(static_cast<unsigned>(valid_ids[i].get()));
            }
            TT_FATAL(
                false,
                "DeviceContext: sub_device_id {} is not in the device's active sub-device list (load a sub-device "
                "manager and set_sub_device_stall_group first). Valid IDs: [{}]",
                static_cast<unsigned>(sub_device_id.get()),
                valid_str);
        }
        g_current_sub_device_stack.push_back({*m, sub_device_id});
        return CurrentSubDeviceGuard(new CurrentSubDeviceGuardImpl(*m, sub_device_id));
    }
    return CurrentSubDeviceGuard();
}

tt::tt_metal::IDevice* DeviceContext::raw_device() const {
    return std::visit([](auto* d) -> tt::tt_metal::IDevice* { return d; }, device_);
}

tt::tt_metal::distributed::MeshDevice* DeviceContext::raw_mesh_device() const noexcept {
    if (const auto* m = std::get_if<tt::tt_metal::distributed::MeshDevice*>(&device_)) {
        return *m;
    }
    return nullptr;
}

tt::tt_metal::IDevice* DeviceContext::get_reference_device() const {
    if (const auto* m = std::get_if<tt::tt_metal::distributed::MeshDevice*>(&device_)) {
        return get_reference_device_from_mesh(*m);
    }
    return std::get<tt::tt_metal::IDevice*>(device_);
}

DeviceContext device_context(const tt::tt_metal::Tensor& tensor) { return DeviceContext(tensor.device()); }

}  // namespace ttnn
