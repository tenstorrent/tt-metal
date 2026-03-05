// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/device.hpp"
#include <memory>
#include <optional>

namespace tt::tt_metal {
class IDevice;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace ttnn::execution_context {

struct CurrentSubDeviceGuardImpl;

/** RAII guard for set_current_sub_device. Restores the previous current sub-device when destroyed. */
class CurrentSubDeviceGuard {
public:
    CurrentSubDeviceGuard();
    CurrentSubDeviceGuard(const CurrentSubDeviceGuard&) = delete;
    CurrentSubDeviceGuard& operator=(const CurrentSubDeviceGuard&) = delete;
    CurrentSubDeviceGuard(CurrentSubDeviceGuard&&) noexcept = default;
    CurrentSubDeviceGuard& operator=(CurrentSubDeviceGuard&&) noexcept = default;

private:
    explicit CurrentSubDeviceGuard(CurrentSubDeviceGuardImpl* p);
    std::unique_ptr<CurrentSubDeviceGuardImpl, void (*)(CurrentSubDeviceGuardImpl*)> ptr_;
    friend CurrentSubDeviceGuard set_current_sub_device(
        tt::tt_metal::distributed::MeshDevice* device, tt::tt_metal::SubDeviceId sub_device_id);
};

/**
 * Returns the current sub-device ID for the given device in this thread.
 * For MeshDevice, uses the thread-local execution context if set; otherwise
 * returns the first sub-device (get_sub_device_ids().at(0)).
 * For non-mesh IDevice, always returns the first sub-device.
 */
tt::tt_metal::SubDeviceId get_current_sub_device_id(tt::tt_metal::IDevice* device);

/**
 * Overload for MeshDevice* for call sites that have the concrete type.
 */
tt::tt_metal::SubDeviceId get_current_sub_device_id(tt::tt_metal::distributed::MeshDevice* device);

/**
 * Sets the current sub-device for the given MeshDevice in this thread.
 * Returns a CurrentSubDeviceGuard that restores the previous current sub-device on scope exit.
 * Only applies to MeshDevice; for other devices this is a no-op (returns an empty guard).
 */
CurrentSubDeviceGuard set_current_sub_device(
    tt::tt_metal::distributed::MeshDevice* device, tt::tt_metal::SubDeviceId sub_device_id);

/**
 * Returns the effective sub-device ID: explicit_id if present, otherwise
 * the current sub-device from the execution context for the device.
 * Use this in ops that accept an optional sub_device_id parameter.
 */
inline tt::tt_metal::SubDeviceId get_effective_sub_device_id(
    tt::tt_metal::IDevice* device, const std::optional<tt::tt_metal::SubDeviceId>& explicit_id) {
    return explicit_id.value_or(get_current_sub_device_id(device));
}

/**
 * Overload for MeshDevice*.
 */
inline tt::tt_metal::SubDeviceId get_effective_sub_device_id(
    tt::tt_metal::distributed::MeshDevice* device, const std::optional<tt::tt_metal::SubDeviceId>& explicit_id) {
    return explicit_id.value_or(get_current_sub_device_id(device));
}

}  // namespace ttnn::execution_context
