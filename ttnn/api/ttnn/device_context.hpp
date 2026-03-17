// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include <memory>
#include <optional>
#include <variant>

namespace tt::tt_metal {
class Tensor;
namespace distributed {
class MeshDevice;
}
}  // namespace tt::tt_metal

namespace ttnn {

/**
 * Single entry point for device and execution context in TTNN.
 *
 * Use DeviceContext for all grid size, sub-device, and current-context operations.
 * Construct from a device pointer or via device_context(tensor). Do not use raw
 * device->grid_size() or similar; use the get_* methods on this class.
 *
 * Mesh-related methods (set_current_sub_device, raw_mesh_device, is_mesh_device) are
 * exception-free: they never throw; they return nullptr, no-op guard, or false when
 * the device is not a MeshDevice.
 */

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
    friend class DeviceContext;
};

class DeviceContext {
public:
    explicit DeviceContext(tt::tt_metal::IDevice* device);
    explicit DeviceContext(tt::tt_metal::distributed::MeshDevice* device);
    /** Construct from a tensor; device is obtained from the tensor internally. */
    explicit DeviceContext(const tt::tt_metal::Tensor& tensor);

    tt::tt_metal::CoreCoord get_compute_with_storage_grid_size() const;
    tt::tt_metal::CoreCoord get_grid_size() const;
    tt::tt_metal::CoreCoord get_logical_grid_size() const;
    tt::tt_metal::CoreCoord get_dram_grid_size() const;

    tt::tt_metal::SubDeviceId get_current_sub_device_id() const;
    tt::tt_metal::SubDeviceId get_effective_sub_device_id(
        const std::optional<tt::tt_metal::SubDeviceId>& explicit_id) const;

    /** Worker core range set for the effective sub-device (explicit_id if provided, else context-driven). */
    tt::tt_metal::CoreRangeSet get_worker_cores(
        tt::tt_metal::HalProgrammableCoreType core_type,
        std::optional<tt::tt_metal::SubDeviceId> explicit_sub_device_id = std::nullopt) const;

    bool is_mesh_device() const noexcept;
    CurrentSubDeviceGuard set_current_sub_device(tt::tt_metal::SubDeviceId sub_device_id);

    tt::tt_metal::IDevice* raw_device() const;
    tt::tt_metal::distributed::MeshDevice* raw_mesh_device() const noexcept;

    /** For a MeshDevice, returns the first physical device; for a single device, returns raw_device(). */
    tt::tt_metal::IDevice* get_reference_device() const;

private:
    using DeviceVariant = std::variant<tt::tt_metal::IDevice*, tt::tt_metal::distributed::MeshDevice*>;
    DeviceVariant device_;
};

DeviceContext device_context(const tt::tt_metal::Tensor& tensor);

}  // namespace ttnn
