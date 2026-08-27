// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>
#include <string_view>

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::matmul::registry {

enum class AttestationArchitecture : std::uint8_t { Blackhole = 1 };
enum class AttestationBoardClass : std::uint32_t {
    BlackholeP100 = 1,
    BlackholeP150 = 2,
    BlackholeGalaxy = 3,
};
enum class AttestationClusterClass : std::uint8_t {
    BlackholeP100 = 1,
    BlackholeP150 = 2,
    BlackholeP150X2 = 3,
    BlackholeP150X4 = 4,
    BlackholeP150X8 = 5,
    BlackholeGalaxy = 6,
};

enum class DeviceAttestationStatus : std::uint8_t {
    Success,
    QueryFailed,
    DeviceUninitialized,
    RemoteDevice,
    NotOneChip,
    ActiveSubDeviceManager,
    UnsupportedArchitecture,
    UnsupportedBoard,
    UnsupportedCluster,
    BoardClusterMismatch,
    FirmwareUnavailable,
    InvalidCapability,
};

// Canonical facts are intentionally free of process-local device/chip IDs. A
// measured certificate is portable across physical chips only when every
// capability and topology fact below is equal.
struct DeviceAttestationFacts {
    AttestationArchitecture architecture{};
    AttestationBoardClass board_class{};
    AttestationClusterClass cluster_class{};
    bool device_initialized = false;
    bool remote_only = false;
    bool active_sub_device_manager_is_default = false;
    std::uint32_t device_count = 0;
    std::uint32_t mesh_rows = 0;
    std::uint32_t mesh_cols = 0;
    std::uint32_t system_mesh_id = 0;
    std::uint32_t compute_grid_x = 0;
    std::uint32_t compute_grid_y = 0;
    std::uint32_t physical_grid_x = 0;
    std::uint32_t physical_grid_y = 0;
    std::uint32_t logical_grid_x = 0;
    std::uint32_t logical_grid_y = 0;
    std::uint32_t dram_grid_x = 0;
    std::uint32_t dram_grid_y = 0;
    std::uint32_t tensix_harvesting_mask = 0;
    std::uint32_t num_hw_cqs = 0;
    std::uint32_t num_dram_channels = 0;
    std::uint32_t l1_size_per_core = 0;
    std::uint64_t dram_size_per_channel = 0;
    bool firmware_bundle_present = false;
    std::uint32_t firmware_bundle_major = 0;
    std::uint32_t firmware_bundle_minor = 0;
    std::uint32_t firmware_bundle_patch = 0;
    bool ethernet_firmware_present = false;
    std::uint32_t ethernet_firmware_major = 0;
    std::uint32_t ethernet_firmware_minor = 0;
    std::uint32_t ethernet_firmware_patch = 0;
};

struct DeviceAttestation {
    std::uint32_t architecture = 0;
    std::uint32_t board_capability_class = 0;
    compact::Sha256 topology_sha256{};
    compact::Sha256 runtime_capability_sha256{};
};

struct DeviceAttestationResult {
    DeviceAttestationStatus status = DeviceAttestationStatus::QueryFailed;
    DeviceAttestation attestation{};
};

compact::Sha256 registry_sha256(std::span<const std::uint8_t> bytes) noexcept;
std::string_view device_attestation_status_name(DeviceAttestationStatus status) noexcept;
DeviceAttestationResult derive_device_attestation(const DeviceAttestationFacts& facts) noexcept;

using DeviceAttestationProvider = DeviceAttestationResult (*)(const tt::tt_metal::distributed::MeshDevice&) noexcept;

DeviceAttestationResult production_device_attestation(const tt::tt_metal::distributed::MeshDevice& device) noexcept;

inline DeviceAttestationResult query_device_attestation(
    const tt::tt_metal::distributed::MeshDevice& device,
    const DeviceAttestationProvider provider = &production_device_attestation) noexcept {
    return provider != nullptr ? provider(device)
                               : DeviceAttestationResult{.status = DeviceAttestationStatus::QueryFailed};
}

}  // namespace ttnn::operations::matmul::registry
