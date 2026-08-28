// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <span>

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_descriptor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::matmul::registry {

inline constexpr std::uint16_t kRegistryCompatibilitySchemaVersion = 1;

struct BuildCompatibilityDigests {
    compact::Sha256 semantic_source_sha256{};
    compact::Sha256 build_identity_sha256{};
};

enum class DeviceCompatibilityStatus : std::uint8_t {
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

enum class AttestationBoardClass : std::uint32_t {
    BlackholeP100 = 1,
    BlackholeP150 = 2,
    BlackholeGalaxy = 3,
};

enum class AttestationClusterClass : std::uint32_t {
    BlackholeP100 = 1,
    BlackholeP150 = 2,
    BlackholeP150X2 = 3,
    BlackholeP150X4 = 4,
    BlackholeP150X8 = 5,
    BlackholeGalaxy = 6,
};

struct DeviceCompatibilityFacts {
    std::uint32_t architecture = 0;
    AttestationBoardClass board_class{};
    AttestationClusterClass cluster_class{};
    bool device_initialized = false;
    bool remote_only = false;
    bool active_sub_device_manager_is_default = false;
    std::uint32_t device_count = 0;
    std::uint32_t mesh_rows = 0;
    std::uint32_t mesh_cols = 0;
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

struct DeviceCompatibilityResult {
    DeviceCompatibilityStatus status = DeviceCompatibilityStatus::QueryFailed;
    compact::Sha256 runtime_capability_sha256{};
};

enum class CompatibilityStatus : std::uint8_t {
    Compatible,
    CompatibilityUnavailable,
    DeviceAttestationUnavailable,
    DeviceQueryFailed,
    DeviceUninitialized,
    RemoteDevice,
    NotOneChipDevice,
    ActiveSubDeviceManager,
    UnsupportedArchitecture,
    UnsupportedBoard,
    UnsupportedCluster,
    BoardClusterMismatch,
    FirmwareUnavailable,
    InvalidDeviceCapability,
    SemanticSourceMismatch,
    BuildIdentityMismatch,
    RuntimeCapabilityMismatch,
};

compact::Sha256 registry_sha256(std::span<const std::uint8_t> bytes) noexcept;
DeviceCompatibilityResult derive_device_compatibility(const DeviceCompatibilityFacts& facts) noexcept;
CompatibilityStatus validate_registry_compatibility(
    const compact::TableMetadata& expected,
    const BuildCompatibilityDigests& actual_build,
    const DeviceCompatibilityResult& actual_device) noexcept;

BuildCompatibilityDigests compiled_build_compatibility() noexcept;
DeviceCompatibilityResult production_device_compatibility(const tt::tt_metal::distributed::MeshDevice& device) noexcept;
CompatibilityStatus production_registry_compatibility(const tt::tt_metal::distributed::MeshDevice& device) noexcept;

using RegistryCompatibilityProvider = CompatibilityStatus (*)(const tt::tt_metal::distributed::MeshDevice&) noexcept;

}  // namespace ttnn::operations::matmul::registry
