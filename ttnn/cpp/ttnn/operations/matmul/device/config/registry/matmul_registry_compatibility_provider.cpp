// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/config/registry/matmul_registry_compatibility.hpp"

#include <tt-metalium/cluster.hpp>
#include <tt-metalium/mesh_device.hpp>

#include "matmul_registry_build_attestation.hpp"
#include "tt_metal/impl/context/context_types.hpp"
#include "tt_metal/impl/context/metal_context.hpp"
#include "tt_metal/llrt/tt_cluster.hpp"
#include "ttnn/operations/matmul/device/config/registry/checked_in/matmul_registry_data.hpp"

namespace ttnn::operations::matmul::registry {

BuildCompatibilityDigests compiled_build_compatibility() noexcept {
    return {
        .semantic_source_sha256 = generated_build::kActualSemanticSourceSha256,
        .build_identity_sha256 = generated_build::kActualBuildIdentitySha256};
}

DeviceCompatibilityResult production_device_compatibility(
    const tt::tt_metal::distributed::MeshDevice& device) noexcept {
    try {
        if (!device.is_initialized()) {
            return {.status = DeviceCompatibilityStatus::DeviceUninitialized};
        }
        if (device.is_remote_only()) {
            return {.status = DeviceCompatibilityStatus::RemoteDevice};
        }
        if (device.num_devices() != 1 || device.num_rows() != 1 || device.num_cols() != 1) {
            return {.status = DeviceCompatibilityStatus::NotOneChip};
        }

        auto& context = tt::tt_metal::MetalContext::instance(tt::tt_metal::extract_context_id(&device));
        const auto& cluster = context.get_cluster();
        if (cluster.arch() != tt::ARCH::BLACKHOLE) {
            return {.status = DeviceCompatibilityStatus::UnsupportedArchitecture};
        }
        const auto device_ids = device.get_device_ids();
        if (device_ids.size() != 1 || device_ids.front() < 0) {
            return {.status = DeviceCompatibilityStatus::NotOneChip};
        }
        const auto chip_id = static_cast<tt::ChipId>(device_ids.front());

        AttestationBoardClass board_class;
        switch (cluster.get_board_type(chip_id)) {
            case tt::BoardType::P100: board_class = AttestationBoardClass::BlackholeP100; break;
            case tt::BoardType::P150: board_class = AttestationBoardClass::BlackholeP150; break;
            case tt::BoardType::UBB_BLACKHOLE: board_class = AttestationBoardClass::BlackholeGalaxy; break;
            default: return {.status = DeviceCompatibilityStatus::UnsupportedBoard};
        }

        AttestationClusterClass cluster_class;
        switch (cluster.get_cluster_type()) {
            case tt::tt_metal::ClusterType::P100: cluster_class = AttestationClusterClass::BlackholeP100; break;
            case tt::tt_metal::ClusterType::P150: cluster_class = AttestationClusterClass::BlackholeP150; break;
            case tt::tt_metal::ClusterType::P150_X2: cluster_class = AttestationClusterClass::BlackholeP150X2; break;
            case tt::tt_metal::ClusterType::P150_X4: cluster_class = AttestationClusterClass::BlackholeP150X4; break;
            case tt::tt_metal::ClusterType::P150_X8: cluster_class = AttestationClusterClass::BlackholeP150X8; break;
            case tt::tt_metal::ClusterType::BLACKHOLE_GALAXY:
                cluster_class = AttestationClusterClass::BlackholeGalaxy;
                break;
            default: return {.status = DeviceCompatibilityStatus::UnsupportedCluster};
        }

        const auto firmware_bundle = cluster.get_cluster_desc()->get_cluster_firmware_bundle_version();
        const auto ethernet_firmware = cluster.get_ethernet_firmware_version();
        const auto compute_grid = device.compute_with_storage_grid_size();
        const auto physical_grid = device.grid_size();
        const auto logical_grid = device.logical_grid_size();
        const auto dram_grid = device.dram_grid_size();
        return derive_device_compatibility(DeviceCompatibilityFacts{
            .architecture = static_cast<std::uint32_t>(cluster.arch()),
            .board_class = board_class,
            .cluster_class = cluster_class,
            .device_initialized = true,
            .remote_only = false,
            .active_sub_device_manager_is_default =
                device.get_active_sub_device_manager_id() == device.get_default_sub_device_manager_id(),
            .device_count = static_cast<std::uint32_t>(device.num_devices()),
            .mesh_rows = static_cast<std::uint32_t>(device.num_rows()),
            .mesh_cols = static_cast<std::uint32_t>(device.num_cols()),
            .compute_grid_x = compute_grid.x,
            .compute_grid_y = compute_grid.y,
            .physical_grid_x = physical_grid.x,
            .physical_grid_y = physical_grid.y,
            .logical_grid_x = logical_grid.x,
            .logical_grid_y = logical_grid.y,
            .dram_grid_x = dram_grid.x,
            .dram_grid_y = dram_grid.y,
            .tensix_harvesting_mask = cluster.get_harvesting_mask(chip_id),
            .num_hw_cqs = device.num_hw_cqs(),
            .num_dram_channels = static_cast<std::uint32_t>(device.num_dram_channels()),
            .l1_size_per_core = device.l1_size_per_core(),
            .dram_size_per_channel = device.dram_size_per_channel(),
            .firmware_bundle_present = firmware_bundle.has_value(),
            .firmware_bundle_major = firmware_bundle.has_value() ? firmware_bundle->major : 0,
            .firmware_bundle_minor = firmware_bundle.has_value() ? firmware_bundle->minor : 0,
            .firmware_bundle_patch = firmware_bundle.has_value() ? firmware_bundle->patch : 0,
            .ethernet_firmware_present = ethernet_firmware.has_value(),
            .ethernet_firmware_major = ethernet_firmware.has_value() ? ethernet_firmware->major : 0,
            .ethernet_firmware_minor = ethernet_firmware.has_value() ? ethernet_firmware->minor : 0,
            .ethernet_firmware_patch = ethernet_firmware.has_value() ? ethernet_firmware->patch : 0});
    } catch (...) {
        return {.status = DeviceCompatibilityStatus::QueryFailed};
    }
}

CompatibilityStatus production_registry_compatibility(const tt::tt_metal::distributed::MeshDevice& device) noexcept {
    return validate_registry_compatibility(
        generated::metadata(), compiled_build_compatibility(), production_device_compatibility(device));
}

}  // namespace ttnn::operations::matmul::registry
