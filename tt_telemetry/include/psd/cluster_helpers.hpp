// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * psd/cluster_helpers.hpp
 *
 * Helper functions to replace functionality that was present in LLRT's Cluster classd. We now use
 * tt::umd::Cluster directly.
 */

#include <map>
#include <unordered_map>
#include <vector>

#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <tt-metalium/assert.hpp>

static inline uint16_t get_bus_id(const std::unique_ptr<tt::umd::Cluster>& cluster, chip_id_t chip) {
    return cluster->get_chip(chip)->get_tt_device()->get_pci_device()->get_device_info().pci_bus;
}

static inline const std::unordered_set<chip_id_t>& get_devices_controlled_by_mmio_device(
    const std::unique_ptr<tt::umd::Cluster>& cluster, chip_id_t mmio_device_id) {
    const auto& cluster_descriptor = cluster->get_cluster_description();
    TT_ASSERT(
        cluster_descriptor->get_chips_grouped_by_closest_mmio().count(mmio_device_id),
        "Expected device {} to be an MMIO device!",
        mmio_device_id);
    return cluster_descriptor->get_chips_grouped_by_closest_mmio().at(mmio_device_id);
}

std::map<chip_id_t, std::vector<std::vector<chip_id_t>>> discover_tunnels_from_mmio_device(
    const std::unique_ptr<tt::umd::Cluster>& cluster);
