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

#include <third_party/umd/device/api/umd/device/cluster.hpp>

static inline uint16_t get_bus_id(const std::unique_ptr<tt::umd::Cluster>& cluster, chip_id_t chip) {
    return cluster->get_chip(chip)->get_tt_device()->get_pci_device()->get_device_info().pci_bus;
}
