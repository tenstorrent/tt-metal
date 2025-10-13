// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

/*
 * topology/topology.hpp
 *
 * Utilities for understanding topology and mapping between different identifiers.
 */

#include <memory>
#include <unordered_map>

#include <tt_metal/fabric/physical_system_descriptor.hpp>

namespace tt::umd {
class Cluster;
}

// Hash function for std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID>
struct ASICLocationAndTrayIDHash {
    std::size_t operator()(const std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID>& p) const {
        auto h1 = std::hash<uint32_t>{}(*p.first);
        auto h2 = std::hash<uint32_t>{}(*p.second);
        constexpr std::size_t hash_combine_prime = 0x9e3779b9;
        return h1 ^ (h2 + hash_combine_prime + (h1 << 6) + (h1 >> 2));
    }
};

// TODO: Open to a better name for this
class TopologyHelper {
public:
    TopologyHelper(
        const std::unique_ptr<tt::umd::Cluster>& cluster,
        const std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor>& psd);
    std::optional<ChipId> get_local_chip_id_for_asic_location_and_tray(
        tt::tt_metal::ASICLocation asic_location, tt::tt_metal::TrayID tray_id);
    std::optional<tt::tt_metal::ASICDescriptor> get_asic_descriptor_for_local_chip(ChipId chip_id);

    const std::string my_host_name;

private:
    std::unordered_map<std::pair<tt::tt_metal::ASICLocation, tt::tt_metal::TrayID>, ChipId, ASICLocationAndTrayIDHash>
        asic_location_and_tray_id_to_local_chip_id_;
    std::unordered_map<ChipId, tt::tt_metal::ASICDescriptor> local_chip_id_to_asic_descriptor_;
};
