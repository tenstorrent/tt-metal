// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <third_party/umd/device/api/umd/device/cluster.hpp>
#include <tt_stl/assert.hpp>

#include <topology/topology.hpp>

template <typename Key, typename Value, typename KeyHash>
static std::unordered_map<Value, Key> invert_map(const std::unordered_map<Key, Value, KeyHash>& map) {
    std::unordered_map<Value, Key> inverse_map;
    for (const auto& [key, value] : map) {
        inverse_map.insert({value, key});
    }
    return inverse_map;
}

TopologyHelper::TopologyHelper(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const std::unique_ptr<tt::tt_metal::PhysicalSystemDescriptor>& psd) :
    my_host_name(psd->my_host_name()) {
    // Get mapping of chip ID <-> ASIC unique ID. This is only valid for the local host!
    const std::unordered_map<tt::ChipId, uint64_t>& chip_id_to_unique_id =
        cluster->get_cluster_description()->get_chip_unique_ids();
    std::unordered_map<uint64_t, tt::ChipId> unique_id_to_chip_id = invert_map(chip_id_to_unique_id);

    // Produce the following maps (all valid only for the local host because chip ID is not
    // globally unique):
    //  (asic_location,tray_id) -> chip_id
    //  chip_id -> (asic_location,tray_id)
    //  chip_id -> ASICDescriptor
    for (auto [unique_id, asic_descriptor] : psd->get_asic_descriptors()) {
        auto it = unique_id_to_chip_id.find(*unique_id);
        if (it == unique_id_to_chip_id.end()) {
            continue;
        }
        tt::ChipId chip_id = it->second;

        auto key = std::make_pair(asic_descriptor.asic_location, asic_descriptor.tray_id);
        TT_FATAL(
            asic_location_and_tray_id_to_local_chip_id_.count(key) == 0,
            "Duplicate key (asic_location={}, tray_id={}) found in mapping",
            *asic_descriptor.asic_location,
            *asic_descriptor.tray_id);
        asic_location_and_tray_id_to_local_chip_id_.insert({key, chip_id});
        local_chip_id_to_asic_descriptor_.insert({chip_id, asic_descriptor});
    }
}

std::optional<tt::ChipId> TopologyHelper::get_local_chip_id_for_asic_location_and_tray(
    tt::tt_metal::ASICLocation asic_location, tt::tt_metal::TrayID tray_id) {
    auto it = asic_location_and_tray_id_to_local_chip_id_.find({asic_location, tray_id});
    if (it == asic_location_and_tray_id_to_local_chip_id_.end()) {
        return std::nullopt;
    }
    return it->second;
}

std::optional<tt::tt_metal::ASICDescriptor> TopologyHelper::get_asic_descriptor_for_local_chip(tt::ChipId chip_id) {
    auto it = local_chip_id_to_asic_descriptor_.find(chip_id);
    if (it == local_chip_id_to_asic_descriptor_.end()) {
        return std::nullopt;
    }
    return it->second;
}
