// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt_stl/assert.hpp>

#include <telemetry/ethernet/ethernet_helpers.hpp>

template <typename ChipId>
static std::map<tt::ChipId, std::map<tt::EthernetChannel, std::tuple<ChipId, tt::EthernetChannel>>>
make_ordered_ethernet_connections(
    const std::unordered_map<
        tt::ChipId,
        std::unordered_map<tt::EthernetChannel, std::tuple<ChipId, tt::EthernetChannel>>>& unordered_connections) {
    std::map<tt::ChipId, std::map<tt::EthernetChannel, std::tuple<ChipId, tt::EthernetChannel>>> ordered_connections;

    for (const auto& [chip_id, channel_map] : unordered_connections) {
        for (const auto& [channel, connection_tuple] : channel_map) {
            ordered_connections[chip_id][channel] = connection_tuple;
        }
    }

    return ordered_connections;
}

std::map<tt::ChipId, std::map<tt::EthernetChannel, std::tuple<tt::ChipId, tt::EthernetChannel>>>
get_ordered_ethernet_connections(const std::unique_ptr<tt::umd::Cluster>& cluster) {
    return make_ordered_ethernet_connections(cluster->get_cluster_description()->get_ethernet_connections());
}

std::map<tt::ChipId, std::map<tt::EthernetChannel, std::tuple<uint64_t, tt::EthernetChannel>>>
get_ordered_ethernet_connections_to_remote_devices(const std::unique_ptr<tt::umd::Cluster>& cluster) {
    return make_ordered_ethernet_connections(
        cluster->get_cluster_description()->get_ethernet_connections_to_remote_devices());
}

bool is_ethernet_endpoint_up(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    ChipId chip_id,
    uint32_t channel,
    uint32_t link_up_addr,
    bool force_refresh_link_status) {
    const SocDescriptor& soc_desc = cluster->get_soc_descriptor(chip_id);
    tt::umd::CoreCoord ethernet_core = soc_desc.get_eth_core_for_channel(channel, CoordSystem::LOGICAL);

    uint32_t link_up_value = 0;
    cluster->read_from_device(&link_up_value, chip_id, ethernet_core, link_up_addr, sizeof(uint32_t));

    if (cluster->get_tt_device(chip_id)->get_arch() == tt::ARCH::WORMHOLE_B0) {
        return link_up_value == 6;  // see eth_fw_api.h
    } else if (cluster->get_tt_device(chip_id)->get_arch() == tt::ARCH::BLACKHOLE) {
        return link_up_value == 1;
    }

    TT_ASSERT(false, "Unsupported architecture for chip {}", chip_id);
    return false;
}
