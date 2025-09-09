// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/assert.hpp>

#include <telemetry/ethernet/ethernet_helpers.hpp>

static auto make_ordered_ethernet_connections(const auto &unordered_connections) {
    std::map<
        tt::umd::chip_id_t,
        std::map<
            tt::umd::ethernet_channel_t,
            std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>
        >
    > ordered_connections;

    for (const auto& [chip_id, channel_map] : unordered_connections) {
        for (const auto& [channel, connection_tuple] : channel_map) {
            ordered_connections[chip_id][channel] = connection_tuple;
        }
    }

    return ordered_connections;
}

std::map<
    tt::umd::chip_id_t,
    std::map<tt::umd::ethernet_channel_t, std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>>>
get_ordered_ethernet_connections(const std::unique_ptr<tt::umd::Cluster>& cluster) {
    return make_ordered_ethernet_connections(cluster->get_cluster_description()->get_ethernet_connections());
}

bool is_ethernet_endpoint_up(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const EthernetEndpoint& endpoint,
    uint32_t link_up_addr,
    bool force_refresh_link_status) {
    tt::umd::TTDevice* device = cluster->get_tt_device(endpoint.chip.id);

    uint32_t link_up_value = 0;
    tt::umd::CoreCoord ethernet_core = tt::umd::CoreCoord(
        endpoint.ethernet_core.x, endpoint.ethernet_core.y, tt::umd::CoreType::ETH, tt::umd::CoordSystem::LOGICAL);
    cluster->read_from_device(&link_up_value, endpoint.chip.id, ethernet_core, link_up_addr, sizeof(uint32_t));

    if (cluster->get_tt_device(endpoint.chip.id)->get_arch() == tt::ARCH::WORMHOLE_B0) {
        return link_up_value == 6;  // see eth_fw_api.h
    } else if (cluster->get_tt_device(endpoint.chip.id)->get_arch() == tt::ARCH::BLACKHOLE) {
        return link_up_value == 1;
    }

    TT_ASSERT(false, "Unsupported architecture for chip {}", endpoint.chip);
    return false;
}
