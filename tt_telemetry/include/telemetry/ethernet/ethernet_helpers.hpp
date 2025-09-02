#pragma once

/*
 * telemetry/ethernet/ethernet_helpers.hpp
 *
 * Misc. helper functions.
 */

#include <map>
#include <unordered_map>

#include <third_party/umd/device/api/umd/device/cluster.h>

#include <telemetry/ethernet/ethernet_endpoint.hpp>

std::map<
    tt::umd::chip_id_t,
    std::map<tt::umd::ethernet_channel_t, std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>>>
get_ordered_ethernet_connections(const std::unique_ptr<tt::umd::Cluster>& cluster);
bool is_ethernet_endpoint_up(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    const EthernetEndpoint& endpoint,
    bool force_refresh_link_status = false);
