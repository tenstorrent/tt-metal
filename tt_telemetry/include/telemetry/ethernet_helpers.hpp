#pragma once

/*
 * telemetry/ethernet_helpers.hpp
 *
 * Misc. helper functions.
 */

#include <map>
#include <unordered_map>

#include <telemetry/ethernet_endpoint.hpp>

std::map<
    tt::umd::chip_id_t, 
    std::map<
        tt::umd::ethernet_channel_t, 
        std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>
    >
> get_ordered_ethernet_connections(const tt::Cluster &cluster);
std::unordered_map<tt::umd::ethernet_channel_t, CoreCoord> map_ethernet_channel_to_core_coord(const tt::Cluster &cluster, tt::umd::chip_id_t chip_id); 