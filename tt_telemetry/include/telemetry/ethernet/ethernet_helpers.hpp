#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/ethernet/ethernet_helpers.hpp
 *
 * Misc. helper functions.
 */

#include <map>
#include <unordered_map>

#include <third_party/umd/device/api/umd/device/cluster.hpp>

namespace tt::umd {
class Cluster;
}

std::map<tt::ChipId, std::map<tt::EthernetChannel, std::tuple<tt::ChipId, tt::EthernetChannel>>>
get_ordered_ethernet_connections(const std::unique_ptr<tt::umd::Cluster>& cluster);
std::map<tt::ChipId, std::map<tt::EthernetChannel, std::tuple<uint64_t, tt::EthernetChannel>>>
get_ordered_ethernet_connections_to_remote_devices(const std::unique_ptr<tt::umd::Cluster>& cluster);
bool is_ethernet_endpoint_up(
    const std::unique_ptr<tt::umd::Cluster>& cluster,
    tt::ChipId chip_id,
    uint32_t channel,
    uint32_t link_up_addr,
    bool force_refresh_link_status);
