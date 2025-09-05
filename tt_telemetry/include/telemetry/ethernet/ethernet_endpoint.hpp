#pragma once

// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

/*
 * telemetry/ethernet/ethernet_endpoint.hpp
 *
 * Describes Ethernet link endpoints (single side).
 */

 #include <map>
 #include <ostream>
 #include <vector>

 #include <telemetry/ethernet/chip_identifier.hpp>

 struct EthernetEndpoint {
    ChipIdentifier chip;
    tt::umd::CoreCoord ethernet_core;
    tt::umd::ethernet_channel_t channel;

    bool operator<(const EthernetEndpoint &other) const;
    bool operator==(const EthernetEndpoint &other) const;

    std::vector<std::string> telemetry_path() const;
};

std::ostream &operator<<(std::ostream &os, const EthernetEndpoint &ep);

size_t hash_value(const EthernetEndpoint &ep);

namespace tt {
namespace umd {
size_t hash_value(const xy_pair& xy);
}
}  // namespace tt

namespace std {
    template<>
    struct hash<EthernetEndpoint> {
        size_t operator()(const EthernetEndpoint& ep) const noexcept {
            return hash_value(ep);
        }
    };
}

std::map<ChipIdentifier, std::vector<EthernetEndpoint>> get_ethernet_endpoints_by_chip(
    const std::unique_ptr<tt::umd::Cluster>& cluster);
