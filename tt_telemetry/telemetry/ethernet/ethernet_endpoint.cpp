// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <telemetry/ethernet/ethernet_endpoint.hpp>
#include <telemetry/ethernet/ethernet_helpers.hpp>

#include <boost/functional/hash.hpp>

/**************************************************************************************************
 EthernetEndpoint Class
**************************************************************************************************/

bool EthernetEndpoint::operator<(const EthernetEndpoint &other) const {
    if (chip != other.chip) {
        return chip < other.chip;
    } else if (channel != other.channel) {
        return channel < other.channel;
    }
    return ethernet_core < other.ethernet_core;
}

bool EthernetEndpoint::operator==(const EthernetEndpoint &other) const {
    return chip == other.chip && ethernet_core == other.ethernet_core && channel == other.channel;
}

std::vector<std::string> EthernetEndpoint::telemetry_path() const {
    auto path = chip.telemetry_path();
    path.push_back("channel" + std::to_string(channel));
    return path;
}

// Static because CoreCoord is defined outside of tt-telemetry and this should be included there
static std::ostream &operator<<(std::ostream &os, const tt::umd::CoreCoord &core) {
    os << core.str();
    return os;
}

std::ostream &operator<<(std::ostream &os, const EthernetEndpoint &ep) {
    os << "<Endpoint: " << ep.chip << ", Channel " << ep.channel << ", Core " << ep.ethernet_core << '>';
    return os;
}

// Required by Boost for hashing
namespace tt {
namespace umd {
size_t hash_value(const xy_pair& xy) {
    std::size_t seed = 0;
    boost::hash_combine(seed, xy.x);
    boost::hash_combine(seed, xy.y);
    return seed;
}
}  // namespace umd
}  // namespace tt

size_t hash_value(const EthernetEndpoint &ep) {
    size_t seed = 0;
    boost::hash_combine(seed, ep.chip);
    boost::hash_combine(seed, ep.ethernet_core);
    boost::hash_combine(seed, ep.channel);
    return seed;
}


/**************************************************************************************************
 Ethernet Endpoint Retrieval

 Functions to retrieve Ethernet endpoints and return them in a convenient, sorted form.
**************************************************************************************************/

std::map<ChipIdentifier, std::vector<EthernetEndpoint>> get_ethernet_endpoints_by_chip(
    const std::unique_ptr<tt::umd::Cluster>& cluster) {
    std::map<ChipIdentifier, std::vector<EthernetEndpoint>> ethernet_endpoints_by_chip;

    for (const auto& [chip_id, remote_chip_and_channel_by_channel] :
         cluster->get_cluster_description()->get_ethernet_connections()) {
        tt::umd::TTDevice* device = cluster->get_tt_device(chip_id);

        // Create a SOC descriptor just for the purpose of mapping Ethernet channel to core coordinates
        const tt_SocDescriptor& soc_desc = cluster->get_soc_descriptor(chip_id);

        ChipIdentifier chip = get_chip_identifier_from_umd_chip_id(device, chip_id);
        std::vector<EthernetEndpoint>& endpoints_this_chip = ethernet_endpoints_by_chip[chip];

        for (const auto& [channel, remote_chip_and_channel] : remote_chip_and_channel_by_channel) {
            // Construct EthernetEndpoint from its components
            tt::umd::CoreCoord ethernet_core = soc_desc.get_eth_core_for_channel(channel, tt::umd::CoordSystem::LOGICAL);
            EthernetEndpoint endpoint{.chip = chip, .ethernet_core = ethernet_core, .channel = channel};

            // Add to list of endpoints for current chip
            endpoints_this_chip.push_back(endpoint);
        }
    }

    return ethernet_endpoints_by_chip;
}
