#pragma once

/*
 * telemetry/ethernet/ethernet_endpoint.hpp
 *
 * Describes Ethernet link endpoints (single side).
 */

 #include <map>
 #include <ostream>
 #include <vector>

 #include <telemetry/ethernet/chip_identifier.hpp>
 #include <tt-metalium/core_coord.hpp>
 #include <tt-metalium/cluster.hpp>

 struct EthernetEndpoint {
    ChipIdentifier chip;
    CoreCoord ethernet_core;
    tt::umd::ethernet_channel_t channel;

    bool operator<(const EthernetEndpoint &other) const;
    bool operator==(const EthernetEndpoint &other) const;

    std::vector<std::string> telemetry_path() const;
};

std::ostream &operator<<(std::ostream &os, const EthernetEndpoint &ep);

size_t hash_value(const EthernetEndpoint &ep);

namespace tt { 
    namespace umd {
       size_t hash_value(const xy_pair &xy);
    }
}

namespace std {
    template<>
    struct hash<EthernetEndpoint> {
        size_t operator()(const EthernetEndpoint& ep) const noexcept {
            return hash_value(ep);
        }
    };
}

std::map<ChipIdentifier, std::vector<EthernetEndpoint>> get_ethernet_endpoints_by_chip(const tt::Cluster &cluster);