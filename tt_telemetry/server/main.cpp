/*
 * main.cpp
 * tt-telemetry main server app.
 * 
 * TODO
 * ----
 * - Need to handle other cluster types (including N300, etc., which have most of their Ethernet
 *   cores unused), ensuring we don't mark legitimately unused connections as "down".
 * - Simple REST exporter and web GUI.  
 */

#include <iostream>
#include <optional>

#include <boost/functional/hash.hpp>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"


/**************************************************************************************************
 Chip Identification

 Enriched chip identification, including information about tray and board position, where
 applicable (e.g., on Galaxy machines). The basic tt::umd::chip_id_t is always the fallback.
**************************************************************************************************/

struct GalaxyUbbIdentifier {
    std::uint32_t tray_id;
    std::uint32_t asic_id;

    bool operator<(const GalaxyUbbIdentifier &other) const {
        if (tray_id != other.tray_id) {
            return tray_id < other.tray_id;
        }
        return asic_id < other.asic_id;
    }

    bool operator==(const GalaxyUbbIdentifier &other) const {
        return tray_id == other.tray_id && asic_id == other.asic_id;
    }
};

struct ChipIdentifier {
    tt::umd::chip_id_t id;
    std::optional<GalaxyUbbIdentifier> galaxy_ubb;

    bool operator<(const ChipIdentifier &other) const {
        if (id != other.id) {
            return id < other.id;
        }
        return galaxy_ubb < other.galaxy_ubb;
    }

    bool operator==(const ChipIdentifier &other) const {
        return id == other.id && galaxy_ubb == other.galaxy_ubb;
    }
};

static std::ostream &operator<<(std::ostream &os, const ChipIdentifier &chip) {
    if (chip.galaxy_ubb.has_value()) {
        os << "Tray " << chip.galaxy_ubb.value().tray_id << ", N" << chip.galaxy_ubb.value().asic_id << " (Chip " << chip.id << ')';
    } else {
        os << "Chip " << chip.id;
    }
    return os;
}

// For Boost compatibility
static size_t hash_value(const GalaxyUbbIdentifier &g) {
    std::size_t seed = 0;
    boost::hash_combine(seed, g.tray_id);
    boost::hash_combine(seed, g.asic_id);
    return seed;
}

static size_t hash_value(const ChipIdentifier &c) {
    std::size_t seed = 0;
    boost::hash_combine(seed, c.id);
    boost::hash_combine(seed, c.galaxy_ubb);
    return seed;
}

namespace std {
    template<>
    struct hash<GalaxyUbbIdentifier> {
        std::size_t operator()(const GalaxyUbbIdentifier &g) const noexcept {
            return hash_value(g);
        }
    };
    
    template<>
    struct hash<ChipIdentifier> {
        std::size_t operator()(const ChipIdentifier &c) const noexcept {
            return hash_value(c);
        }
    };
}

static ChipIdentifier get_chip_identifier_from_umd_chip_id(chip_id_t chip_id) {
    const std::unordered_map<tt::ARCH, std::vector<std::uint16_t>> ubb_bus_ids = {
        {tt::ARCH::WORMHOLE_B0, {0xC0, 0x80, 0x00, 0x40}},
        {tt::ARCH::BLACKHOLE, {0x00, 0x40, 0xC0, 0x80}},
    };
    const auto& cluster = tt::tt_metal::MetalContext::instance().get_cluster();
    const auto& tray_bus_ids = ubb_bus_ids.at(cluster.arch());
    const auto bus_id = cluster.get_bus_id(chip_id);
    auto tray_bus_id_it = std::find(tray_bus_ids.begin(), tray_bus_ids.end(), bus_id & 0xF0);
    if (tray_bus_id_it != tray_bus_ids.end()) {
        auto ubb_asic_id = bus_id & 0x0F;
        return { .id = chip_id, .galaxy_ubb = GalaxyUbbIdentifier{tray_bus_id_it - tray_bus_ids.begin() + 1, ubb_asic_id} };
    }
    return { .id = chip_id, .galaxy_ubb = {} }; // invalid UBB ID if not found
}


/**************************************************************************************************
 Chip Link Identification

 Contains information on link endpoints (i.e., Ethernet cores) and bidirectional links between two
 identifiable cores.
**************************************************************************************************/

struct ChipLinkEndpoint {
    ChipIdentifier chip;
    CoreCoord ethernet_core;
    tt::umd::ethernet_channel_t channel;

    bool operator<(const ChipLinkEndpoint &other) const {
        if (chip != other.chip) {
            return chip < other.chip;
        } else if (ethernet_core != other.ethernet_core) {
            return ethernet_core < other.ethernet_core;
        }
        return channel < other.channel;
    }

    bool operator==(const ChipLinkEndpoint &other) const {
        return chip == other.chip && ethernet_core == other.ethernet_core && channel == other.channel;
    }
};

using ChipLink = std::pair<ChipLinkEndpoint, ChipLinkEndpoint>;

static std::ostream &operator<<(std::ostream &os, const CoreCoord &core) {
    os << core.str();
    return os;
}

static std::ostream &operator<<(std::ostream &os, const ChipLinkEndpoint &ep) {
    os << "<Endpoint " << ep.chip << " Core " << ep.ethernet_core << '>';
    return os;
}

namespace tt { 
    namespace umd {
        static size_t hash_value(const xy_pair &xy) {
            std::size_t seed = 0;
            boost::hash_combine(seed, xy.x);
            boost::hash_combine(seed, xy.y);
            return seed;
        }
    }
}

namespace std {
    template<>
    struct hash<ChipLinkEndpoint> {
        std::size_t operator()(const ChipLinkEndpoint& ep) const noexcept {
            std::size_t seed = 0;
            boost::hash_combine(seed, ep.chip);
            boost::hash_combine(seed, ep.ethernet_core);
            boost::hash_combine(seed, ep.channel);
            return seed;
        }
    };
}


/**************************************************************************************************
 Cluster Type
**************************************************************************************************/

static std::ostream &operator<<(std::ostream &os, const tt::tt_metal::ClusterType cluster_type) {
    switch (cluster_type) {
    case tt::tt_metal::ClusterType::INVALID:    os << "Invalid"; break;
    case tt::tt_metal::ClusterType::N150:       os << "N150"; break;
    case tt::tt_metal::ClusterType::N300:       os << "N300"; break;
    case tt::tt_metal::ClusterType::T3K:        os << "T3K"; break;
    case tt::tt_metal::ClusterType::GALAXY:     os << "Galaxy"; break;
    case tt::tt_metal::ClusterType::TG:         os << "TG"; break;
    case tt::tt_metal::ClusterType::P100:       os << "P100"; break;
    case tt::tt_metal::ClusterType::P150:       os << "P150"; break;
    case tt::tt_metal::ClusterType::P150_X2:    os << "P150 x2"; break;
    case tt::tt_metal::ClusterType::P150_X4:    os << "P150 x4"; break;
    case tt::tt_metal::ClusterType::SIMULATOR_WORMHOLE_B0: os << "Simulator Blackhole B0"; break;
    case tt::tt_metal::ClusterType::SIMULATOR_BLACKHOLE:   os << "Simulator Blackhole"; break;
    case tt::tt_metal::ClusterType::N300_2x2:   os << "N300 2x2"; break;
    default:
        os << "Unknown (" << int(cluster_type) << ')';
        break;
    }
    return os;
}


/**************************************************************************************************
 Link Status App
**************************************************************************************************/

std::map<tt::umd::ethernet_channel_t, CoreCoord> map_ethernet_channel_to_core_coord(const tt::Cluster &cluster, tt::umd::chip_id_t chip_id) {
    std::map<tt::umd::ethernet_channel_t, CoreCoord> ethernet_channel_to_core_coord;
    for (const auto &[core, channel]: cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        //TODO: assert channel not already in map
        ethernet_channel_to_core_coord.insert({ channel, core });
    }
    return ethernet_channel_to_core_coord;
}

// TODO: using get_ethernet_connections() in here. This will only map links on the same host. We will want a more generic scheme eventually for remote ones.
std::unordered_map<ChipLinkEndpoint, ChipLinkEndpoint> map_endpoints_to_remote_endpoints(const tt::Cluster &cluster) {
    std::unordered_map<ChipLinkEndpoint, ChipLinkEndpoint> endpoint_to_remote;

    for (const auto &[chip_id, remote_chip_and_channel_by_channel]: cluster.get_ethernet_connections()) {
        ChipIdentifier from_chip = get_chip_identifier_from_umd_chip_id(chip_id);
        
        auto ethernet_channel_to_core_coord = map_ethernet_channel_to_core_coord(cluster, chip_id);
        
        for (const auto &[channel, remote_chip_and_channel]: remote_chip_and_channel_by_channel) {
            //TODO: assert that channel is in map
            CoreCoord from_ethernet_core = ethernet_channel_to_core_coord[channel];
            ChipLinkEndpoint from{ .chip = from_chip, .ethernet_core = from_ethernet_core, .channel = channel };

            tt::umd::chip_id_t remote_chip_id;
            tt::umd::ethernet_channel_t remote_channel;
            std::tie(remote_chip_id, remote_channel) = remote_chip_and_channel;

            ChipIdentifier to_chip = get_chip_identifier_from_umd_chip_id(remote_chip_id);

            auto remote_ethernet_channel_to_core_coord = map_ethernet_channel_to_core_coord(cluster, remote_chip_id);

            //TODO: assert channel is in map
            CoreCoord remote_ethernet_core = remote_ethernet_channel_to_core_coord[remote_channel];
            ChipLinkEndpoint to{ .chip = to_chip, .ethernet_core = remote_ethernet_core, .channel = remote_channel };

            endpoint_to_remote[from] = to;
        }
    }

    return endpoint_to_remote;
}

auto make_ordered_ethernet_connections(const auto &unordered_connections) {
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
    
static bool is_ethernet_link_up(const tt::Cluster &cluster, tt::umd::chip_id_t chip_id, tt::umd::ethernet_channel_t ethernet_channel) {
    for (const auto &[core, channel]: cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        if (ethernet_channel == channel) {
            // Found the channel on the chip we are interested in, we now have the core coordinates
            return cluster.is_ethernet_link_up(chip_id, core);
        }
    }

    // Invalid chip ID or channel -> not connected
    return false;
}

int main() {
    const tt::tt_metal::MetalContext &instance = tt::tt_metal::MetalContext::instance();
    const tt::Cluster &cluster = instance.get_cluster();
    const std::map<
        tt::umd::chip_id_t, 
        std::map<
            tt::umd::ethernet_channel_t, 
            std::tuple<tt::umd::chip_id_t, tt::umd::ethernet_channel_t>
        >
    > ethernet_connections = make_ordered_ethernet_connections(cluster.get_ethernet_connections());
    tt::tt_metal::ClusterType cluster_type = cluster.get_cluster_type();

    std::cout << "Cluster Type: " << cluster_type << std::endl;

    for (const auto &[chip_id, remote_chip_and_channel_by_channel]: ethernet_connections) {
        std::cout << get_chip_identifier_from_umd_chip_id(chip_id) << std::endl;
        for (const auto &[channel, remote_chip_and_channel]: remote_chip_and_channel_by_channel) {
            tt::umd::chip_id_t remote_chip_id;
            tt::umd::ethernet_channel_t remote_channel;
            std::tie(remote_chip_id, remote_channel) = remote_chip_and_channel;
            std::cout << "  Channel " << channel << " -> [" << get_chip_identifier_from_umd_chip_id(remote_chip_id) 
                      << "], Channel " << remote_channel 
                      << " (Link Status: " << (is_ethernet_link_up(cluster, chip_id, channel) ? "UP" : "DOWN") << '/' << (is_ethernet_link_up(cluster, remote_chip_id, remote_channel) ? "UP" : "DOWN") <<  ')'
                      << std::endl;
        }
    }

    std::cout << std::endl;
    std::cout << "Links:" << std::endl;

    std::unordered_map<ChipLinkEndpoint, ChipLinkEndpoint> endpoint_to_remote = map_endpoints_to_remote_endpoints(cluster);
    
    for (const auto &[from, to]: endpoint_to_remote) {
        std::cout << from << " -> " << to << std::endl;
    }

    return 0;
}