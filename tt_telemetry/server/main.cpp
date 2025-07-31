#include <iostream>

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/control_plane.hpp>
#include <tt-metalium/mesh_graph.hpp>
#include "impl/context/metal_context.hpp"


/**************************************************************************************************
 Galaxy UBB ID

 Identifies the position of a chip in a Galaxy system. This is only valid for Galaxy boxes.
**************************************************************************************************/

struct GalaxyUbbId {
    std::uint32_t tray_id;
    std::uint32_t asic_id;
}; 

static std::ostream &operator<<(std::ostream &os, const GalaxyUbbId &ubb_id) {
    os << "[Tray " << ubb_id.tray_id << " N" << ubb_id.asic_id << ']';
    return os;
}

static GalaxyUbbId get_galaxy_ubb_id(chip_id_t chip_id) {
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
        return GalaxyUbbId{tray_bus_id_it - tray_bus_ids.begin() + 1, ubb_asic_id};
    }
    return GalaxyUbbId{0, 0};  // Invalid UBB ID if not found
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
        std::cout << "Chip " << chip_id << ' ' << get_galaxy_ubb_id(chip_id) << std::endl;
        for (const auto &[channel, remote_chip_and_channel]: remote_chip_and_channel_by_channel) {
            tt::umd::chip_id_t remote_chip_id;
            tt::umd::ethernet_channel_t remote_channel;
            std::tie(remote_chip_id, remote_channel) = remote_chip_and_channel;
            std::cout << "  Channel " << channel << " -> Chip " << remote_chip_id << ' ' << get_galaxy_ubb_id(remote_chip_id) << ", Channel " << remote_channel << std::endl;
        }
    }
    return 0;
}