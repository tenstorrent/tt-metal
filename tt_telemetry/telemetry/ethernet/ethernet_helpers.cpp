#include "impl/context/metal_context.hpp"
#include <tt-metalium/hal_types.hpp>

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
get_ordered_ethernet_connections(const tt::Cluster& cluster) {
    return make_ordered_ethernet_connections(cluster.get_ethernet_connections());
}

std::unordered_map<tt::umd::ethernet_channel_t, CoreCoord> map_ethernet_channel_to_core_coord(const tt::Cluster &cluster, tt::umd::chip_id_t chip_id) {
    // logical_eth_core_to_chan_map should be a 1:1 mapping and therefore easily invertible
    std::unordered_map<tt::umd::ethernet_channel_t, CoreCoord> ethernet_channel_to_core_coord;
    for (const auto &[core, channel]: cluster.get_soc_desc(chip_id).logical_eth_core_to_chan_map) {
        TT_ASSERT(ethernet_channel_to_core_coord.count(channel) == 0, "Multiple Ethernet cores in logical_eth_core_to_chan_map map to Ethernet channel {}", channel);
        ethernet_channel_to_core_coord.insert({ channel, core });
    }
    return ethernet_channel_to_core_coord;
}

bool is_ethernet_endpoint_up(const tt::Cluster &cluster, const EthernetEndpoint &ep) {
    tt_cxy_pair virtual_eth_core = tt_cxy_pair(ep.chip.id, cluster.get_virtual_coordinate_from_logical_coordinates(ep.chip.id, ep.ethernet_core, CoreType::ETH));
    uint32_t link_up_addr = tt::tt_metal::MetalContext::instance().hal().get_dev_addr(tt::tt_metal::HalProgrammableCoreType::ACTIVE_ETH, tt::tt_metal::HalL1MemAddrType::LINK_UP);
    uint32_t link_up_value = 0;
    cluster.read_core(&link_up_value, sizeof(uint32_t), virtual_eth_core, link_up_addr);
    if (cluster.arch() == tt::ARCH::WORMHOLE_B0) {
        return link_up_value == 6;  // see eth_fw_api.h
    } else if (cluster.arch() == tt::ARCH::BLACKHOLE) {
        return link_up_value == 1;
    }
    TT_ASSERT(false, "Unsupported architecture for chip {}", ep.chip);
    return false;
}
